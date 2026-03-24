import torch
import math
import numpy as np

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://huggingface.co/papers/2403.03206v1.
    """

    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting

class FlowMatchScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=False,
        time_shift_type="exponential",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift_val = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.time_shift_type = time_shift_type
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        sigmas = timesteps / num_train_timesteps

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = torch.from_numpy(timesteps).float()
        self.sigmas = torch.from_numpy(sigmas).float()
        self.inference_sigmas = None

    def set_timesteps(self, num_inference_steps, device="cpu", mu=None):
  
        timesteps = np.linspace(self.num_train_timesteps, 1, num_inference_steps)
        sigmas = timesteps / self.num_train_timesteps

        if self.use_dynamic_shifting:
            if mu is None: 
                raise ValueError
            
            if self.time_shift_type == "exponential":
                sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** 1.0)
            else:
                sigmas = mu / (mu + (1 / sigmas - 1) ** 1.0)
        else:
            sigmas = self.shift_val * sigmas / (1 + (self.shift_val - 1) * sigmas)

        self.inference_sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        self.inference_sigmas = torch.cat([self.inference_sigmas, torch.zeros(1, device=device)])

    def q_sample(self, x_start, t, noise=None):
        
        """
        x_t = (1 - sigma_t) * x_0 + sigma_t * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device)

        sigma_t = extract(self.sigmas, t, x_start.shape).to(x_start.dtype)
        out = (1.0 - sigma_t) * x_start + sigma_t * noise
        weighting=compute_loss_weighting_for_sd3(weighting_scheme="cosmap",sigmas=sigma_t)
        return out,weighting

    def pred_x0_from_noised_img(self, noised_img, model_output, t):

        sigma_t = extract(self.sigmas, t, noised_img.shape)
        x0_pred = noised_img - sigma_t * model_output
        return x0_pred

    def pred_eps_from_x0_img(self, noised_img, model_output, t):

        sigma_t = extract(self.inference_sigmas, t, noised_img.shape)
        eps = noised_img + (1.0 - sigma_t) * model_output
        return eps

    def step_pred(
        self,
        model_output,
        cur_t,
        prev_t,
        sample,
        gamma=0.0,
        posterior_evaluation=0.0,
        type="flow",
    ):

        t_tensor = torch.tensor([cur_t] * sample.shape[0], dtype=torch.long, device=sample.device)
        prev_t_tensor = torch.tensor([prev_t] * sample.shape[0], dtype=torch.long, device=sample.device)
        sigma_cur = extract(self.sigmas, t_tensor, sample.shape)
        sigma_prev = extract(self.sigmas, prev_t_tensor, sample.shape)
        dt = sigma_prev - sigma_cur

        if type == "flow":
            v_pred = model_output
        elif type == "x_0":
            x0_pred = model_output
            v_pred = (sample - x0_pred) / sigma_cur
        else:
            raise ValueError
        norm=torch.linalg.norm(posterior_evaluation)
        if gamma is None:
            gamma = torch.sqrt(torch.tensor(3e7)) * sigma_cur / norm
        
        mean = sample + dt * v_pred - gamma * posterior_evaluation
        
        return mean