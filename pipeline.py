import os
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
from typing import Tuple
from schedulers._utils import load_scheduler, inverse_operator, RequiresGradContext
from utils.utils import visualize_3d_patch
from modules.config import load_model
from scipy.ndimage import sobel


def _padded_image(image, window_size, stride, padding_mode="constant", device="cpu"):
    is_tensor = isinstance(image, torch.Tensor)
    original_device = image.device if is_tensor else None
    if is_tensor:
        if image.is_cuda:
            image_np = image.cpu().numpy()
        else:
            image_np = image.numpy()
    else:
        image_np = image
    pad_dims = []
    for dim in range(3):
        if (image_np.shape[dim] - window_size[dim]) % stride[dim] != 0:
            pad = stride[dim] - (image_np.shape[dim] - window_size[dim]) % stride[dim]
        else:
            pad = 0
        pad_dims.append((0, pad))
    padded_image = np.pad(image_np, pad_width=tuple(pad_dims), mode=padding_mode)
    if is_tensor:
        padded_tensor = torch.from_numpy(padded_image).to(original_device)
        return padded_tensor, pad_dims
    else:
        return padded_image, pad_dims


def _unpad_image(padded_image, pad_dims):
    is_tensor = isinstance(padded_image, torch.Tensor)
    original_device = padded_image.device if is_tensor else None
    
    if is_tensor:
        if padded_image.is_cuda:
            padded_np = padded_image.cpu().numpy()
        else:
            padded_np = padded_image.numpy()
    else:
        padded_np = padded_image
    slices = []
    for front_pad, back_pad in pad_dims:
        if back_pad == 0:
            slices.append(slice(None))
        else:
            slices.append(slice(front_pad, -back_pad if back_pad != 0 else None))
    unpadded = padded_np[tuple(slices)]
    if is_tensor:
        return torch.from_numpy(unpadded).to(original_device)
    else:
        return unpadded


def _get_relative_grid(stride, start: Tuple[int, int, int]) -> Tuple[float, ...]:
    step = [stride[i] for i in range(3)]
    return tuple((start[i] % step[i]) / step[i] for i in range(3))


def _get_combined_coords(window, stride, start, vol_shape) -> Tuple[float, ...]:
    rel_coords = _get_relative_grid(start=start, stride=stride)
    abs_coords = []
    for i in range(3):
        max_pos = vol_shape[i] - window[i]
        abs_coords.append(start[i] / max_pos if max_pos > 0 else 0.0)
    return (*rel_coords, *abs_coords)


def _sliding_window(
    image: np.ndarray, window_size: Tuple[int, int, int], stride: Tuple[int, int, int]
):
    is_tensor = isinstance(image, torch.Tensor)
    original_device = image.device if is_tensor else None
    
    if is_tensor:
        if image.is_cuda:
            image_np = image.cpu().numpy()
        else:
            image_np = image.numpy()
    else:
        image_np = image
    
    patches, positions, positions_emb = [], [], []
    z_steps = (image_np.shape[0] - window_size[0]) // stride[0] + 1
    y_steps = (image_np.shape[1] - window_size[1]) // stride[1] + 1
    x_steps = (image_np.shape[2] - window_size[2]) // stride[2] + 1
    
    for z in range(z_steps):
        for y in range(y_steps):
            for x in range(x_steps):
                z_start, y_start, x_start = z * stride[0], y * stride[1], x * stride[2]
                patch = image_np[
                    z_start : z_start + window_size[0],
                    y_start : y_start + window_size[1],
                    x_start : x_start + window_size[2],
                ]
                patches.append(patch)
                positions_emb.append(
                    _get_combined_coords(
                        window_size, stride, [z_start, y_start, x_start], image_np.shape
                    )
                )
                positions.append((z_start, y_start, x_start))
    if is_tensor:
        patches_tensor = [torch.from_numpy(patch).to(original_device) for patch in patches]
        return patches_tensor, positions, positions_emb
    else:
        return np.array(patches), positions, positions_emb


def gradient_weight(patch):
    is_tensor = isinstance(patch, torch.Tensor)
    if is_tensor:
        if patch.is_cuda:
            patch_np = patch.cpu().numpy()
        else:
            patch_np = patch.numpy()
    else:
        patch_np = patch
    
    gx, gy, gz = sobel(patch_np, axis=0), sobel(patch_np, axis=1), sobel(patch_np, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    weight = 1 / (1 + grad_mag)
    normalized_weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
    if is_tensor:
        return torch.from_numpy(normalized_weight).to(patch.device)
    else:
        return normalized_weight


def _merge_patches(output_patches, positions, target_shape, window_size, device="cpu"):
    is_tensor = isinstance(output_patches[0], torch.Tensor)
    if is_tensor:
        merged = torch.zeros(target_shape, device=device)
        weight_sum = torch.zeros(target_shape, device=device)
        for patch, (z, y, x) in zip(output_patches, positions):
            merged[
                z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]
            ] += patch
            weight_sum[
                z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]
            ] += 1
        weight_sum[weight_sum == 0] = 1
        return merged / weight_sum
    else:
        merged = np.zeros(target_shape)
        weight_sum = np.zeros(target_shape)
        
        for patch, (z, y, x) in zip(output_patches, positions):
            merged[
                z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]
            ] += patch
            weight_sum[
                z : z + window_size[0], y : y + window_size[1], x : x + window_size[2]
            ] += 1
        
        weight_sum[weight_sum == 0] = 1
        return merged / weight_sum


class QsmRecPipeline:
    def __init__(self, unet: torch.nn.Module, scheduler):
        self.unet = unet
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps=1000,mu=1.0)
    def _prepare_model_input(self, image, sample, add_noise_step, window_size, stride, device):
        if sample is None and image is None:
            raise ValueError("sample and image can not be None at same time")
        # print("image range",image.max(), image.min())
        if image is not None:
            _sample,_ = self.scheduler.q_sample(x_start=image, noise=None, t=add_noise_step)
            _sample = _sample.squeeze()
            patches, positions, position_emb = _sliding_window(_sample, window_size, stride)
        else:
            _sample = sample.squeeze()
            patches, positions, position_emb = _sliding_window(_sample, window_size, stride)

        return patches, positions, position_emb, _sample

    def _calculate_posterior_prior(self, t_tensor, x_cur, model_output, measurement, B0_dir, pix_dim, type="x_0"):

        if type == "flow":
            x_0_hat = self.scheduler.pred_x0_from_noised_img(
                t=t_tensor, noised_img=x_cur, model_output=model_output
            ).detach().clone()
        
        elif type == "x_0":
            x_0_hat = model_output.detach().clone()
        else:
            raise ValueError("type must be 'x_0' or 'epsilon'")
        x_0_hat.requires_grad_(True)
        inverse = inverse_operator(x_0_hat, B0_dir, pix_dim)
        difference = inverse - measurement
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(norm, x_0_hat, retain_graph=False)[0]

        return norm_grad

    def __call__(
        self,
        num_inference_steps=25,
        noise_step=50,
        window_size=(64, 64, 64),
        stride=(32, 32, 32),
        local_phase_image=None,
        B0_dir=np.array([0.0, 0.0, 1.0]),
        pix_dim=np.array([1.0, 1.0, 1.0]),
        type="x_0",
        device="cuda",
    ):

        ts = (torch.linspace(int(noise_step) + 1, 1, num_inference_steps + 1).to(device).to(torch.long))
        local_phase_image, padims = _padded_image(local_phase_image, window_size, stride, padding_mode="edge", device=device)
        if not isinstance(local_phase_image, torch.Tensor):
            local_phase_image = torch.from_numpy(local_phase_image)
        local_phase_image = local_phase_image[None, None, :, :, :].float().to(device)
        sample = None
        step_results = []
        with tqdm(total=num_inference_steps, desc="Denoising...", colour="#00ff00", ncols=80, leave=False,) as pbar:
            for i in range(1, num_inference_steps + 1):
                cur_t = ts[i - 1] - 1
                prev_t = ts[i] - 1
                t = ts[i]
                output_patches = []
                t_tensor = torch.tensor([cur_t] * local_phase_image.shape[0], dtype=torch.long).to(device)
                patches, positions, position_emb, _sample = self._prepare_model_input(
                    image=local_phase_image if sample is None else None,
                    add_noise_step=torch.tensor((noise_step,), device=device),
                    window_size=window_size,
                    stride=stride,
                    sample=sample if sample is not None else None,
                    device=device,
                )
                for patch, emb in zip(patches, position_emb):
                    with torch.no_grad():
                        if not isinstance(patch, torch.Tensor):
                            patch_tensor = torch.from_numpy(patch).float().to(device)
                        else:
                            patch_tensor = patch.float().to(device)

                        input_tensor = patch_tensor.unsqueeze(0).unsqueeze(0)
                        emb = torch.from_numpy(np.array(emb)).to(device).float().unsqueeze(0)
                        timesteps = t.expand(input_tensor.shape[0]).long()
                        output = self.unet(x=input_tensor, timesteps=timesteps, position_labels=emb)

                        output_patches.append(output.squeeze().cpu().numpy())
                merged_output = _merge_patches(
                    output_patches, positions, _sample.shape, window_size, device=device
                )

                if not isinstance(merged_output, torch.Tensor):
                    model_output = torch.from_numpy(merged_output).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    model_output = merged_output.unsqueeze(0).unsqueeze(0).to(device)

                if not isinstance(_sample, torch.Tensor):
                    _sample_tensor = torch.from_numpy(_sample).unsqueeze(0).unsqueeze(0).to(device)
                else:
                    _sample_tensor = _sample.unsqueeze(0).unsqueeze(0).to(device)
                
                posterior_evaluation = self._calculate_posterior_prior(
                    x_cur=_sample_tensor,
                    measurement=local_phase_image,
                    B0_dir=B0_dir,
                    pix_dim=pix_dim,
                    t_tensor=t_tensor,
                    model_output=model_output,
                    type=type,
                )
                sample = self.scheduler.step_pred(
                    model_output=model_output,
                    cur_t=cur_t,
                    prev_t=prev_t,
                    sample=_sample_tensor,
                    gamma=None,
                    posterior_evaluation=posterior_evaluation,
                    type=type,
                )
                pbar.update(1)
                cur_image = sample.squeeze().detach().cpu().numpy()
                cur_image = _unpad_image(cur_image, padims)
                step_results.append(cur_image)
        step_results = np.stack(step_results, axis=-1)
        return step_results/10, cur_image/10
