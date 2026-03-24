import logging
from typing import List, Optional, Tuple, Union
import torch
import json

logger = logging.getLogger(__name__) 
def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

import importlib
from typing import Dict, Any
import inspect

def load_scheduler(config_path) -> Any:
    
    with open(config_path, "r") as f:
        config = json.load(f)
    class_path = config["scheduler_type"].split(".")
    module_name = ".".join(class_path[:-1])
    class_name = class_path[-1]

    module = importlib.import_module(module_name)
    scheduler_class = getattr(module, class_name)

    init_params = inspect.signature(scheduler_class.__init__).parameters
    valid_params = {}
    for param_name in init_params:
        if param_name == "self":
            continue
        if param_name in config["params"]:
            valid_params[param_name] = config["params"][param_name]
        else:
            param_default = init_params[param_name].default
            if param_default is not inspect.Parameter.empty:
                valid_params[param_name] = param_default

    return scheduler_class(**valid_params)

def calc_d2_matrix1(local_phase, B0_dir,pix_dim):
        shape, pix_dim = torch.tensor(local_phase.shape), torch.tensor([1.0, 1.0, 1.0])
        field_of_view = shape[-3:] * pix_dim
        rx, ry, rz = torch.meshgrid(torch.arange(-shape[-3] // 2, shape[-3] // 2 ,device=local_phase.device),
                                    torch.arange(-shape[-2] // 2, shape[-2] // 2 ,device=local_phase.device),
                                    torch.arange(-shape[-1] // 2, shape[-1] // 2 ,device=local_phase.device))
        rx, ry, rz = rx / field_of_view[0], ry / field_of_view[1], rz / field_of_view[2]
        sq_dist = rx ** 2 + ry ** 2 + rz ** 2
        sq_dist[sq_dist == 0] = 1e6
        d2 = ((B0_dir[0] * rx + B0_dir[1] * ry + B0_dir[2] * rz) ** 2) / sq_dist
        d2 = (1 / 3 - d2)
        d2[d2.shape[0] // 2, d2.shape[1] // 2, d2.shape[2] // 2] = 0
        d2 = d2[None, :, :, :].to(local_phase.device)
        return d2


def inverse_operator(data,B0_dir,pix_dim):
        B0_dir=torch.from_numpy(B0_dir).to(data.device)
        B,C,D,H,W=data.shape
        data=data.view(B*C,D,H,W).contiguous()
        state1 = torch.fft.ifftshift(
            torch.fft.fftn(torch.fft.fftshift(data, dim=(1, 2, 3)), dim=(1, 2, 3), norm="ortho"),
            dim=(1, 2, 3))
        kernel=calc_d2_matrix1(data,B0_dir,pix_dim)
        state3 = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kernel*state1, dim=(1,2,3)), dim=(1, 2, 3), norm="ortho"),
            dim=(1, 2, 3))

        state3=state3.reshape(B,C,D,H,W).contiguous()
        return state3.real
    

def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, torch.nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
    
    
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)