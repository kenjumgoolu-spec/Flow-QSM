import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import torch
import torchvision.transforms as transforms
from typing import Optional, Tuple, List
def visualize_3d_patch(patch: Union[np.ndarray, torch.Tensor], 
                       slice_idx: Union[str, Tuple[int, int, int]] = 'middle',
                       title: str = '3D Patch Visualization',
                       save_path: str = None,
                       cmap: str = 'gray',
                       figsize: tuple = (12, 4)):
    """可视化3D医学图像切块 (已修复 eval 变量名错误)"""
    
    if isinstance(patch, torch.Tensor):
        patch = patch.detach().cpu().numpy()
    
    if patch.ndim == 4:
        patch = np.squeeze(patch, axis=0)
    
    d, h, w = patch.shape
    
    # 确定切片位置并防止越界
    if slice_idx == 'middle':
        d_idx, h_idx, w_idx = d // 2, h // 2, w // 2
    elif slice_idx == 'random':
        d_idx, h_idx, w_idx = np.random.randint(0, d), np.random.randint(0, h), np.random.randint(0, w)
    elif isinstance(slice_idx, (tuple, list)):
        d_idx = min(slice_idx[0], d - 1)
        h_idx = min(slice_idx[1], h - 1)
        w_idx = min(slice_idx[2], w - 1)
    else:
        d_idx, h_idx, w_idx = d // 2, h // 2, w // 2

    # 提取切片：Axial(D), Coronal(H), Sagittal(W)
    # 分别对应：第0维固定，第1维固定，第2维固定
    slices = [
        ("Axial", patch[d_idx, :, :], d_idx),
        ("Coronal", patch[:, h_idx, :], h_idx),
        ("Sagittal", patch[:, :, w_idx], w_idx)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14)
    
    for ax, (name, img, real_idx) in zip(axes, slices):
        min_val, max_val = np.min(img), np.max(img)
        # img.T 转置是为了让 H, W 符合人类视觉习惯的纵横比
        im = ax.imshow(img.T, cmap=cmap, origin='lower', aspect='equal')
        
        # 直接使用传入的 real_idx，不再使用 eval
        ax.set_title(f"{name} View (Idx: {real_idx})\nMin: {min_val:.2f} Max: {max_val:.2f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"图像已保存至：{save_path}")
    else:
        plt.show()
    plt.close()

class MedicalTransform:
    """医学图像三维数据增强组合"""
    def __init__(self):
        self.spatial_transform = transforms.Compose([
            Random3DFlip(p=0.5),
            Random3DRotation(p=0.5)
        ])
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        
        if torch.rand(1) < 0.05:  # 5%概率打印形状
            return x
        
        x = self.spatial_transform(x)
        return x

class Random3DFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        
        for dim in [1, 2, 3]:  
            if np.random.rand() < self.p:
                x = torch.flip(x, [dim])
        return x

class Random3DRotation:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.p:
            return x
            
        axes_options = [(1, 2),  
                        (1, 3),  
                        (2, 3)]  
        axes = axes_options[np.random.randint(3)]
        k = np.random.randint(1, 4)
        return torch.rot90(x, k=k, dims=axes)