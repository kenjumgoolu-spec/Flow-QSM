import os
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Optional, Tuple, Union, List
from itertools import product
from utils.utils import visualize_3d_patch, MedicalTransform, Random3DFlip, Random3DRotation

class NiiPatchDataset(Dataset):
    def __init__(self,
                 file_list: list,
                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                 sample_overlap: int = 0,
                 normalize: bool = True,
                 blank_threshold: float = 0.05,
                 max_attempts: int = 100,
                 transform: Optional[callable] = None,
                 mode: str = 'random',
                 stride: Optional[Union[int, Tuple[int, int, int]]] = None):
        super().__init__()
        self.file_list = file_list
        self.patch_size = patch_size
        self.sample_overlap = self._parse_overlap(sample_overlap)
        self.normalize = normalize
        self.blank_threshold = blank_threshold
        self.max_attempts = max_attempts
        self.transform = transform
        self.mode = mode
        if mode == 'random_fixed_count':
            if stride is None:
                raise ValueError("mode")
            self.stride = self._parse_stride(stride)
            self.patch_metadata = []  
            volumes = []
            for idx, path in enumerate(self.file_list):
                vol = self._load_nii(path)  
                volumes.append(vol)
                vol_shape = vol.shape
                num_needed = self._compute_num_patches(vol_shape)
                valid_starts = self._collect_valid_patches_random(vol, num_needed)
                for start in valid_starts:
                    self.patch_metadata.append((idx, start))
            self.num_patches = len(self.patch_metadata)

        elif mode == 'grid':
            self.stride = self._parse_stride(stride) if stride is not None else patch_size
            self.patch_metadata = []
            for idx, path in enumerate(self.file_list):
                vol = self._load_nii(path)
                vol_shape = vol.shape
                all_starts = self._compute_grid_starts(vol_shape)
                valid_starts = self._filter_valid_patches(vol, all_starts)
                for start in valid_starts:
                    self.patch_metadata.append((idx, start))
            self.num_patches = len(self.patch_metadata)

        else:  
            pass

    def _collect_valid_patches_random(self, volume: np.ndarray, num_needed: int) -> List[Tuple[int, int, int]]:
        valid_starts = []
        attempts = 0
        max_attempts = self.max_attempts * num_needed  # 允许更多尝试
        while len(valid_starts) < num_needed and attempts < max_attempts:
            start = self._get_random_start(volume.shape)
            if self.blank_threshold <= 0:
                # 不过滤，直接加入
                valid_starts.append(start)
            else:
                patch = self._extract_patch(volume, start)
                valid_ratio = np.mean(patch != 0)
                if valid_ratio >= self.blank_threshold:
                    valid_starts.append(start)
            attempts += 1
        if len(valid_starts) < num_needed:
            print(f"Warning: Could only collect {len(valid_starts)} valid patches out of {num_needed} needed.")
        return valid_starts

    def _filter_valid_patches(self, volume: np.ndarray, starts: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:

        if self.blank_threshold <= 0:
            return starts  # 不过滤
        valid = []
        for start in starts:
            patch = self._extract_patch(volume, start)
            if np.mean(patch != 0) >= self.blank_threshold:
                valid.append(start)
        return valid

    def __len__(self) -> int:
        if self.mode in ('random_fixed_count', 'grid'):
            return self.num_patches
        else:
            return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        if self.mode in ('random_fixed_count', 'grid'):
            file_idx, start = self.patch_metadata[idx]
            volume = self._load_nii(self.file_list[file_idx])
            patch = self._extract_patch(volume, start)
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
            if self.transform:
                patch_tensor = self.transform(patch_tensor)

            grid_coord = self._get_combined_coords(start, volume.shape)
            return {"patch": patch_tensor, "pos": torch.FloatTensor(grid_coord)}

        else:  # mode == 'random'（原有随机采样）
            file_path = self.file_list[idx]
            volume = self._load_nii(file_path)
            vol_shape = volume.shape

            patch, start_coords = self._sample_valid_patch(volume)

            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0)
            if self.transform:
                patch_tensor = self.transform(patch_tensor)

            grid_coord = self._get_combined_coords(start_coords, vol_shape)
            return {"patch": patch_tensor, "pos": torch.FloatTensor(grid_coord)}

    def _compute_num_patches(self, vol_shape: Tuple[int, int, int]) -> int:
        num = 1
        for i in range(3):
            dim_steps = max(1, vol_shape[i] // self.stride[i])
            num *= dim_steps
        return num

    def _compute_grid_starts(self, vol_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        starts_per_dim = []
        for dim in range(3):
            if vol_shape[dim] < self.patch_size[dim]:
                print(f"Warning: Volume shape {vol_shape} smaller than patch size {self.patch_size} in dim {dim}.")
                starts_per_dim.append([0])
                continue
            max_start = vol_shape[dim] - self.patch_size[dim]
            cur_start = 0
            dim_starts = []
            while cur_start <= max_start:
                dim_starts.append(cur_start)
                cur_start += self.stride[dim]
            starts_per_dim.append(dim_starts)
        return list(product(*starts_per_dim))

    def _parse_stride(self, stride: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        if isinstance(stride, int):
            return (stride,) * 3
        if len(stride) == 3:
            return tuple(stride)
        raise ValueError("stride")

    def _parse_overlap(self, overlap) -> Tuple[int, int, int]:
        if isinstance(overlap, int):
            return (overlap,) * 3
        if len(overlap) == 3:
            return tuple(overlap)
        raise ValueError("sample_overlap")

    def _load_nii(self, path: str) -> np.ndarray:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        if self.normalize:
            data = 2 * (data - data.min()) / (data.max() - data.min() + 1e-8) - 1.0
        return data

    def _sample_valid_patch(self, volume: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        for _ in range(self.max_attempts):
            start = self._get_random_start(volume.shape)
            patch = self._extract_patch(volume, start)
            valid_ratio = np.mean(patch != 0)
            if valid_ratio >= self.blank_threshold:
                return patch, start
        print(f"Warning: Failed to find valid patch after {self.max_attempts} attempts")
        return patch, start

    def _get_random_start(self, vol_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(
            np.random.randint(0, max(1, vol_shape[i] - self.patch_size[i]))
            for i in range(3)
        )

    def _extract_patch(self, volume: np.ndarray, start: Tuple[int, int, int]) -> np.ndarray:
        slices = tuple(slice(s, s + self.patch_size[i]) for i, s in enumerate(start))
        return volume[slices]

    def _get_combined_coords(self, start: Tuple[int, int, int], vol_shape: Tuple[int, int, int]) -> Tuple[float, ...]:
        rel_coords = self._get_relative_grid(start)
        abs_coords = []
        for i in range(3):
            max_pos = vol_shape[i] - self.patch_size[i]
            if max_pos <= 0:
                abs_coords.append(0.0)
            else:
                abs_coords.append(start[i] / max_pos)
        return (*rel_coords, *abs_coords)

    def _get_relative_grid(self, start: Tuple[int, int, int]) -> Tuple[float, ...]:
        step = [self.patch_size[i] - self.sample_overlap[i] for i in range(3)]
        return tuple(
            (start[i] % step[i]) / step[i] if step[i] != 0 else 0.0
            for i in range(3)
        )

    @classmethod
    def from_config(cls, config_path: str) -> "NiiPatchDataset":
        config_path = Path(config_path).expanduser()
        with open(config_path, 'r') as f:
            config = json.load(f)['data']

        file_list = [str(Path(p).expanduser()) for p in config['data_file_path']]
        transform = cls._build_transform(config.get('transform_params'))

        mode = config.get('mode', 'random')
        stride = config.get('stride', None)

        return cls(
            file_list=file_list,
            patch_size=tuple(config['patch_size']),
            sample_overlap=config['sample_overlap'],
            normalize=config['normalize'],
            blank_threshold=config['blank_threshold'],
            max_attempts=config['max_attempts'],
            transform=transform,
            mode=mode,
            stride=stride
        )

    @staticmethod
    def _build_transform(params: dict) -> Optional[MedicalTransform]:
        if not params:
            return None

        class CustomTransform(MedicalTransform):
            def __init__(self, params):
                super().__init__()
                self.spatial_transform = transforms.Compose([
                    Random3DFlip(p=params.get('flip_prob', 0.5)),
                    Random3DRotation(p=params.get('rotation_prob', 0.5))
                ])

        return CustomTransform(params) if params else None

