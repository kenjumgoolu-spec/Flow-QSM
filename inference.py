import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import nibabel as nib
from typing import List, Tuple, Dict, Any
import os
from pipeline import QsmRecPipeline
from modules.config import load_model
from schedulers._utils import load_scheduler
import re
import json
import time

class QsmInferenceDataset(Dataset):    
    def __init__(self, input_files: List[str], mask_files: List[str], B0_dirs: List[List[float]], 
                 pix_dims: List[List[float]], types: List[str], subject_ids: List[str]):
  
        assert len(input_files) == len(mask_files) == len(B0_dirs) == len(pix_dims) == len(types) == len(subject_ids), \
            "error"
        
        self.input_files = input_files
        self.mask_files = mask_files
        self.B0_dirs = B0_dirs
        self.pix_dims = pix_dims
        self.types = types
        self.subject_ids = subject_ids

        self.file_info = []
        for i in range(len(input_files)):
            self.file_info.append({
                'input_file': input_files[i],
                'mask_file': mask_files[i],
                'B0_dir': B0_dirs[i],
                'pix_dim': pix_dims[i],
                'type': types[i],
                'subject_id': subject_ids[i]
            })
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        info = self.file_info[idx]

        return {
            'idx': idx,
            'input_file': info['input_file'],
            'mask_file': info['mask_file'], 
            'B0_dir': np.array(info['B0_dir'], dtype=np.float32),
            'pix_dim': np.array(info['pix_dim'], dtype=np.float32),
            'type': info['type'],
            'subject_id': info['subject_id']
        }


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def extract_subject_id(file_path: str) -> str:
    match = re.search(r'(Subject\d+|Sub\d+)', file_path)
    if match:
        return match.group(1)
    return os.path.splitext(os.path.basename(file_path))[0]


def load_nifti_to_numpy(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
    return img.get_fdata()


def load_nifti_to_tensor(file_path: str, device: torch.device, subject_id: str) -> torch.Tensor:
    nii_data = load_nifti_to_numpy(file_path)
    if subject_id in ['Sub008', 'Sub009']:
        tensor_data = torch.from_numpy(nii_data / (7 * 42.57)) 
    else:
        tensor_data = torch.from_numpy(nii_data)      
    return tensor_data.to(device)


def initialize_pipeline(args, rank):
    torch.cuda.set_device(rank)
    model = load_model(args.model_config)
    ckpt = torch.load(args.model_checkpoint, map_location="cpu")
    if "module" in ckpt:  
        ckpt = ckpt["module"]
    model.load_state_dict(ckpt, strict=False)
    if torch.cuda.is_available() and args.device == "cuda":
        print(f"=> Using GPU {rank} for inference with DDP")
        model = model.cuda(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)
    else:
        model = model.to(args.device)
    model.eval()
    scheduler = load_scheduler(args.scheduler_config)
    pipeline = QsmRecPipeline(unet=model, scheduler=scheduler)
    return pipeline


def process_batch_distributed(rank, world_size, dataset, args_dict):
    args = argparse.Namespace(**args_dict)
    
    setup(rank, world_size)
    
    try:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)
        
        model_dir = os.path.basename(os.path.dirname(args.model_checkpoint))
        output_dir = os.path.join("./inference_results", model_dir)
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        pipeline = initialize_pipeline(args, rank)
        
        for batch in dataloader:
            start_time=time.time()
            idx = batch['idx'].item()
            input_file = batch['input_file'][0]
            mask_file = batch['mask_file'][0]
            B0_dir = batch['B0_dir'][0].numpy()
            pix_dim = batch['pix_dim'][0].numpy()
            inference_type = batch['type'][0]
            subject_id = batch['subject_id'][0]
            
            print(f"Rank {rank}: Processing {subject_id} from {input_file}")
            local_phase_image = load_nifti_to_tensor(input_file, rank, subject_id)
            mask_data = load_nifti_to_numpy(mask_file)
            four_d_image, result = pipeline(
                num_inference_steps=args.num_inference_steps,
                noise_step=args.noise_step,
                window_size=args.window_size,
                stride=args.stride,
                local_phase_image=local_phase_image,
                B0_dir=B0_dir,
                pix_dim=pix_dim,
                device=rank,
                type=inference_type
            )
            result_np = result.cpu().numpy() if isinstance(result, torch.Tensor) else result
            four_d_result_np = four_d_image.cpu().numpy() if isinstance(four_d_image, torch.Tensor) else four_d_image
            result_img = nib.Nifti1Image(result_np * mask_data, affine=np.eye(4))
            four_d_img = nib.Nifti1Image(four_d_result_np * mask_data[..., np.newaxis], affine=np.eye(4))
            end_time=time.time()
            print(f"full reconstruction time {end_time-start_time}")
            output_filename = f"result_{subject_id}.nii.gz"
            output_filename_ = f"4d_result_{subject_id}.nii.gz"
            nib.save(result_img, os.path.join(output_dir, output_filename))
            nib.save(four_d_img, os.path.join(output_dir, output_filename_))

            print(f"Rank {rank}: Saved result to: {os.path.join(output_dir, output_filename)}")
            
    finally:
        cleanup()


def create_inference_dataset_from_config(config_file: str) -> QsmInferenceDataset:
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    input_files = []
    mask_files = []
    B0_dirs = []
    pix_dims = []
    types = []
    subject_ids = []
    
    for task in config['inference_tasks']:
        input_files.append(task['input_file'])
        mask_files.append(task['mask_file'])
        B0_dirs.append(task.get('B0_dir', [0.0, 0.0, 1.0]))
        pix_dims.append(task.get('pix_dim', [1.0, 1.0, 1.0]))
        types.append(task.get('type', 'epsilon'))
        subject_ids.append(extract_subject_id(task['input_file']))
    return QsmInferenceDataset(input_files, mask_files, B0_dirs, pix_dims, types, subject_ids)


def create_inference_dataset_from_args(args) -> QsmInferenceDataset:
    if hasattr(args, 'config_file') and args.config_file:
        return create_inference_dataset_from_config(args.config_file)
    input_files = [args.input]
    mask_files = [args.mask]
    B0_dirs = [args.B0_dir]
    pix_dims = [args.pix_dim]
    types = [args.type]
    subject_ids = [extract_subject_id(args.input)]  
    return QsmInferenceDataset(input_files, mask_files, B0_dirs, pix_dims, types, subject_ids)


def process_files_distributed(args):
    world_size = torch.cuda.device_count()
    dataset = create_inference_dataset_from_args(args)
    print(f"Starting distributed inference with {world_size} GPUs for {len(dataset)} tasks")
    args_dict = vars(args)
    torch.multiprocessing.spawn(
        process_batch_distributed,
        args=(world_size, dataset, args_dict),
        nprocs=world_size,
        join=True
    )


def process_files_single(args):
    dataset = create_inference_dataset_from_args(args)
    
    model_dir = os.path.basename(os.path.dirname(args.model_checkpoint))
    output_dir = os.path.join("./inference_results", model_dir)
    os.makedirs(output_dir, exist_ok=True)
    pipeline = initialize_pipeline(args, 0)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        print(f"Processing {sample['subject_id']} from {sample['input_file']}")
        local_phase_image = load_nifti_to_tensor(sample['input_file'], args.device, sample['subject_id'])
        mask_data = load_nifti_to_numpy(sample['mask_file'])
        four_d_image, result = pipeline(
            num_inference_steps=args.num_inference_steps,
            noise_step=args.noise_step,
            window_size=args.window_size,
            stride=args.stride,
            local_phase_image=local_phase_image,
            B0_dir=sample['B0_dir'],
            pix_dim=sample['pix_dim'],
            device=args.device,
            type=sample['type']
        )
        
        result_np = result.cpu().numpy() if isinstance(result, torch.Tensor) else result
        four_d_result_np = four_d_image.cpu().numpy() if isinstance(four_d_image, torch.Tensor) else four_d_image
        result_img = nib.Nifti1Image(result_np * mask_data, affine=np.eye(4))
        four_d_img = nib.Nifti1Image(four_d_result_np * mask_data[..., np.newaxis], affine=np.eye(4))
        output_filename = f"result_{sample['subject_id']}.nii.gz"
        output_filename_ = f"4d_result_{sample['subject_id']}.nii.gz"
        nib.save(result_img, os.path.join(output_dir, output_filename))
        nib.save(four_d_img, os.path.join(output_dir, output_filename_))
        print(f"Saved result to: {os.path.join(output_dir, output_filename)}")


def get_args():
    parser = argparse.ArgumentParser(description="Process NIfTI files with QSM reconstruction pipeline.")
    parser.add_argument("--input", "-i", type=str, help="Input NIfTI file path (single file mode)")
    parser.add_argument("--mask", "-m", type=str, help="Mask NIfTI file path (single file mode)")
    parser.add_argument("--config-file", type=str, help="JSON config file containing multiple inference tasks")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--scheduler_config", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--noise_step", type=int, default=125)
    parser.add_argument("--window_size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--stride", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--B0_dir", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--pix_dim", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--type", type=str, default="epsilon")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training with multiple GPUs")
    args = parser.parse_args()
    if not args.config_file and (not args.input or not args.mask):
        parser.error("Either --config-file or both --input and --mask must be provided")
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    if args.distributed and torch.cuda.device_count() > 1:
        print(f"Using distributed inference with {torch.cuda.device_count()} GPUs")
        process_files_distributed(args)
    else:
        if args.distributed:
            print(f"Warning: Only {torch.cuda.device_count()} GPUs available, using single GPU mode")
        print("Using single GPU inference")
        process_files_single(args)