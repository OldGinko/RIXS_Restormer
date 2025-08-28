import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import random
from torchvision.transforms.functional import normalize

class RIXSDataset(Dataset):
    """RIXS Dataset for training with HDF5 files.
    
    This dataset loads RIXS data from HDF5 files containing noisy/clean image pairs.
    
    Args:
        opt (dict): Configuration dictionary containing:
            - dataroot_gt (str): Path to HDF5 file with clean images
            - dataroot_lq (str): Path to HDF5 file with noisy images (optional, can be same as gt)
            - gt_size (int): Size of patches to extract for training
            - phase (str): 'train' or 'val'
            - geometric_augs (bool): Whether to apply geometric augmentations
            - mean (list): Mean values for normalization (optional)
            - std (list): Std values for normalization (optional)
    """
    
    def _find_first_dataset(self, h5_group, max_depth=3, current_depth=0):
        """Recursively find the first dataset in an HDF5 group."""
        if current_depth >= max_depth:
            return None
            
        for key in h5_group.keys():
            obj = h5_group[key]
            if isinstance(obj, h5py.Dataset):
                if current_depth == 0:
                    return key
                else:
                    return key  # Return relative path
            elif isinstance(obj, h5py.Group):
                sub_path = self._find_first_dataset(obj, max_depth, current_depth + 1)
                if sub_path is not None:
                    return f"{key}/{sub_path}"
        return None
    
    def __init__(self, opt):
        super(RIXSDataset, self).__init__()
        self.opt = opt
        
        # Training parameters
        self.phase = opt['phase']
        self.gt_size = opt['gt_size']
        self.geometric_augs = opt['geometric_augs']
        
        # Check if we should use real RIXS data or synthetic noise
        self.use_real_rixs_data = opt.get('use_real_rixs_data', False)
        
        # Noise parameters for synthetic noise generation (only used if not using real data)
        if not self.use_real_rixs_data:
            self.sigma_type = opt.get('sigma_type', 'random')
            self.sigma_range = opt.get('sigma_range', [0, 50])

        self.gt_path = opt['dataroot_gt']
        # For real RIXS data, LQ and GT are in the same file by default
        self.lq_path = opt.get('dataroot_lq', self.gt_path if self.use_real_rixs_data else None)

        # Load GT (Ground-Truth) data
        with h5py.File(self.gt_path, 'r') as f:
            if self.use_real_rixs_data:
                # For real RIXS data, use high-count data as ground truth
                if 'hc/data' in f:
                    self.gt_data = f['hc/data'][:]
                    print("Using RIXS high count data as ground truth: hc/data")
                elif 'hc' in f and 'data' in f['hc']:
                    self.gt_data = f['hc']['data'][:]
                    print("Using RIXS high count data as ground truth: hc/data")
                else:
                    raise ValueError(f"No high-count RIXS data found in {self.gt_path}. Expected 'hc/data' structure.")
            else:
                # For synthetic noise, use any available dataset as ground truth
                if 'clean_images' in f and isinstance(f['clean_images'], h5py.Dataset):
                    self.gt_data = f['clean_images'][:]
                elif 'images' in f and isinstance(f['images'], h5py.Dataset):
                    self.gt_data = f['images'][:]
                elif 'data' in f and isinstance(f['data'], h5py.Dataset):
                     self.gt_data = f['data'][:]
                else:
                    # If no standard keys are found, search for the first dataset in the file
                    dataset_path = self._find_first_dataset(f)
                    if dataset_path is None:
                        raise ValueError(f"No datasets found in HDF5 file: {self.gt_path}")
                    self.gt_data = f[dataset_path][:]

        # Load LQ (Low-Quality) data based on configuration
        self.lq_data = None
        if self.use_real_rixs_data and self.lq_path is not None and self.lq_path.lower() != 'none':
            # Load real low-count RIXS data
            with h5py.File(self.lq_path, 'r') as f:
                if 'lc/data' in f:
                    self.lq_data = f['lc/data'][:]
                    print("Using real RIXS low count data: lc/data")
                elif 'lc' in f and 'data' in f['lc']:
                    self.lq_data = f['lc']['data'][:]
                    print("Using real RIXS low count data: lc/data")
                else:
                    raise ValueError(f"No low-count RIXS data found in {self.lq_path}. Expected 'lc/data' structure.")
        elif not self.use_real_rixs_data and self.lq_path is not None and self.lq_path.lower() != 'none':
            # Load synthetic noise data (for compatibility)
            with h5py.File(self.lq_path, 'r') as f:
                if 'noisy_images' in f and isinstance(f['noisy_images'], h5py.Dataset):
                    self.lq_data = f['noisy_images'][:]
                elif 'images' in f and isinstance(f['images'], h5py.Dataset):
                    self.lq_data = f['images'][:]
                elif 'data' in f and isinstance(f['data'], h5py.Dataset):
                    self.lq_data = f['data'][:]
                else:
                    dataset_path = self._find_first_dataset(f)
                    if dataset_path is not None:
                        self.lq_data = f[dataset_path][:]
                    else:
                        print(f"Warning: No datasets found in LQ HDF5 file: {self.lq_path}. Proceeding with on-the-fly noise generation.")
        else:
            self.lq_data = None
            if self.use_real_rixs_data:
                print("Warning: use_real_rixs_data=True but no valid lq_path provided. Falling back to synthetic noise.")
            else:
                print("LQ data will be generated on-the-fly with synthetic noise.")

        if self.lq_data is not None and len(self.gt_data) != len(self.lq_data):
            raise ValueError(f"Mismatch between number of GT ({len(self.gt_data)}) and LQ ({len(self.lq_data)}) images.")

        # Ensure data is in the right format
        if len(self.gt_data.shape) == 3:
            # Add channel dimension if missing (H, W, C)
            self.gt_data = self.gt_data[..., np.newaxis]
            if self.lq_data is not None:
                self.lq_data = self.lq_data[..., np.newaxis]
        
        # Convert to float32 and normalize to [0, 1]
        self.gt_data = self.gt_data.astype(np.float32)
        if self.lq_data is not None:
            self.lq_data = self.lq_data.astype(np.float32)
        
        # Normalize to [0, 1] range
        self.gt_data = (self.gt_data - self.gt_data.min()) / (self.gt_data.max() - self.gt_data.min())
        if self.lq_data is not None:
            self.lq_data = (self.lq_data - self.lq_data.min()) / (self.lq_data.max() - self.lq_data.min())
        
        self.num_samples = len(self.gt_data)
        
        # Normalization parameters
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        
        print(f"Loaded RIXS dataset with {self.num_samples} samples")
        print(f"GT data shape: {self.gt_data.shape}")
        if self.lq_data is not None:
            print(f"LQ data shape: {self.lq_data.shape}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        gt = self.gt_data[index]

        # generate LQ image if not provided
        if self.lq_data is not None:
            lq = self.lq_data[index]
        else:
            # Generate synthetic noise on the fly
            # For now, using simple Gaussian noise.
            # This can be replaced with a more sophisticated noise model if needed.
            if self.sigma_type == 'constant':
                sigma = self.sigma_range
            else: # random
                sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            
            # Normalize sigma to be a percentage of the image's max value
            sigma_normalized = sigma / 100.0 
            noise = np.random.normal(0, sigma_normalized, gt.shape).astype(gt.dtype)
            lq = gt + noise
            
        # Add channel dimension if necessary, assuming [H, W] -> [H, W, C]
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=2)
            lq = np.expand_dims(lq, axis=2)
        
        # Training augmentations
        if self.phase == 'train':
            # Random crop
            gt, lq = self._random_crop(gt, lq, self.gt_size)
            
            # Geometric augmentations
            if self.geometric_augs:
                gt, lq = self._geometric_augmentation(gt, lq)
        
        # Convert to torch tensors and change to CHW format
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        lq = torch.from_numpy(lq.transpose(2, 0, 1)).float()
        
        # Normalize if specified
        if self.mean is not None and self.std is not None:
            normalize(gt, self.mean, self.std, inplace=True)
            normalize(lq, self.mean, self.std, inplace=True)
        
        return {
            'gt': gt,
            'lq': lq,
            'gt_path': f"{self.gt_path}_{index}",
            'lq_path': f"{self.lq_path}_{index}"
        }
    
    def _random_crop(self, gt_img, lq_img, crop_size):
        """Random crop for training."""
        h, w = gt_img.shape[:2]
        
        if h < crop_size or w < crop_size:
            # Pad if image is smaller than crop size
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            gt_img = np.pad(gt_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            lq_img = np.pad(lq_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            h, w = gt_img.shape[:2]
        
        # Random crop
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        gt_img = gt_img[top:top + crop_size, left:left + crop_size]
        lq_img = lq_img[top:top + crop_size, left:left + crop_size]
        
        return gt_img, lq_img
    
    def _geometric_augmentation(self, gt_img, lq_img):
        """Apply geometric augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            gt_img = np.flip(gt_img, axis=1)
            lq_img = np.flip(lq_img, axis=1)
        
        # Random vertical flip
        if random.random() > 0.5:
            gt_img = np.flip(gt_img, axis=0)
            lq_img = np.flip(lq_img, axis=0)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            gt_img = np.rot90(gt_img, k, axes=(0, 1))
            lq_img = np.rot90(lq_img, k, axes=(0, 1))
        
        return gt_img.copy(), lq_img.copy()


class RIXSTestDataset(Dataset):
    """RIXS Dataset for testing/validation with HDF5 files.
    
    This dataset loads RIXS data for testing without augmentations.
    """
    
    def _find_first_dataset(self, h5_group, max_depth=3, current_depth=0):
        """Recursively find the first dataset in an HDF5 group."""
        if current_depth >= max_depth:
            return None
            
        for key in h5_group.keys():
            obj = h5_group[key]
            if isinstance(obj, h5py.Dataset):
                if current_depth == 0:
                    return key
                else:
                    return key  # Return relative path
            elif isinstance(obj, h5py.Group):
                sub_path = self._find_first_dataset(obj, max_depth, current_depth + 1)
                if sub_path is not None:
                    return f"{key}/{sub_path}"
        return None
    
    def __init__(self, opt):
        super(RIXSTestDataset, self).__init__()
        self.opt = opt
        
        # Check if we should use real RIXS data or synthetic noise
        self.use_real_rixs_data = opt.get('use_real_rixs_data', False)
        
        # Load data from HDF5 files
        self.gt_path = opt['dataroot_gt']
        # For real RIXS data, LQ and GT are in the same file by default
        self.lq_path = opt.get('dataroot_lq', self.gt_path if self.use_real_rixs_data else self.gt_path)
        
        # Load the HDF5 files
        with h5py.File(self.gt_path, 'r') as f:
            if self.use_real_rixs_data:
                # For real RIXS data, use high-count data as ground truth
                if 'hc/data' in f:
                    self.gt_data = f['hc/data'][:]
                    print("Using RIXS high count data as ground truth: hc/data")
                elif 'hc' in f and 'data' in f['hc']:
                    self.gt_data = f['hc']['data'][:]
                    print("Using RIXS high count data as ground truth: hc/data")
                else:
                    raise ValueError(f"No high-count RIXS data found in {self.gt_path}. Expected 'hc/data' structure.")
            else:
                # For synthetic noise, try different possible key names for the dataset
                if 'clean_images' in f and isinstance(f['clean_images'], h5py.Dataset):
                    self.gt_data = f['clean_images'][:]
                elif 'images' in f and isinstance(f['images'], h5py.Dataset):
                    self.gt_data = f['images'][:]
                elif 'data' in f and isinstance(f['data'], h5py.Dataset):
                    self.gt_data = f['data'][:]
                elif 'hc/data' in f:  # RIXS hierarchical structure
                    self.gt_data = f['hc/data'][:]
                    print("Using RIXS high count data: hc/data")
                elif 'hc' in f and 'data' in f['hc']:
                    self.gt_data = f['hc']['data'][:]
                    print("Using RIXS high count data: hc/data")
                else:
                    # Find the first dataset (not group) in the file recursively
                    dataset_path = self._find_first_dataset(f)
                    
                    if dataset_path is None:
                        raise ValueError(f"No datasets found in HDF5 file: {self.gt_path}")
                    
                    self.gt_data = f[dataset_path][:]
                    print(f"Using dataset path: {dataset_path}")
            
        # Load LQ data based on configuration
        if self.use_real_rixs_data:
            # For real RIXS data, load LC data from the same file as GT (or separate file if specified)
            lq_file_path = self.lq_path if self.lq_path is not None else self.gt_path
            with h5py.File(lq_file_path, 'r') as f:
                if 'lc/data' in f:
                    self.lq_data = f['lc/data'][:]
                    print("Using real RIXS low count data for validation: lc/data")
                elif 'lc' in f and 'data' in f['lc']:
                    self.lq_data = f['lc']['data'][:]
                    print("Using real RIXS low count data for validation: lc/data")
                else:
                    raise ValueError(f"No low-count RIXS data found in {lq_file_path}. Expected 'lc/data' structure.")
        elif not self.use_real_rixs_data and self.lq_path is not None and self.lq_path != self.gt_path and opt.get('dataroot_lq') != 'none':
            # Load synthetic noise data (for compatibility)
            with h5py.File(self.lq_path, 'r') as f:
                if 'noisy_images' in f and isinstance(f['noisy_images'], h5py.Dataset):
                    self.lq_data = f['noisy_images'][:]
                elif 'images' in f and isinstance(f['images'], h5py.Dataset):
                    self.lq_data = f['images'][:]
                elif 'data' in f and isinstance(f['data'], h5py.Dataset):
                    self.lq_data = f['data'][:]
                elif 'lc/data' in f:  # RIXS hierarchical structure
                    self.lq_data = f['lc/data'][:]
                    print("Using RIXS low count data: lc/data")
                elif 'lc' in f and 'data' in f['lc']:
                    self.lq_data = f['lc']['data'][:]
                    print("Using RIXS low count data: lc/data")
                else:
                    # Find the first dataset (not group) in the file recursively
                    dataset_path = self._find_first_dataset(f)
                    
                    if dataset_path is None:
                        raise ValueError(f"No datasets found in HDF5 file: {self.lq_path}")
                    
                    self.lq_data = f[dataset_path][:]
        else:
            # Generate synthetic noisy data for validation
            self.sigma_test = opt.get('sigma_test', 25)
            self.lq_data = None  # Will generate on the fly
            if self.use_real_rixs_data:
                print("Warning: use_real_rixs_data=True but no valid lq_path provided. Falling back to synthetic noise for validation.")
            else:
                print("LQ data will be generated on-the-fly with synthetic noise for validation.")
            
        # Ensure data is in the right format
        if len(self.gt_data.shape) == 3:
            # Add channel dimension if missing
            self.gt_data = self.gt_data[..., np.newaxis]
            if self.lq_data is not None:
                self.lq_data = self.lq_data[..., np.newaxis]
        
        # Convert to float32 and normalize to [0, 1]
        self.gt_data = self.gt_data.astype(np.float32)
        if self.lq_data is not None:
            self.lq_data = self.lq_data.astype(np.float32)
        
        # Normalize to [0, 1] range
        self.gt_data = (self.gt_data - self.gt_data.min()) / (self.gt_data.max() - self.gt_data.min())
        if self.lq_data is not None:
            self.lq_data = (self.lq_data - self.lq_data.min()) / (self.lq_data.max() - self.lq_data.min())
        
        self.num_samples = len(self.gt_data)
        
        # Normalization parameters
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        
        print(f"Loaded RIXS test dataset with {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        gt = self.gt_data[index]

        # generate LQ image if not provided
        if self.lq_data is not None:
            lq = self.lq_data[index]
        else:
            # Generate synthetic noise on the fly for validation
            sigma_normalized = self.sigma_test / 100.0
            noise = np.random.normal(0, sigma_normalized, gt.shape).astype(gt.dtype)
            lq = gt + noise

        # Add channel dimension if necessary
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=2)
            lq = np.expand_dims(lq, axis=2)
        
        # Convert to torch tensors and change to CHW format
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        lq = torch.from_numpy(lq.transpose(2, 0, 1)).float()
        
        # Normalize if specified
        if self.mean is not None and self.std is not None:
            normalize(gt, self.mean, self.std, inplace=True)
            normalize(lq, self.mean, self.std, inplace=True)
        
        return {
            'gt': gt,
            'lq': lq,
            'gt_path': f"{self.gt_path}_{index}",
            'lq_path': f"{self.lq_path}_{index}"
        } 