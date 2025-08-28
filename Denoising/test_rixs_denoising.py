import argparse
import cv2
import glob
import numpy as np
import os
import torch
import h5py
from collections import OrderedDict

from basicsr.models import create_model
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.options import parse
from basicsr.models.archs.restormer_arch import Restormer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Path to input HDF5 file with noisy RIXS images')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save denoised results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained Restormer model')
    parser.add_argument('--tile_size', type=int, default=128,
                        help='Tile size for processing large images')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Tile overlap for seamless processing')
    parser.add_argument('--save_input', action='store_true',
                        help='Save input images alongside results')
    parser.add_argument('--skip_hdf5_save', action='store_true',
                        help='Skip saving HDF5 results file (saves storage space)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create Restormer model with training configuration parameters
    print("Creating Restormer model...")
    model = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    ).to(device)
    
    # Load model checkpoint
    print(f'Loading model weights from {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model state dict from checkpoint
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            model_state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            # If checkpoint is just the state dict
            model_state_dict = checkpoint
    else:
        # If checkpoint is the model itself
        print("Checkpoint appears to be a model object directly")
        model = checkpoint
    
    # Load state dict into model
    if isinstance(checkpoint, dict):
        model.load_state_dict(model_state_dict)
        print("Model weights loaded successfully")
    
    model.eval()
    
    # Load input data
    print(f'Loading input data from {args.input_path}')
    with h5py.File(args.input_path, 'r') as f:
        # Try different possible keys for the image data
        if 'images' in f:
            input_images = f['images'][:]
        elif 'noisy_images' in f:
            input_images = f['noisy_images'][:]
        elif 'test_data' in f:
            input_images = f['test_data'][:]
        elif 'lc' in f and 'data' in f['lc']:
            # Use low-count images for testing (noisy input)
            input_images = f['lc']['data'][:]
            print("Using low-count (noisy) images from lc/data")
        elif 'hc' in f and 'data' in f['hc']:
            # Fallback to high-count images
            input_images = f['hc']['data'][:]
            print("Using high-count images from hc/data")
        else:
            keys = list(f.keys())
            print(f'Available keys: {keys}')
            print("Error: Could not find image data in expected format")
            return
    
    print(f'Loaded {input_images.shape[0]} images of size {input_images.shape[1:]}')
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process each image
    denoised_images = []
    
    for i, img in enumerate(input_images):
        print(f'Processing image {i+1}/{len(input_images)}')
        
        # Keep original scaling (same as training)
        # Do NOT normalize per-image as this changes the scale the model was trained on
        img = img.astype(np.float32)
        
        # Add channel dimension if needed
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        
        # Convert to tensor
        img_tensor = img2tensor(img, bgr2rgb=False, float32=True).unsqueeze(0).to(device)
        
        # Denoise with model
        with torch.no_grad():
            if img_tensor.shape[2] > args.tile_size or img_tensor.shape[3] > args.tile_size:
                # Use tiled processing for large images
                output = tile_process(img_tensor, model, args.tile_size, args.tile_overlap)
            else:
                output = model(img_tensor)
        
        # Convert back to numpy using appropriate range for RIXS data
        # Since we're not normalizing input, output will be in original data range (~0 to 3)
        output_img = tensor2img(output, rgb2bgr=False, min_max=(0, 3), out_type=np.float32)
        
        # Remove channel dimension if grayscale
        if len(output_img.shape) == 3 and output_img.shape[2] == 1:
            output_img = output_img[:, :, 0]
        
        denoised_images.append(output_img)
        
        # Save individual images if requested
        if args.save_input:
            input_save_path = os.path.join(args.output_path, f'input_{i:04d}.png')
            # Normalize for PNG saving (0-255) with safe division
            img_data = img[:, :, 0]
            img_range = img_data.max() - img_data.min()
            if img_range > 0:
                img_normalized = ((img_data - img_data.min()) / img_range * 255).astype(np.uint8)
            else:
                img_normalized = np.zeros_like(img_data, dtype=np.uint8)
            cv2.imwrite(input_save_path, img_normalized)
        
        output_save_path = os.path.join(args.output_path, f'denoised_{i:04d}.png')
        # Normalize for PNG saving (0-255) with safe division
        output_range = output_img.max() - output_img.min()
        if output_range > 0:
            output_normalized = ((output_img - output_img.min()) / output_range * 255).astype(np.uint8)
        else:
            output_normalized = np.zeros_like(output_img, dtype=np.uint8)
        cv2.imwrite(output_save_path, output_normalized)
    
    # Save results as HDF5 (unless skipped)
    if not args.skip_hdf5_save:
        denoised_images = np.array(denoised_images)
        output_hdf5_path = os.path.join(args.output_path, 'denoised_results.hdf5')
        
        with h5py.File(output_hdf5_path, 'w') as f:
            f.create_dataset('denoised_images', data=denoised_images)
            f.create_dataset('input_images', data=input_images)
        
        print(f'Results saved to {output_hdf5_path}')
    else:
        print('HDF5 saving skipped to save storage space')
    
    print(f'Individual images saved to {args.output_path}')


def tile_process(img, model, tile_size, tile_overlap):
    """Process large images using tiled approach."""
    batch, channel, height, width = img.shape
    output_height, output_width = height, width
    
    # Calculate number of tiles
    tiles_x = (width + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)
    tiles_y = (height + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)
    
    # Initialize output
    output = torch.zeros_like(img)
    
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate tile coordinates
            start_x = x * (tile_size - tile_overlap)
            start_y = y * (tile_size - tile_overlap)
            end_x = min(start_x + tile_size, width)
            end_y = min(start_y + tile_size, height)
            
            # Extract tile
            tile = img[:, :, start_y:end_y, start_x:end_x]
            
            # Process tile
            with torch.no_grad():
                tile_output = model(tile)
            
            # Handle overlap blending (simple approach)
            if x == 0 and y == 0:
                # First tile - no blending needed
                output[:, :, start_y:end_y, start_x:end_x] = tile_output
            else:
                # Blend with existing output
                # Simple averaging in overlap regions
                output[:, :, start_y:end_y, start_x:end_x] = (
                    output[:, :, start_y:end_y, start_x:end_x] + tile_output
                ) / 2
    
    return output


if __name__ == '__main__':
    main() 