import argparse
import h5py
import numpy as np
import json
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def calculate_2d_psnr(img1, img2):
    """Calculate 2D PSNR between two images."""
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)


def calculate_2d_ssim(img1, img2):
    """Calculate 2D SSIM between two images."""
    return structural_similarity(img1, img2, data_range=1.0)


def calculate_1d_profile_metrics(img1, img2):
    """Calculate 1D profile metrics (MAE, MSE) by averaging over one dimension."""
    # Average over rows to get 1D profiles
    profile1 = np.mean(img1, axis=0)
    profile2 = np.mean(img2, axis=0)
    
    mae = np.mean(np.abs(profile1 - profile2))
    mse = np.mean((profile1 - profile2) ** 2)
    
    return mae, mse


def calculate_2d_metrics(img1, img2):
    """Calculate 2D MAE and MSE."""
    mae = np.mean(np.abs(img1 - img2))
    mse = np.mean((img1 - img2) ** 2)
    
    return mae, mse


def evaluate_rixs_results(ground_truth_path, denoised_path, output_path):
    """Evaluate RIXS denoising results and compute comprehensive metrics."""
    
    # Load ground truth images
    print(f"Loading ground truth from: {ground_truth_path}")
    with h5py.File(ground_truth_path, 'r') as f:
        if 'images' in f:
            gt_images = f['images'][:]
        elif 'clean_images' in f:
            gt_images = f['clean_images'][:]
        elif 'test_data' in f:
            gt_images = f['test_data'][:]
        elif 'hc' in f and 'data' in f['hc']:
            # Use high-count (clean) images for ground truth
            gt_images = f['hc']['data'][:]
            print("Using high-count (clean) images from hc/data as ground truth")
        else:
            keys = list(f.keys())
            print(f"Available GT keys: {keys}")
            print("Error: Could not find ground truth data in expected format")
            return None
    
    # Load denoised images
    print(f"Loading denoised results from: {denoised_path}")
    if not os.path.exists(denoised_path):
        print(f"Error: Denoised results file not found: {denoised_path}")
        print("This might be because HDF5 saving was skipped to save storage space.")
        print("Evaluation cannot proceed without denoised results.")
        return None
        
    with h5py.File(denoised_path, 'r') as f:
        if 'denoised_images' in f:
            denoised_images = f['denoised_images'][:]
        elif 'images' in f:
            denoised_images = f['images'][:]
        else:
            keys = list(f.keys())
            print(f"Available denoised keys: {keys}")
            denoised_images = f[keys[0]][:]
    
    print(f"GT shape: {gt_images.shape}, Denoised shape: {denoised_images.shape}")
    
    # Ensure same number of images
    min_images = min(len(gt_images), len(denoised_images))
    gt_images = gt_images[:min_images]
    denoised_images = denoised_images[:min_images]
    
    # Normalize images to [0, 1] if needed
    if gt_images.max() > 1.0:
        gt_images = gt_images.astype(np.float32) / gt_images.max()
    if denoised_images.max() > 1.0:
        denoised_images = denoised_images.astype(np.float32) / denoised_images.max()
    
    # Initialize metric lists
    psnr_2d_list = []
    ssim_2d_list = []
    mae_1d_list = []
    mse_1d_list = []
    mae_2d_list = []
    mse_2d_list = []
    
    # Calculate metrics for each image
    for i in range(len(gt_images)):
        gt_img = gt_images[i]
        denoised_img = denoised_images[i]
        
        # Ensure 2D images (remove channel dimension if present)
        if len(gt_img.shape) == 3 and gt_img.shape[2] == 1:
            gt_img = gt_img[:, :, 0]
        if len(denoised_img.shape) == 3 and denoised_img.shape[2] == 1:
            denoised_img = denoised_img[:, :, 0]
        
        # Calculate 2D metrics
        psnr_2d = calculate_2d_psnr(gt_img, denoised_img)
        ssim_2d = calculate_2d_ssim(gt_img, denoised_img)
        
        # Calculate 1D profile metrics
        mae_1d, mse_1d = calculate_1d_profile_metrics(gt_img, denoised_img)
        
        # Calculate 2D pixel-wise metrics
        mae_2d, mse_2d = calculate_2d_metrics(gt_img, denoised_img)
        
        # Append to lists
        psnr_2d_list.append(psnr_2d)
        ssim_2d_list.append(ssim_2d)
        mae_1d_list.append(mae_1d)
        mse_1d_list.append(mse_1d)
        mae_2d_list.append(mae_2d)
        mse_2d_list.append(mse_2d)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(gt_images)} images")
    
    # Calculate summary statistics
    results = {
        '2D_PSNR_mean': float(np.mean(psnr_2d_list)),
        '2D_PSNR_std': float(np.std(psnr_2d_list)),
        '2D_SSIM_mean': float(np.mean(ssim_2d_list)),
        '2D_SSIM_std': float(np.std(ssim_2d_list)),
        '1D_MAE_mean': float(np.mean(mae_1d_list)),
        '1D_MAE_std': float(np.std(mae_1d_list)),
        '1D_MSE_mean': float(np.mean(mse_1d_list)),
        '1D_MSE_std': float(np.std(mse_1d_list)),
        '2D_MAE_mean': float(np.mean(mae_2d_list)),
        '2D_MAE_std': float(np.std(mae_2d_list)),
        '2D_MSE_mean': float(np.mean(mse_2d_list)),
        '2D_MSE_std': float(np.std(mse_2d_list)),
        'num_images': len(gt_images)
    }
    
    # Save detailed results
    detailed_results = {
        'summary': results,
        'per_image': {
            '2D_PSNR': psnr_2d_list,
            '2D_SSIM': ssim_2d_list,
            '1D_MAE': mae_1d_list,
            '1D_MSE': mse_1d_list,
            '2D_MAE': mae_2d_list,
            '2D_MSE': mse_2d_list
        }
    }
    
    # Save results (convert numpy types to native Python types for JSON)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json_compatible_results = convert_numpy_types(detailed_results)
    with open(output_path, 'w') as f:
        json.dump(json_compatible_results, f, indent=2)
    
    # Print summary
    print("\n=== RIXS Denoising Evaluation Results ===")
    print(f"Number of images: {results['num_images']}")
    print(f"2D PSNR: {results['2D_PSNR_mean']:.2f} ± {results['2D_PSNR_std']:.2f} dB")
    print(f"2D SSIM: {results['2D_SSIM_mean']:.4f} ± {results['2D_SSIM_std']:.4f}")
    print(f"1D MAE: {results['1D_MAE_mean']:.6f} ± {results['1D_MAE_std']:.6f}")
    print(f"1D MSE: {results['1D_MSE_mean']:.6f} ± {results['1D_MSE_std']:.6f}")
    print(f"2D MAE: {results['2D_MAE_mean']:.6f} ± {results['2D_MAE_std']:.6f}")
    print(f"2D MSE: {results['2D_MSE_mean']:.6f} ± {results['2D_MSE_std']:.6f}")
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate RIXS denoising results')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth HDF5 file')
    parser.add_argument('--denoised', type=str, required=True,
                        help='Path to denoised results HDF5 file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save evaluation results JSON')
    
    args = parser.parse_args()
    
    evaluate_rixs_results(args.ground_truth, args.denoised, args.output)


if __name__ == '__main__':
    main() 