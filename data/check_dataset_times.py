#!/usr/bin/env python3
"""
Check available time intervals in different JHTDB datasets
for high-resolution streaming applications.
"""

from givernylocal.turbulence_dataset import turb_dataset
import numpy as np

def check_dataset_times(dataset_name, auth_token):
    """Check time resolution for a specific dataset."""
    print(f"\n=== {dataset_name.upper()} DATASET ===")
    
    try:
        dataset = turb_dataset(
            dataset_title=dataset_name,
            output_path='./temp_check',
            auth_token=auth_token
        )
        
        # Try different ways to get time information
        times = None
        
        # Method 1: Check if dataset has metadata with time info
        if hasattr(dataset, 'metadata'):
            metadata = dataset.metadata
            if 'timerange' in metadata:
                time_info = metadata['timerange']
                print(f"Time range from metadata: {time_info}")
            if 'timestep' in metadata:
                dt = metadata['timestep']
                print(f"Time step from metadata: {dt}")
        
        # Method 2: Try to extract time series data to infer available times
        # Use a simple point and try different time values
        if dataset_name == 'transition_bl':
            test_point = np.array([[100.0, 1.0, 0.0]], dtype=np.float64)  # Valid for transition_bl
        elif dataset_name in ['isotropic1024coarse', 'isotropic1024fine', 'mhd1024']:
            test_point = np.array([[3.14, 3.14, 3.14]], dtype=np.float64)  # Center of 2π domain
        elif dataset_name in ['channel', 'channel5200']:
            test_point = np.array([[3.14, 1.0, 1.57]], dtype=np.float64)  # Channel flow coordinates
        else:
            test_point = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)  # Generic point
        
        # Test common time ranges for different datasets
        if dataset_name in ['isotropic1024coarse', 'isotropic1024fine']:
            test_times = np.arange(0.0, 10.0, 0.1)  # Common range for isotropic
        elif dataset_name in ['channel', 'channel5200']:
            test_times = np.arange(0.0, 50.0, 1.0)  # Common range for channel
        elif dataset_name == 'transition_bl':
            test_times = np.arange(0.0, 5.0, 0.1)  # Common range for transition
        else:
            test_times = np.arange(0.0, 10.0, 0.5)  # Generic range
        
        valid_times = []
        print(f"Testing time availability (first 10 values)...")
        
        for i, t in enumerate(test_times[:10]):  # Test first 10 times only
            try:
                from givernylocal.turbulence_toolkit import getData
                result = getData(dataset, 'velocity', t, 'none', 'none', 'field', test_point)
                valid_times.append(t)
                print(f"  t={t:.3f} ✓")
            except Exception:
                print(f"  t={t:.3f} ✗")
                continue
        
        if len(valid_times) >= 2:
            times = np.array(valid_times)
            dt_values = np.diff(times)
            
            print(f"Found {len(valid_times)} valid times (from sample)")
            print(f"Time range: {min(times):.6f} to {max(times):.6f}")
            print(f"Time intervals (Δt): {dt_values}")
            print(f"Average Δt: {np.mean(dt_values):.6f}")
            print(f"Min Δt: {np.min(dt_values):.6f}")
            print(f"Uniform time spacing: {np.allclose(dt_values, dt_values[0]) if len(dt_values) > 0 else 'N/A'}")
            
            return {
                'dataset': dataset_name,
                'times': times,
                'dt_avg': np.mean(dt_values),
                'dt_min': np.min(dt_values),
                'uniform': np.allclose(dt_values, dt_values[0])
            }
        else:
            print(f"Dataset appears to be snapshot-based or time-invariant")
            print(f"Available snapshot times: {valid_times}")
            return {
                'dataset': dataset_name,
                'times': np.array(valid_times) if valid_times else np.array([0.0]),
                'dt_avg': 0.0,
                'dt_min': 0.0,
                'uniform': False,
                'snapshot_based': True
            }
        
    except Exception as e:
        print(f"Error accessing {dataset_name}: {e}")
        return None

def main():
    auth_token = 'edu.jhu.pha.turbulence.testing-201406'
    
    # Datasets with high time resolution (good for streaming)
    high_res_datasets = [
        'isotropic1024coarse',
        'isotropic1024fine', 
        'channel',
        'mixing',
        'mhd1024'
    ]
    
    # Boundary layer datasets (may have lower time resolution)
    boundary_layer_datasets = [
        'transition_bl',
        'channel5200'
    ]
    
    print("CHECKING HIGH TIME RESOLUTION DATASETS FOR STREAMING:")
    print("=" * 60)
    
    best_datasets = []
    
    # Check high-resolution datasets first
    for dataset_name in high_res_datasets:
        result = check_dataset_times(dataset_name, auth_token)
        if result and result['dt_min'] > 0:
            best_datasets.append(result)
    
    # Check boundary layer datasets
    print(f"\nCHECKING BOUNDARY LAYER DATASETS:")
    print("=" * 40)
    
    for dataset_name in boundary_layer_datasets:
        result = check_dataset_times(dataset_name, auth_token)
        if result and result['dt_min'] > 0:
            best_datasets.append(result)
    
    # Rank by time resolution
    if best_datasets:
        print(f"\nRANKING BY TIME RESOLUTION (best for streaming):")
        print("=" * 50)
        
        best_datasets.sort(key=lambda x: x['dt_min'])
        
        for i, dataset in enumerate(best_datasets, 1):
            freq_hz = 1.0 / dataset['dt_min'] if dataset['dt_min'] > 0 else 0
            print(f"{i}. {dataset['dataset']}")
            print(f"   Min Δt: {dataset['dt_min']:.6f} → {freq_hz:.1f} Hz equivalent")
            print(f"   Avg Δt: {dataset['dt_avg']:.6f}")
            print(f"   Uniform: {dataset['uniform']}")
            print()
        
        # Recommend best dataset
        best = best_datasets[0]
        print(f"🎯 RECOMMENDED FOR STREAMING: {best['dataset']}")
        print(f"   → Highest time resolution: Δt = {best['dt_min']:.6f}")
        print(f"   → Equivalent frequency: {1.0/best['dt_min']:.1f} Hz")
        print(f"   → {len(best['times'])} time snapshots available")

if __name__ == "__main__":
    main()