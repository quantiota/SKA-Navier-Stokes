#!/usr/bin/env python3
"""
SKA Training Data Batch Extractor

Extract large datasets (3500+ samples) for SKA machine learning training
by processing in batches to avoid JHTDB limits.
"""

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import argparse

from streaming_extractor import StreamingTurbulenceExtractor


def generate_transition_monitoring_points(n_points: int, domain_bounds: dict) -> np.ndarray:
    """
    Generate strategically placed monitoring points for laminar-turbulent transition.
    
    Places points to capture:
    - Laminar region (upstream)
    - Transition onset 
    - Transition development
    - Fully turbulent region (downstream)
    - Different wall-normal heights
    """
    points = []
    
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y'] 
    z_mid = (domain_bounds['z'][0] + domain_bounds['z'][1]) / 2
    
    # Streamwise stations (transition development)
    x_stations = np.linspace(x_min, x_max, int(n_points * 0.7))  # 70% streamwise
    
    # Wall-normal heights (boundary layer sampling)
    y_heights = [1.0, 3.0, 8.0, 15.0]  # Different BL heights
    
    # Primary streamwise line at mid-boundary layer height
    for i, x in enumerate(x_stations):
        y = 3.0  # Mid boundary layer
        z = z_mid + np.random.uniform(-50, 50)  # Slight spanwise variation
        points.append([x, y, z])
    
    # Fill remaining points with vertical profiles at key x-locations
    remaining_points = n_points - len(points)
    x_key_locations = [100, 200, 300, 400]  # Key transition locations
    
    for i in range(remaining_points):
        x = x_key_locations[i % len(x_key_locations)]
        y = y_heights[i % len(y_heights)]
        z = z_mid + np.random.uniform(-30, 30)
        points.append([x, y, z])
    
    return np.array(points[:n_points])


def extract_ska_training_data(auth_token: str, 
                             output_path: str,
                             total_samples: int = 3500,
                             n_points: int = 20,
                             batch_size: int = 500,
                             dataset: str = 'transition_bl'):
    """
    Extract large dataset for SKA training in manageable batches.
    
    Args:
        total_samples: Total number of time samples needed (3500 for 5min at 12Hz)
        n_points: Number of spatial monitoring points
        batch_size: Samples per batch (to avoid JHTDB limits)
        dataset: JHTDB dataset to use
    """
    
    print(f"🎯 SKA TRAINING DATA EXTRACTION")
    print(f"Target: {total_samples} samples at {n_points} points")
    print(f"Strategy: Extract in {batch_size}-sample batches")
    print("=" * 50)
    
    # Initialize extractor
    extractor = StreamingTurbulenceExtractor(auth_token, dataset, output_path)
    
    # Calculate number of batches needed
    n_batches = int(np.ceil(total_samples / batch_size))
    
    print(f"Will extract {n_batches} batches of ~{batch_size} samples each")
    
    # Generate comprehensive spatial points for training diversity
    if dataset == 'isotropic1024coarse':
        domain_bounds = {
            'x': (0.0, 2*np.pi),
            'y': (0.0, 2*np.pi), 
            'z': (0.0, 2*np.pi)
        }
    elif dataset == 'transition_bl':
        domain_bounds = {
            'x': (50.0, 500.0),
            'y': (0.5, 15.0),
            'z': (0.0, 200.0)
        }
    
    # Generate monitoring points strategically for transition analysis
    if dataset == 'transition_bl':
        # Strategic placement for laminar-turbulent transition
        points = generate_transition_monitoring_points(n_points, domain_bounds)
    else:
        # Random distribution for isotropic turbulence
        points = np.random.uniform(
            low=[domain_bounds['x'][0], domain_bounds['y'][0], domain_bounds['z'][0]],
            high=[domain_bounds['x'][1], domain_bounds['y'][1], domain_bounds['z'][1]],
            size=(n_points, 3)
        )
    
    print(f"\nGenerated {n_points} monitoring points across domain:")
    for i, point in enumerate(points[:5]):  # Show first 5
        print(f"  Point {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
    if n_points > 5:
        print(f"  ... and {n_points-5} more points")
    
    # Storage for all data
    all_dataframes = []
    successful_samples = 0
    
    # Extract data in batches
    for batch_idx in range(n_batches):
        print(f"\n📦 BATCH {batch_idx + 1}/{n_batches}")
        
        # Calculate time range for this batch
        samples_this_batch = min(batch_size, total_samples - successful_samples)
        time_start = batch_idx * batch_size * extractor.dt_uniform
        time_end = time_start + (samples_this_batch - 1) * extractor.dt_uniform
        
        print(f"  Extracting {samples_this_batch} samples")
        print(f"  Time range: {time_start:.3f} to {time_end:.3f}")
        
        try:
            # Extract batch data
            start_time = time.time()
            
            df_batch = extractor.extract_time_series_points(
                points=points,
                time_range=(time_start, time_end),
                variables=['velocity'],
                add_timestamps=True
            )
            
            batch_time = time.time() - start_time
            
            if len(df_batch) > 0:
                all_dataframes.append(df_batch)
                successful_samples += len(df_batch)
                
                print(f"  ✅ Extracted {len(df_batch)} samples in {batch_time:.1f}s")
                print(f"  📊 Total progress: {successful_samples}/{total_samples} samples")
            else:
                print(f"  ⚠️  No data returned for batch {batch_idx + 1}")
            
            # Brief pause between batches to be nice to JHTDB servers
            if batch_idx < n_batches - 1:
                print(f"  ⏸️  Pausing 2s before next batch...")
                time.sleep(2)
                
        except Exception as e:
            print(f"  ❌ Error in batch {batch_idx + 1}: {e}")
            print(f"  Continuing with next batch...")
            continue
    
    # Combine all batches
    if all_dataframes:
        print(f"\n🔄 COMBINING {len(all_dataframes)} BATCHES...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"✅ EXTRACTION COMPLETE!")
        print(f"📊 Final dataset: {len(combined_df)} samples × {n_points} points")
        print(f"⏱️  Time range: {combined_df['simulation_time'].min():.3f} to {combined_df['simulation_time'].max():.3f}")
        print(f"🎯 Target achievement: {len(combined_df)/total_samples*100:.1f}%")
        
        # Save final dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ska_training_{len(combined_df)}samples_{timestamp}"
        
        extractor.save_for_questdb(combined_df, filename)
        
        # Additional training-specific metadata
        training_metadata = {
            'purpose': 'SKA machine learning training',
            'extraction_date': datetime.now().isoformat(),
            'total_samples': len(combined_df),
            'spatial_points': n_points,
            'batch_extraction': True,
            'n_batches': len(all_dataframes),
            'target_samples': total_samples,
            'success_rate': len(combined_df)/total_samples,
            'time_resolution_hz': 1.0/extractor.dt_uniform,
            'domain_bounds': domain_bounds,
            'dataset_source': dataset
        }
        
        import json
        metadata_path = os.path.join(output_path, f"{filename}_training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        print(f"💾 Training metadata: {metadata_path}")
        
        return combined_df, points
    
    else:
        print("❌ EXTRACTION FAILED - No data retrieved")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Extract large datasets for SKA training')
    
    parser.add_argument('--auth-token', required=True, help='JHTDB auth token')
    parser.add_argument('--output-path', required=True, help='Output directory')
    parser.add_argument('--samples', type=int, default=3500, 
                       help='Total samples needed (default: 3500 for 5min at 12Hz)')
    parser.add_argument('--points', type=int, default=20,
                       help='Number of spatial monitoring points')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Samples per batch (to avoid JHTDB limits)')
    parser.add_argument('--dataset', default='transition_bl',
                       help='JHTDB dataset to use')
    
    args = parser.parse_args()
    
    extract_ska_training_data(
        auth_token=args.auth_token,
        output_path=args.output_path,
        total_samples=args.samples,
        n_points=args.points,
        batch_size=args.batch_size,
        dataset=args.dataset
    )


if __name__ == "__main__":
    main()