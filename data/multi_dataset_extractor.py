#!/usr/bin/env python3
"""
Multi-Dataset SKA Training Extractor

Combine multiple JHTDB datasets to create large training sets (3500+ samples)
by using all available time steps from several datasets.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import argparse

from streaming_extractor import StreamingTurbulenceExtractor


def extract_multi_dataset_training(auth_token: str, output_path: str, target_samples: int = 3500):
    """Extract from multiple datasets to reach target sample count."""
    
    # Datasets ranked by time resolution (from time checker results)
    datasets = [
        {'name': 'isotropic1024coarse', 'max_samples': 1000, 'dt': 0.1},
        {'name': 'transition_bl', 'max_samples': 50, 'dt': 0.1}, 
        {'name': 'mixing', 'max_samples': 200, 'dt': 0.5},
        {'name': 'mhd1024', 'max_samples': 100, 'dt': 0.5},
        {'name': 'channel', 'max_samples': 500, 'dt': 1.0},
        {'name': 'channel5200', 'max_samples': 400, 'dt': 1.0}
    ]
    
    print(f"🎯 MULTI-DATASET SKA TRAINING EXTRACTION")
    print(f"Target: {target_samples} samples from multiple datasets")
    print("=" * 60)
    
    all_dataframes = []
    total_extracted = 0
    n_points = 20
    
    for dataset_info in datasets:
        if total_extracted >= target_samples:
            break
            
        dataset_name = dataset_info['name']
        max_samples = min(dataset_info['max_samples'], target_samples - total_extracted)
        
        print(f"\n📦 EXTRACTING FROM {dataset_name.upper()}")
        print(f"   Target: {max_samples} samples")
        
        try:
            # Initialize extractor for this dataset
            extractor = StreamingTurbulenceExtractor(auth_token, dataset_name, output_path)
            
            # Generate appropriate spatial points for dataset
            if dataset_name == 'transition_bl':
                domain_bounds = {'x': (50.0, 300.0), 'y': (0.5, 10.0), 'z': (0.0, 100.0)}
            elif 'channel' in dataset_name:
                domain_bounds = {'x': (0.0, 4*np.pi), 'y': (0.0, 2.0), 'z': (0.0, 2*np.pi)}
            else:  # isotropic datasets
                domain_bounds = {'x': (0.0, 2*np.pi), 'y': (0.0, 2*np.pi), 'z': (0.0, 2*np.pi)}
            
            points = np.random.uniform(
                low=[domain_bounds['x'][0], domain_bounds['y'][0], domain_bounds['z'][0]],
                high=[domain_bounds['x'][1], domain_bounds['y'][1], domain_bounds['z'][1]],
                size=(n_points, 3)
            )
            
            # Extract available time steps
            available_times = len(extractor.times)
            samples_to_extract = min(max_samples, available_times)
            
            if samples_to_extract > 0:
                time_range = (extractor.times[0], extractor.times[samples_to_extract-1])
                
                print(f"   Available times: {available_times}")
                print(f"   Extracting: {samples_to_extract} samples")
                print(f"   Time range: {time_range[0]:.3f} to {time_range[1]:.3f}")
                
                df = extractor.extract_time_series_points(
                    points=points,
                    time_range=time_range,
                    variables=['velocity'],
                    add_timestamps=True
                )
                
                if len(df) > 0:
                    # Add dataset identifier
                    df['dataset_source'] = dataset_name
                    df['dataset_dt'] = dataset_info['dt']
                    
                    all_dataframes.append(df)
                    total_extracted += len(df)
                    
                    print(f"   ✅ Extracted {len(df)} samples")
                    print(f"   📊 Total progress: {total_extracted}/{target_samples}")
                else:
                    print(f"   ⚠️  No data returned")
            else:
                print(f"   ⚠️  No available time steps")
                
        except Exception as e:
            print(f"   ❌ Error with {dataset_name}: {e}")
            continue
    
    # Combine all datasets
    if all_dataframes:
        print(f"\n🔄 COMBINING {len(all_dataframes)} DATASETS...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by original simulation time and reset timestamps
        combined_df = combined_df.sort_values('simulation_time').reset_index(drop=True)
        
        # Create unified timestamps for streaming
        base_time = datetime.now()
        time_deltas = pd.to_timedelta(combined_df.index * 0.1, unit='s')  # 10 Hz rate
        combined_df['timestamp'] = base_time + time_deltas
        
        print(f"✅ MULTI-DATASET EXTRACTION COMPLETE!")
        print(f"📊 Final dataset: {len(combined_df)} samples × {n_points} points")
        print(f"🎯 Target achievement: {len(combined_df)/target_samples*100:.1f}%")
        
        # Dataset composition summary
        print(f"\n📋 DATASET COMPOSITION:")
        for dataset in combined_df['dataset_source'].unique():
            count = len(combined_df[combined_df['dataset_source'] == dataset])
            percentage = count / len(combined_df) * 100
            print(f"   {dataset}: {count} samples ({percentage:.1f}%)")
        
        # Save combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ska_multi_dataset_{len(combined_df)}samples_{timestamp}"
        
        # Use first extractor to save (any will work for saving)
        extractor.save_for_questdb(combined_df, filename)
        
        return combined_df
    
    else:
        print("❌ MULTI-DATASET EXTRACTION FAILED")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract from multiple datasets for SKA training')
    
    parser.add_argument('--auth-token', required=True, help='JHTDB auth token')
    parser.add_argument('--output-path', required=True, help='Output directory')
    parser.add_argument('--samples', type=int, default=3500, help='Target total samples')
    
    args = parser.parse_args()
    
    extract_multi_dataset_training(args.auth_token, args.output_path, args.samples)


if __name__ == "__main__":
    main()