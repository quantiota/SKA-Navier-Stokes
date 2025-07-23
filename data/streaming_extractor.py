#!/usr/bin/env python3
"""
High-Resolution Streaming Data Extractor for QuestDB

Extracts turbulence data with maximum time resolution for real-time streaming applications.
Optimized for creating continuous time series that can be streamed to QuestDB.
"""

import numpy as np
import pandas as pd
import argparse
import os
from typing import List, Tuple, Dict
import json
from datetime import datetime, timedelta

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, write_interpolation_tsv_file


class StreamingTurbulenceExtractor:
    """Extract high-resolution time series data for streaming applications."""
    
    def __init__(self, auth_token: str, dataset_title: str, output_path: str):
        """Initialize the streaming extractor."""
        self.auth_token = auth_token
        self.dataset_title = dataset_title
        self.output_path = output_path
        
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Initializing {dataset_title} for streaming extraction...")
        self.dataset = turb_dataset(
            dataset_title=dataset_title,
            output_path=output_path,
            auth_token=auth_token
        )
        
        # Get time information by testing available times
        self.times, self.dt_uniform = self._discover_available_times()
        
        print(f"Dataset has {len(self.times)} time snapshots")
        print(f"Time resolution: Δt = {self.dt_uniform:.6f}")
        print(f"Equivalent frequency: {1.0/self.dt_uniform:.2f} Hz")
    
    def _discover_available_times(self):
        """Discover available time snapshots for the dataset."""
        # Use a test point to discover available times
        if self.dataset_title == 'transition_bl':
            test_point = np.array([[100.0, 1.0, 0.0]], dtype=np.float64)
        elif self.dataset_title in ['isotropic1024coarse', 'isotropic1024fine', 'mhd1024']:
            test_point = np.array([[3.14, 3.14, 3.14]], dtype=np.float64)
        elif self.dataset_title in ['channel', 'channel5200']:
            test_point = np.array([[3.14, 1.0, 1.57]], dtype=np.float64)
        else:
            test_point = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        
        # Test time ranges based on dataset
        if self.dataset_title in ['isotropic1024coarse', 'isotropic1024fine']:
            test_times = np.arange(0.0, 10.0, 0.1)
        elif self.dataset_title in ['channel', 'channel5200']:
            test_times = np.arange(0.0, 50.0, 1.0)
        elif self.dataset_title == 'transition_bl':
            test_times = np.arange(0.0, 5.0, 0.1)
        elif self.dataset_title in ['mixing', 'mhd1024']:
            test_times = np.arange(0.0, 10.0, 0.5)
        else:
            test_times = np.arange(0.0, 10.0, 0.5)
        
        valid_times = []
        print(f"Discovering available times for {self.dataset_title}...")
        
        # Test times (limit to reasonable number for performance)
        max_test_times = min(100, len(test_times))
        
        for i, t in enumerate(test_times[:max_test_times]):
            try:
                result = getData(self.dataset, 'velocity', t, 'none', 'none', 'field', test_point)
                valid_times.append(t)
                if i % 10 == 0:  # Progress indicator
                    print(f"  Found {len(valid_times)} valid times so far...")
            except:
                continue
        
        if len(valid_times) >= 2:
            times = np.array(valid_times)
            dt_values = np.diff(times)
            dt_uniform = np.mean(dt_values)
            print(f"Found {len(valid_times)} time snapshots with Δt = {dt_uniform:.6f}")
            return times, dt_uniform
        else:
            print(f"Dataset appears to be snapshot-based with {len(valid_times)} snapshots")
            return np.array(valid_times) if valid_times else np.array([0.0]), 1.0
    
    def extract_time_series_points(self,
                                 points: np.ndarray,
                                 time_range: Tuple[float, float] = None,
                                 variables: List[str] = None,
                                 spatial_method: str = 'none',
                                 add_timestamps: bool = True) -> pd.DataFrame:
        """
        Extract time series at specific spatial points for streaming.
        
        Args:
            points: Array of spatial coordinates [(x,y,z), ...]
            time_range: (start_time, end_time) or None for all times
            variables: Variables to extract
            spatial_method: Spatial interpolation method
            add_timestamps: Add realistic timestamps for streaming
            
        Returns:
            DataFrame with time series data ready for QuestDB
        """
        if variables is None:
            variables = ['velocity']
        
        if time_range is None:
            time_indices = range(len(self.times))
            times_to_extract = self.times
        else:
            mask = (self.times >= time_range[0]) & (self.times <= time_range[1])
            time_indices = np.where(mask)[0]
            times_to_extract = self.times[mask]
        
        print(f"Extracting {len(times_to_extract)} time steps at {len(points)} points")
        
        # Initialize results storage
        all_data = []
        
        # Extract data for each time step
        for i, (time_idx, time_val) in enumerate(zip(time_indices, times_to_extract)):
            print(f"  Processing time {i+1}/{len(times_to_extract)}: t={time_val:.6f}")
            
            # Extract all variables at this time
            time_data = {'simulation_time': time_val}
            
            if add_timestamps:
                # Create realistic timestamp (for streaming simulation)
                base_time = datetime.now() - timedelta(seconds=len(times_to_extract)*self.dt_uniform)
                timestamp = base_time + timedelta(seconds=i*self.dt_uniform)
                time_data['timestamp'] = timestamp
            
            for variable in variables:
                try:
                    result = getData(self.dataset, variable, time_val, 'none',
                                   spatial_method, 'field', points)
                    
                    # Process results for each point
                    field_data = np.array(result[0])  # First (and only) time component
                    
                    for point_idx in range(len(points)):
                        point_key = f"point_{point_idx}"
                        x, y, z = points[point_idx]
                        
                        # Store point location
                        time_data[f'{point_key}_x'] = x
                        time_data[f'{point_key}_y'] = y  
                        time_data[f'{point_key}_z'] = z
                        
                        # Store variable components
                        if variable == 'velocity':
                            time_data[f'{point_key}_u'] = field_data[point_idx, 0]
                            time_data[f'{point_key}_v'] = field_data[point_idx, 1]
                            time_data[f'{point_key}_w'] = field_data[point_idx, 2]
                        elif variable == 'pressure':
                            time_data[f'{point_key}_p'] = field_data[point_idx, 0]
                        else:
                            # Generic handling for other variables
                            for comp_idx in range(field_data.shape[1]):
                                time_data[f'{point_key}_{variable}_{comp_idx}'] = field_data[point_idx, comp_idx]
                
                except Exception as e:
                    print(f"    Error extracting {variable} at t={time_val}: {e}")
                    continue
            
            all_data.append(time_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        if add_timestamps:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    
    def extract_streaming_dataset(self,
                                 n_points: int = 10,
                                 domain_bounds: Dict = None,
                                 time_range: Tuple[float, float] = None,
                                 variables: List[str] = None) -> pd.DataFrame:
        """
        Extract a complete streaming dataset with random points.
        
        Args:
            n_points: Number of spatial points to monitor
            domain_bounds: Domain boundaries {x: (min,max), y: (min,max), z:(min,max)}
            time_range: Time range to extract
            variables: Variables to extract
            
        Returns:
            DataFrame ready for QuestDB streaming
        """
        if domain_bounds is None:
            # Use dataset-specific defaults
            if self.dataset_title == 'transition_bl':
                domain_bounds = {
                    'x': (50.0, 300.0),
                    'y': (0.5, 15.0), 
                    'z': (0.0, 200.0)
                }
            else:
                # Generic bounds for isotropic datasets
                domain_bounds = {
                    'x': (0.0, 2*np.pi),
                    'y': (0.0, 2*np.pi),
                    'z': (0.0, 2*np.pi)
                }
        
        if variables is None:
            variables = ['velocity']
        
        # Generate random monitoring points
        points = np.random.uniform(
            low=[domain_bounds['x'][0], domain_bounds['y'][0], domain_bounds['z'][0]],
            high=[domain_bounds['x'][1], domain_bounds['y'][1], domain_bounds['z'][1]],
            size=(n_points, 3)
        )
        
        print(f"Generated {n_points} monitoring points:")
        for i, point in enumerate(points):
            print(f"  Point {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
        
        # Extract time series
        df = self.extract_time_series_points(points, time_range, variables)
        
        return df, points
    
    def save_for_questdb(self, df: pd.DataFrame, filename: str):
        """Save DataFrame in QuestDB-optimized format."""
        
        # Save as CSV (QuestDB native format)
        csv_path = os.path.join(self.output_path, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved streaming data: {csv_path}")
        
        # Save metadata
        metadata = {
            'dataset': self.dataset_title,
            'extraction_time': datetime.now().isoformat(),
            'time_resolution': float(self.dt_uniform),
            'frequency_hz': float(1.0/self.dt_uniform),
            'n_time_steps': len(df),
            'n_spatial_points': len([col for col in df.columns if col.endswith('_x')]),
            'time_range': [float(df['simulation_time'].min()), float(df['simulation_time'].max())],
            'columns': list(df.columns)
        }
        
        metadata_path = os.path.join(self.output_path, f"{filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Streaming metadata: {metadata_path}")
        
        # Create QuestDB SQL schema
        self._create_questdb_schema(df, filename)
    
    def _create_questdb_schema(self, df: pd.DataFrame, table_name: str):
        """Generate QuestDB table creation SQL."""
        
        sql_lines = [f"CREATE TABLE {table_name} ("]
        
        for col in df.columns:
            if col == 'timestamp':
                sql_lines.append("    timestamp TIMESTAMP,")
            elif col == 'simulation_time':
                sql_lines.append("    simulation_time DOUBLE,")
            elif col.endswith(('_x', '_y', '_z', '_u', '_v', '_w', '_p')):
                sql_lines.append(f"    {col} DOUBLE,")
            else:
                sql_lines.append(f"    {col} DOUBLE,")
        
        sql_lines[-1] = sql_lines[-1].rstrip(',')  # Remove last comma
        sql_lines.append(") timestamp(timestamp);")
        
        sql_path = os.path.join(self.output_path, f"{table_name}_schema.sql")
        with open(sql_path, 'w') as f:
            f.write('\n'.join(sql_lines))
        
        print(f"QuestDB schema: {sql_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract high-resolution data for streaming')
    
    parser.add_argument('--auth-token', required=True, help='JHTDB auth token')
    parser.add_argument('--dataset', default='isotropic1024coarse', 
                       help='Dataset name (use high time resolution dataset)')
    parser.add_argument('--output-path', required=True, help='Output directory')
    parser.add_argument('--n-points', type=int, default=10, 
                       help='Number of monitoring points')
    parser.add_argument('--time-steps', type=int, default=100,
                       help='Number of time steps to extract')
    parser.add_argument('--filename', default='turbulence_stream',
                       help='Output filename prefix')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = StreamingTurbulenceExtractor(
        auth_token=args.auth_token,
        dataset_title=args.dataset,
        output_path=args.output_path
    )
    
    # Determine time range
    if args.time_steps < len(extractor.times):
        time_range = (extractor.times[0], extractor.times[args.time_steps-1])
    else:
        time_range = None
    
    print(f"\nExtracting streaming dataset...")
    print(f"Time steps: {args.time_steps}")
    print(f"Monitoring points: {args.n_points}")
    
    # Extract data
    df, points = extractor.extract_streaming_dataset(
        n_points=args.n_points,
        time_range=time_range
    )
    
    # Save for QuestDB
    extractor.save_for_questdb(df, args.filename)
    
    print(f"\n✅ STREAMING DATASET READY!")
    print(f"📊 {len(df)} time steps × {args.n_points} points")
    print(f"⏱️  Time resolution: {extractor.dt_uniform:.6f} ({1.0/extractor.dt_uniform:.1f} Hz)")
    print(f"💾 Saved to: {args.output_path}")


if __name__ == "__main__":
    main()