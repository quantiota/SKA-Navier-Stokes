#!/usr/bin/env python3
"""
2D Laminar-Turbulent Transition Data Extractor

Specialized script for extracting 2D datasets from JHTDB for laminar-turbulent 
transition analysis. Focuses on boundary layer transition phenomena.

Usage:
    python transition_2d_extractor.py --auth-token <token> --output-path <path>
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Tuple, Dict, Optional
import json

# Import the givernylocal library
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, write_interpolation_tsv_file


class TransitionExtractor2D:
    """Specialized class for 2D laminar-turbulent transition analysis."""
    
    def __init__(self, auth_token: str, output_path: str):
        """
        Initialize the transition data extractor.
        
        Args:
            auth_token: JHTDB authorization token
            output_path: Directory where output files will be saved
        """
        self.auth_token = auth_token
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize with transition boundary layer dataset
        self.dataset_title = 'transition_bl'
        print(f"Initializing transition boundary layer dataset...")
        self.dataset = turb_dataset(
            dataset_title=self.dataset_title,
            output_path=output_path,
            auth_token=auth_token
        )
        
        # Transition-specific parameters
        self.transition_config = {
            'spatial_method': 'none',  # Start with simplest method
            'temporal_method': 'none',  # Single time snapshots
            'variables': ['velocity', 'pressure'],  # Key transition variables
            'spatial_operators': ['field', 'gradient', 'laplacian']  # For stability analysis
        }
        
    def extract_boundary_layer_profile(self,
                                     time: float = 0.0,
                                     x_location: float = 100.0,
                                     z_location: float = 0.0,
                                     ny: int = 50,
                                     y_range: Tuple[float, float] = (0.0, 10.0),
                                     variables: List[str] = None,
                                     operators: List[str] = None) -> Dict:
        """
        Extract boundary layer profile at a specific streamwise location.
        Perfect for analyzing transition from laminar to turbulent flow.
        
        Args:
            time: Time snapshot
            x_location: Streamwise position (critical for transition)
            z_location: Spanwise position
            ny: Number of points in wall-normal direction
            y_range: Wall-normal coordinate range
            variables: Variables to extract
            operators: Spatial operators to apply
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        if variables is None:
            variables = self.transition_config['variables']
        if operators is None:
            operators = ['field']
            
        print(f"Extracting boundary layer profile at x={x_location}")
        
        # Generate wall-normal profile points
        y_points = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
        points = np.array([[x_location, y, z_location] for y in y_points], dtype=np.float64)
        
        results = {}
        
        # Extract data for each variable and operator combination
        for variable in variables:
            results[variable] = {}
            for operator in operators:
                print(f"  Extracting {variable} with {operator} operator...")
                
                # Choose appropriate spatial method based on operator
                if operator == 'field':
                    spatial_method = 'none'  # Start with simplest method
                elif operator in ['gradient', 'hessian', 'laplacian']:
                    spatial_method = 'fd4noint'  # Simplest gradient method
                else:
                    spatial_method = self.transition_config['spatial_method']
                
                result = getData(self.dataset, variable, time, 
                               self.transition_config['temporal_method'],
                               spatial_method, 
                               operator, points)
                
                results[variable][operator] = {
                    'data': result,
                    'points': points,
                    'y_coords': y_points,
                    'metadata': {
                        'x_location': x_location,
                        'z_location': z_location,
                        'time': time,
                        'ny': ny,
                        'y_range': y_range
                    }
                }
        
        return results
    
    def extract_streamwise_evolution(self,
                                   time: float = 1.0,
                                   y_location: float = 2.0,
                                   z_location: float = 0.0,
                                   nx: int = 200,
                                   x_range: Tuple[float, float] = (50.0, 500.0),
                                   variables: List[str] = None,
                                   operators: List[str] = None) -> Dict:
        """
        Extract streamwise evolution of flow at fixed wall-normal distance.
        Ideal for tracking transition onset and development.
        
        Args:
            time: Time snapshot
            y_location: Wall-normal distance (in boundary layer)
            z_location: Spanwise position
            nx: Number of streamwise points
            x_range: Streamwise coordinate range
            variables: Variables to extract
            operators: Spatial operators to apply
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        if variables is None:
            variables = self.transition_config['variables']
        if operators is None:
            operators = ['field', 'gradient']
            
        print(f"Extracting streamwise evolution at y={y_location}")
        
        # Generate streamwise points
        x_points = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
        points = np.array([[x, y_location, z_location] for x in x_points], dtype=np.float64)
        
        results = {}
        
        # Extract data for each variable and operator combination
        for variable in variables:
            results[variable] = {}
            for operator in operators:
                print(f"  Extracting {variable} with {operator} operator...")
                
                # Choose appropriate spatial method based on operator
                if operator == 'field':
                    spatial_method = 'none'  # Start with simplest method
                elif operator in ['gradient', 'hessian', 'laplacian']:
                    spatial_method = 'fd4noint'  # Simplest gradient method
                else:
                    spatial_method = self.transition_config['spatial_method']
                
                result = getData(self.dataset, variable, time,
                               self.transition_config['temporal_method'],
                               spatial_method,
                               operator, points)
                
                results[variable][operator] = {
                    'data': result,
                    'points': points,
                    'x_coords': x_points,
                    'metadata': {
                        'y_location': y_location,
                        'z_location': z_location,
                        'time': time,
                        'nx': nx,
                        'x_range': x_range
                    }
                }
        
        return results
    
    def extract_xy_plane(self,
                        time: float = 1.0,
                        z_location: float = 0.0,
                        nx: int = 150,
                        ny: int = 100,
                        x_range: Tuple[float, float] = (50.0, 400.0),
                        y_range: Tuple[float, float] = (0.0, 20.0),
                        variables: List[str] = None,
                        operators: List[str] = None) -> Dict:
        """
        Extract 2D x-y plane showing boundary layer development.
        Perfect for visualizing transition regions and turbulent spots.
        
        Args:
            time: Time snapshot
            z_location: Spanwise position
            nx: Number of streamwise points
            ny: Number of wall-normal points
            x_range: Streamwise coordinate range
            y_range: Wall-normal coordinate range
            variables: Variables to extract
            operators: Spatial operators to apply
            
        Returns:
            Dictionary containing extracted data and metadata
        """
        if variables is None:
            variables = ['velocity']
        if operators is None:
            operators = ['field']
            
        print(f"Extracting x-y plane at z={z_location}")
        print(f"Grid: {nx}x{ny} points")
        
        # Generate 2D grid points
        x_points = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
        y_points = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
        
        points = np.array([
            axis.ravel() for axis in np.meshgrid(x_points, y_points, z_location, indexing='ij')
        ], dtype=np.float64).T
        
        results = {}
        
        # Extract data for each variable and operator combination
        for variable in variables:
            results[variable] = {}
            for operator in operators:
                print(f"  Extracting {variable} with {operator} operator...")
                
                # Choose appropriate spatial method based on operator
                if operator == 'field':
                    spatial_method = 'none'  # Start with simplest method
                elif operator in ['gradient', 'hessian', 'laplacian']:
                    spatial_method = 'fd4noint'  # Simplest gradient method
                else:
                    spatial_method = self.transition_config['spatial_method']
                
                result = getData(self.dataset, variable, time,
                               self.transition_config['temporal_method'],
                               spatial_method,
                               operator, points)
                
                results[variable][operator] = {
                    'data': result,
                    'points': points,
                    'x_coords': x_points,
                    'y_coords': y_points,
                    'grid_shape': (nx, ny),
                    'metadata': {
                        'z_location': z_location,
                        'time': time,
                        'nx': nx,
                        'ny': ny,
                        'x_range': x_range,
                        'y_range': y_range
                    }
                }
        
        return results
    
    def compute_transition_indicators(self, velocity_data: Dict) -> Dict:
        """
        Compute key indicators for laminar-turbulent transition.
        
        Args:
            velocity_data: Velocity data from extraction methods
            
        Returns:
            Dictionary containing transition indicators
        """
        indicators = {}
        
        # Get velocity field data
        if 'field' in velocity_data:
            vel_field = np.array(velocity_data['field']['data'][0])
            
            # Velocity components
            u_vel = vel_field[:, 0]  # Streamwise
            v_vel = vel_field[:, 1]  # Wall-normal
            w_vel = vel_field[:, 2]  # Spanwise
            
            # Transition indicators
            indicators['velocity_magnitude'] = np.sqrt(u_vel**2 + v_vel**2 + w_vel**2)
            indicators['streamwise_velocity'] = u_vel
            indicators['wall_normal_velocity'] = v_vel
            indicators['spanwise_velocity'] = w_vel
            
            # Turbulence intensity (simplified)
            u_mean = np.mean(u_vel)
            indicators['turbulence_intensity'] = np.std(u_vel) / u_mean if u_mean != 0 else 0
            
        # Velocity gradients for stability analysis
        if 'gradient' in velocity_data:
            vel_grad = np.array(velocity_data['gradient']['data'][0])
            
            # Shear rates
            indicators['du_dy'] = vel_grad[:, 1]  # Wall-normal shear of streamwise velocity
            indicators['dv_dx'] = vel_grad[:, 3]  # Streamwise gradient of wall-normal velocity
            
            # Vorticity magnitude (simplified 2D)
            indicators['vorticity_z'] = vel_grad[:, 3] - vel_grad[:, 1]  # dv/dx - du/dy
            
        return indicators
    
    def plot_boundary_layer_profile(self, profile_data: Dict, variable: str = 'velocity',
                                   component: int = 0, save_plot: bool = True):
        """Plot boundary layer velocity profile."""
        if variable not in profile_data:
            print(f"Variable {variable} not found in data")
            return
            
        field_data = profile_data[variable]['field']
        y_coords = field_data['y_coords']
        vel_data = np.array(field_data['data'][0])
        
        plt.figure(figsize=(8, 6))
        plt.plot(vel_data[:, component], y_coords, 'b-', linewidth=2, 
                label=f'{variable} component {component}')
        plt.xlabel(f'{variable} component {component}')
        plt.ylabel('Wall-normal distance (y)')
        plt.title(f'Boundary Layer Profile at x={field_data["metadata"]["x_location"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_path, 'boundary_layer_profile.png'), 
                       dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_streamwise_evolution(self, evolution_data: Dict, variable: str = 'velocity',
                                 component: int = 0, save_plot: bool = True):
        """Plot streamwise evolution of flow variable."""
        if variable not in evolution_data:
            print(f"Variable {variable} not found in data")
            return
            
        field_data = evolution_data[variable]['field']
        x_coords = field_data['x_coords']
        vel_data = np.array(field_data['data'][0])
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, vel_data[:, component], 'r-', linewidth=2,
                label=f'{variable} component {component}')
        plt.xlabel('Streamwise position (x)')
        plt.ylabel(f'{variable} component {component}')
        plt.title(f'Streamwise Evolution at y={field_data["metadata"]["y_location"]}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_plot:
            plt.savefig(os.path.join(self.output_path, 'streamwise_evolution.png'),
                       dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_xy_plane_contour(self, plane_data: Dict, variable: str = 'velocity',
                             component: int = 0, save_plot: bool = True):
        """Plot 2D contour of x-y plane."""
        if variable not in plane_data:
            print(f"Variable {variable} not found in data")
            return
            
        field_data = plane_data[variable]['field']
        x_coords = field_data['x_coords']
        y_coords = field_data['y_coords']
        nx, ny = field_data['grid_shape']
        vel_data = np.array(field_data['data'][0])
        
        # Reshape data for contour plot
        vel_component = vel_data[:, component].reshape(nx, ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        plt.figure(figsize=(12, 6))
        contour = plt.contourf(X, Y, vel_component, levels=50, cmap='RdYlBu_r')
        plt.colorbar(contour, label=f'{variable} component {component}')
        plt.xlabel('Streamwise position (x)')
        plt.ylabel('Wall-normal distance (y)')
        plt.title(f'2D {variable} Field - Transition Boundary Layer')
        
        if save_plot:
            plt.savefig(os.path.join(self.output_path, 'xy_plane_contour.png'),
                       dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_transition_dataset(self, data: Dict, filename: str):
        """Save extracted transition data in multiple formats."""
        
        # Save as JSON for metadata
        metadata_file = os.path.join(self.output_path, f'{filename}_metadata.json')
        
        metadata = {}
        for var in data:
            metadata[var] = {}
            for op in data[var]:
                metadata[var][op] = data[var][op]['metadata']
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data as TSV for each variable/operator combination
        for variable in data:
            for operator in data[variable]:
                points = data[variable][operator]['points']
                result = data[variable][operator]['data']
                
                tsv_filename = f'{filename}_{variable}_{operator}'
                write_interpolation_tsv_file(self.dataset, points, result, tsv_filename)
                
        print(f"Transition dataset saved with prefix: {filename}")


def main():
    """Main function with command-line interface for transition analysis."""
    parser = argparse.ArgumentParser(
        description='Extract 2D laminar-turbulent transition data from JHTDB'
    )
    
    # Required arguments
    parser.add_argument('--auth-token', required=True,
                       help='JHTDB authorization token')
    parser.add_argument('--output-path', required=True,
                       help='Output directory path')
    
    # Analysis mode
    parser.add_argument('--mode', 
                       choices=['profile', 'evolution', 'plane', 'comprehensive'],
                       default='comprehensive',
                       help='Type of 2D analysis to perform')
    
    # Optional parameters
    parser.add_argument('--time', type=float, default=1.0,
                       help='Time snapshot to analyze')
    parser.add_argument('--x-location', type=float, default=100.0,
                       help='Streamwise location for profile analysis')
    parser.add_argument('--y-location', type=float, default=2.0,
                       help='Wall-normal location for evolution analysis')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--save-data', default='transition_data',
                       help='Filename prefix for saved data')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TransitionExtractor2D(
        auth_token=args.auth_token,
        output_path=args.output_path
    )
    
    plot_enabled = not args.no_plot
    
    print(f"Starting {args.mode} analysis for laminar-turbulent transition...")
    
    if args.mode == 'profile' or args.mode == 'comprehensive':
        print("\n=== Boundary Layer Profile Analysis ===")
        profile_data = extractor.extract_boundary_layer_profile(
            time=args.time,
            x_location=args.x_location,
            variables=['velocity'],
            operators=['field', 'gradient']
        )
        
        if plot_enabled:
            extractor.plot_boundary_layer_profile(profile_data)
            
        if args.mode != 'comprehensive':
            extractor.save_transition_dataset(profile_data, f"{args.save_data}_profile")
    
    if args.mode == 'evolution' or args.mode == 'comprehensive':
        print("\n=== Streamwise Evolution Analysis ===")
        evolution_data = extractor.extract_streamwise_evolution(
            time=args.time,
            y_location=args.y_location,
            variables=['velocity'],
            operators=['field', 'gradient']
        )
        
        if plot_enabled:
            extractor.plot_streamwise_evolution(evolution_data)
            
        if args.mode != 'comprehensive':
            extractor.save_transition_dataset(evolution_data, f"{args.save_data}_evolution")
    
    if args.mode == 'plane' or args.mode == 'comprehensive':
        print("\n=== 2D Plane Analysis ===")
        plane_data = extractor.extract_xy_plane(
            time=args.time,
            variables=['velocity'],
            operators=['field']
        )
        
        if plot_enabled:
            extractor.plot_xy_plane_contour(plane_data)
            
        if args.mode != 'comprehensive':
            extractor.save_transition_dataset(plane_data, f"{args.save_data}_plane")
    
    if args.mode == 'comprehensive':
        # Combine all data for comprehensive analysis
        all_data = {
            'profile': profile_data,
            'evolution': evolution_data, 
            'plane': plane_data
        }
        
        # Save comprehensive dataset
        print("\n=== Saving Comprehensive Dataset ===")
        for analysis_type, data in all_data.items():
            extractor.save_transition_dataset(data, f"{args.save_data}_{analysis_type}")
    
    print("\n2D Transition analysis completed successfully!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()