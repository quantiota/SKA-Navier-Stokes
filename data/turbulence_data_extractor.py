#!/usr/bin/env python3
"""
Turbulence Data Extractor

Converts JHTDB notebook functionality into a Python script for extracting
turbulence datasets, particularly for laminar-turbulent transition analysis.

Based on the DEMO_Getdata_local.ipynb notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os
from typing import List, Tuple, Optional, Union

# Import the givernylocal library
from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, write_interpolation_tsv_file


class TurbulenceDataExtractor:
    """Main class for extracting turbulence data from JHTDB datasets."""
    
    def __init__(self, auth_token: str, dataset_title: str, output_path: str):
        """
        Initialize the turbulence data extractor.
        
        Args:
            auth_token: JHTDB authorization token
            dataset_title: Name of the turbulence dataset
            output_path: Directory where output files will be saved
        """
        self.auth_token = auth_token
        self.dataset_title = dataset_title
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize dataset
        print(f"Initializing dataset: {dataset_title}")
        self.dataset = turb_dataset(
            dataset_title=dataset_title,
            output_path=output_path,
            auth_token=auth_token
        )
        
    def extract_2d_plane_data(self, 
                             time: float = 1.0,
                             nx: int = 64, 
                             nz: int = 64,
                             x_range: Tuple[float, float] = (0.0, 0.4 * np.pi),
                             y_coord: float = 0.9,
                             z_range: Tuple[float, float] = (0.0, 0.15 * np.pi),
                             variable: str = 'velocity',
                             temporal_method: str = 'none',
                             spatial_method: str = 'lag8',
                             spatial_operator: str = 'field',
                             plot: bool = True) -> Tuple[np.ndarray, List]:
        """
        Extract data from a 2D plane.
        
        Args:
            time: Time to query
            nx, nz: Number of points along x and z axes
            x_range, z_range: Min/max coordinates for x and z axes
            y_coord: Fixed y coordinate for the plane
            variable: Type of data to extract
            temporal_method: Temporal interpolation method
            spatial_method: Spatial interpolation method
            spatial_operator: Spatial operator to apply
            plot: Whether to generate contour plot
            
        Returns:
            Tuple of (points array, result data)
        """
        print(f"Extracting 2D plane data: {nx}x{nz} points")
        
        # Generate point grid
        x_points = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
        z_points = np.linspace(z_range[0], z_range[1], nz, dtype=np.float64)
        
        points = np.array([
            axis.ravel() for axis in np.meshgrid(x_points, y_coord, z_points, indexing='ij')
        ], dtype=np.float64).T
        
        # Extract data
        result = getData(self.dataset, variable, time, temporal_method, 
                        spatial_method, spatial_operator, points)
        
        print(f"Extracted data for {len(points)} points")
        
        # Generate plot if requested
        if plot and nx >= 2 and nz >= 2:
            self._plot_2d_contour(points, result, x_points, z_points, nx, nz, 
                                variable, spatial_operator)
        
        return points, result
    
    def extract_3d_volume_data(self,
                              time: float = 1.0,
                              nx: int = 16,
                              ny: int = 16, 
                              nz: int = 16,
                              x_range: Tuple[float, float] = (3.0, 3.3),
                              y_range: Tuple[float, float] = (-0.9, -0.6),
                              z_range: Tuple[float, float] = (0.2, 0.5),
                              variable: str = 'velocity',
                              temporal_method: str = 'none',
                              spatial_method: str = 'lag8',
                              spatial_operator: str = 'field',
                              plot: bool = True) -> Tuple[np.ndarray, List]:
        """
        Extract data from a 3D volume.
        
        Args:
            time: Time to query
            nx, ny, nz: Number of points along each axis
            x_range, y_range, z_range: Min/max coordinates for each axis
            variable: Type of data to extract
            temporal_method: Temporal interpolation method
            spatial_method: Spatial interpolation method
            spatial_operator: Spatial operator to apply
            plot: Whether to generate volume plot
            
        Returns:
            Tuple of (points array, result data)
        """
        print(f"Extracting 3D volume data: {nx}x{ny}x{nz} points")
        
        # Generate point grid
        x_points = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
        y_points = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
        z_points = np.linspace(z_range[0], z_range[1], nz, dtype=np.float64)
        
        points = np.array([
            axis.ravel() for axis in np.meshgrid(x_points, y_points, z_points, indexing='ij')
        ], dtype=np.float64).T
        
        # Extract data
        result = getData(self.dataset, variable, time, temporal_method,
                        spatial_method, spatial_operator, points)
        
        print(f"Extracted data for {len(points)} points")
        
        # Generate plot if requested
        if plot and nx >= 2 and ny >= 2 and nz >= 2:
            self._plot_3d_volume(points, result, x_points, y_points, z_points, 
                                nx, ny, nz, variable, spatial_operator)
        
        return points, result
    
    def extract_random_points_data(self,
                                  time: float = 1.0,
                                  n_points: int = 1000,
                                  min_xyz: List[float] = [6.1359, -0.61359, 0.6],
                                  max_xyz: List[float] = [21.8656, 0.8656, 8.8656],
                                  variable: str = 'velocity',
                                  temporal_method: str = 'none',
                                  spatial_method: str = 'lag8',
                                  spatial_operator: str = 'field',
                                  plot: bool = True) -> Tuple[np.ndarray, List]:
        """
        Extract data from randomly distributed points.
        
        Args:
            time: Time to query
            n_points: Number of random points
            min_xyz, max_xyz: Min/max coordinates for random distribution
            variable: Type of data to extract
            temporal_method: Temporal interpolation method
            spatial_method: Spatial interpolation method
            spatial_operator: Spatial operator to apply
            plot: Whether to generate scatter plot
            
        Returns:
            Tuple of (points array, result data)
        """
        print(f"Extracting random points data: {n_points} points")
        
        # Generate random points
        points = np.random.uniform(
            low=[min_xyz[0], min_xyz[1], min_xyz[2]],
            high=[max_xyz[0], max_xyz[1], max_xyz[2]],
            size=(n_points, 3)
        )
        
        # Extract data
        result = getData(self.dataset, variable, time, temporal_method,
                        spatial_method, spatial_operator, points)
        
        print(f"Extracted data for {len(points)} points")
        
        # Generate plot if requested
        if plot:
            self._plot_random_scatter(points, result, variable, spatial_operator)
        
        return points, result
    
    def extract_time_series_data(self,
                                time_start: float = 0.1,
                                time_end: float = 0.5,
                                delta_t: float = 0.008,
                                points: Optional[np.ndarray] = None,
                                variable: str = 'velocity',
                                temporal_method: str = 'pchip',
                                spatial_method: str = 'lag8',
                                spatial_operator: str = 'field',
                                plot: bool = True) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        Extract time series data at specific points.
        
        Args:
            time_start: Start time
            time_end: End time
            delta_t: Time step
            points: Spatial points (if None, uses default point)
            variable: Type of data to extract
            temporal_method: Temporal interpolation method
            spatial_method: Spatial interpolation method
            spatial_operator: Spatial operator to apply
            plot: Whether to generate time series plot
            
        Returns:
            Tuple of (points array, result data, times array)
        """
        print(f"Extracting time series data from {time_start} to {time_end}")
        
        if points is None:
            points = np.array([[10.33, 0.9, 4.6]], dtype=np.float64)
        
        option = [time_end, delta_t]
        
        # Extract data
        result, times = getData(self.dataset, variable, time_start, temporal_method,
                               spatial_method, spatial_operator, points, option, 
                               return_times=True)
        
        print(f"Extracted time series for {len(points)} points over {len(times)} time steps")
        
        # Generate plot if requested
        if plot and variable != 'position':
            self._plot_time_series(times, result, points, variable)
        
        return points, result, times
    
    def save_data(self, points: np.ndarray, result: List, filename: str):
        """
        Save extracted data to TSV file.
        
        Args:
            points: Spatial points array
            result: Result data from getData
            filename: Output filename (without extension)
        """
        print(f"Saving data to {filename}.tsv")
        write_interpolation_tsv_file(self.dataset, points, result, filename)
        
    def _plot_2d_contour(self, points, result, x_points, z_points, nx, nz, 
                        variable, spatial_operator):
        """Generate 2D contour plot."""
        time_component = 0
        plot_component = 0
        
        # Reformat result for plotting
        result_time = result[time_component]
        result_array = np.array(result_time)
        data_plot = np.zeros((nx, nz, len(result_time.columns)), dtype=np.float32)
        x_plot = np.zeros((nx, nz), dtype=np.float64)
        z_plot = np.zeros((nx, nz), dtype=np.float64)
        
        for i in range(nx):
            for j in range(nz):
                x_plot[i, j] = x_points[i]
                z_plot[i, j] = z_points[j]
                data_plot[i, j, :] = result_array[i * nz + j, :]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(7, 7), dpi=67)
        contour = ax.contourf(x_plot, z_plot, data_plot[:, :, plot_component],
                             levels=500, cmap='inferno')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(contour, shrink=0.67)
        cbar.set_label(f'{variable} {spatial_operator}', labelpad=12, fontsize=14)
        
        plt.title(f'{self.dataset_title} ({result[time_component].columns[plot_component]})', 
                 fontsize=16)
        plt.xlabel('x', labelpad=7, fontsize=14)
        plt.ylabel('z', labelpad=7, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, '2d_contour_plot.png'), dpi=150)
        plt.show()
    
    def _plot_3d_volume(self, points, result, x_points, y_points, z_points,
                       nx, ny, nz, variable, spatial_operator):
        """Generate 3D volume plot."""
        time_component = 0
        plot_component = 0
        max_axis_points = 16
        
        # Calculate strides
        strides = np.ceil(np.array([nx, ny, nz]) / max_axis_points).astype(np.int32)
        
        # Reformat result for plotting
        result_time = result[time_component]
        result_array = np.array(result_time)
        data_plot = np.zeros((nx, ny, nz, len(result_time.columns)), dtype=np.float32)
        x_plot = np.zeros((nx, ny, nz), dtype=np.float64)
        y_plot = np.zeros((nx, ny, nz), dtype=np.float64)
        z_plot = np.zeros((nx, ny, nz), dtype=np.float64)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x_plot[i, j, k] = x_points[i]
                    y_plot[i, j, k] = y_points[j]
                    z_plot[i, j, k] = z_points[k]
                    data_plot[i, j, k, :] = result_array[i * ny * nz + j * nz + k, :]
        
        # Create volume plot
        fig = go.Figure(data=go.Volume(
            x=x_plot[::strides[0], ::strides[1], ::strides[2]].flatten(),
            y=y_plot[::strides[0], ::strides[1], ::strides[2]].flatten(),
            z=z_plot[::strides[0], ::strides[1], ::strides[2]].flatten(),
            value=data_plot[::strides[0], ::strides[1], ::strides[2], plot_component].flatten(),
            isomin=data_plot[::strides[0], ::strides[1], ::strides[2], plot_component].min(),
            isomax=data_plot[::strides[0], ::strides[1], ::strides[2], plot_component].max(),
            opacity=0.2, surface_count=17, colorscale='inferno'
        ))
        
        fig.update_layout(
            title=f'{self.dataset_title} 3D Volume',
            scene=dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                aspectmode='data'
            ),
            width=600, height=600
        )
        
        fig.write_html(os.path.join(self.output_path, '3d_volume_plot.html'))
        fig.show()
    
    def _plot_random_scatter(self, points, result, variable, spatial_operator):
        """Generate scatter plot and histogram for random points."""
        time_component = 0
        plot_component = 0
        max_scatter_points = 1000
        bins = 20
        
        # Calculate stride
        stride = max(1, len(points) // max_scatter_points)
        
        x_plot, y_plot, z_plot = points[:, 0], points[:, 1], points[:, 2]
        data_plot = np.array(result[time_component])[:, plot_component]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=('3D Scatter', 'Histogram'),
            column_widths=[0.5, 0.5]
        )
        
        # Scatter plot
        scatter = go.Scatter3d(
            x=x_plot[::stride], y=y_plot[::stride], z=z_plot[::stride],
            mode='markers',
            marker=dict(size=4, color=data_plot[::stride], colorscale='inferno', opacity=0.8)
        )
        
        # Histogram
        histogram = go.Histogram(x=data_plot, nbinsx=bins, marker_color='forestgreen')
        
        fig.add_trace(scatter, row=1, col=1)
        fig.add_trace(histogram, row=1, col=2)
        
        fig.update_layout(
            title=f'{self.dataset_title} Random Points Analysis',
            height=600, width=1200
        )
        
        fig.write_html(os.path.join(self.output_path, 'random_scatter_plot.html'))
        fig.show()
    
    def _plot_time_series(self, times, result, points, variable):
        """Generate time series plot."""
        point_component = 0
        plot_component = 0
        
        data_plot = np.array(result)[:, point_component][:, plot_component]
        
        fig, ax = plt.subplots(figsize=(7, 7), dpi=67)
        ax.plot(times, data_plot, color='forestgreen', linewidth=3)
        
        plt.title(f'{self.dataset_title} Time Series at xyz = '
                 f'({points[0][0]:.2f}, {points[0][1]:.2f}, {points[0][2]:.2f})', 
                 fontsize=16)
        plt.xlabel('time', labelpad=7, fontsize=14)
        plt.ylabel(f'{variable}', labelpad=7, fontsize=14)
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'time_series_plot.png'), dpi=150)
        plt.show()


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Extract turbulence data from JHTDB')
    
    # Required arguments
    parser.add_argument('--auth-token', required=True,
                       help='JHTDB authorization token')
    parser.add_argument('--dataset', required=True,
                       help='Dataset name (e.g., transition_bl, channel)')
    parser.add_argument('--output-path', required=True,
                       help='Output directory path')
    
    # Optional arguments
    parser.add_argument('--mode', choices=['2d', '3d', 'random', 'timeseries'], 
                       default='2d', help='Extraction mode')
    parser.add_argument('--variable', default='velocity',
                       help='Variable to extract (velocity, pressure, etc.)')
    parser.add_argument('--time', type=float, default=1.0,
                       help='Time to query')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--save-data', 
                       help='Save data to TSV file with given filename')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = TurbulenceDataExtractor(
        auth_token=args.auth_token,
        dataset_title=args.dataset,
        output_path=args.output_path
    )
    
    # Extract data based on mode
    plot_enabled = not args.no_plot
    
    if args.mode == '2d':
        points, result = extractor.extract_2d_plane_data(
            time=args.time,
            variable=args.variable,
            plot=plot_enabled
        )
    elif args.mode == '3d':
        points, result = extractor.extract_3d_volume_data(
            time=args.time,
            variable=args.variable,
            plot=plot_enabled
        )
    elif args.mode == 'random':
        points, result = extractor.extract_random_points_data(
            time=args.time,
            variable=args.variable,
            plot=plot_enabled
        )
    elif args.mode == 'timeseries':
        points, result, times = extractor.extract_time_series_data(
            variable=args.variable,
            plot=plot_enabled
        )
    
    # Save data if requested
    if args.save_data:
        extractor.save_data(points, result, args.save_data)
    
    print("Data extraction completed successfully!")


if __name__ == "__main__":
    main()