#!/usr/bin/env python3
"""
Example usage of the 2D Laminar-Turbulent Transition Extractor

This script demonstrates how to use the specialized transition extractor
for various types of 2D boundary layer transition analysis.
"""

from transition_2d_extractor import TransitionExtractor2D
import numpy as np
import matplotlib.pyplot as plt
import os

def example_comprehensive_analysis():
    """
    Example: Comprehensive transition analysis workflow
    """
    # Configuration
    auth_token = 'edu.jhu.pha.turbulence.testing-201406'  # Replace with your token
    output_path = './transition_analysis_output'
    
    # Initialize extractor
    extractor = TransitionExtractor2D(auth_token, output_path)
    
    print("=== Comprehensive Laminar-Turbulent Transition Analysis ===\n")
    
    # 1. Boundary Layer Profile at Critical Location
    print("1. Extracting boundary layer profile at transition onset...")
    profile_data = extractor.extract_boundary_layer_profile(
        time=1.0,
        x_location=5.0,  # Critical streamwise location
        z_location=0.0,
        ny=80,
        y_range=(0.0, 1.5),
        variables=['velocity'],
        operators=['field', 'gradient']
    )
    
    # Plot the profile
    extractor.plot_boundary_layer_profile(profile_data, save_plot=True)
    
    # 2. Streamwise Evolution in Boundary Layer
    print("\n2. Extracting streamwise evolution in boundary layer...")
    evolution_data = extractor.extract_streamwise_evolution(
        time=1.0,
        y_location=0.2,  # Within boundary layer
        z_location=0.0,
        nx=200,
        x_range=(0.0, 15.0),
        variables=['velocity'],
        operators=['field', 'gradient']
    )
    
    # Plot streamwise evolution
    extractor.plot_streamwise_evolution(evolution_data, save_plot=True)
    
    # 3. 2D Plane Showing Transition Region
    print("\n3. Extracting 2D x-y plane...")
    plane_data = extractor.extract_xy_plane(
        time=1.0,
        z_location=0.0,
        nx=120,
        ny=80,
        x_range=(0.0, 12.0),
        y_range=(0.0, 1.5),
        variables=['velocity'],
        operators=['field']
    )
    
    # Plot 2D contour
    extractor.plot_xy_plane_contour(plane_data, component=0, save_plot=True)  # u-velocity
    extractor.plot_xy_plane_contour(plane_data, component=1, save_plot=True)  # v-velocity
    
    # 4. Compute transition indicators
    print("\n4. Computing transition indicators...")
    if 'velocity' in evolution_data:
        indicators = extractor.compute_transition_indicators(evolution_data['velocity'])
        
        # Plot turbulence intensity evolution
        x_coords = evolution_data['velocity']['field']['x_coords']
        
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x_coords, indicators['streamwise_velocity'], 'b-', linewidth=2)
        plt.ylabel('Streamwise Velocity (u)')
        plt.title('Transition Indicators vs Streamwise Position')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        if 'du_dy' in indicators:
            plt.plot(x_coords, indicators['du_dy'], 'r-', linewidth=2)
            plt.ylabel('Shear Rate (du/dy)')
        plt.xlabel('Streamwise Position (x)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'transition_indicators.png'), 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    # 5. Save all datasets
    print("\n5. Saving datasets...")
    extractor.save_transition_dataset(profile_data, 'transition_profile')
    extractor.save_transition_dataset(evolution_data, 'transition_evolution')
    extractor.save_transition_dataset(plane_data, 'transition_plane')
    
    print(f"\nAnalysis complete! Results saved to: {output_path}")
    return extractor, profile_data, evolution_data, plane_data

def example_multi_location_analysis():
    """
    Example: Analyze transition at multiple streamwise locations
    """
    auth_token = 'edu.jhu.pha.turbulence.testing-201406'
    output_path = './multi_location_analysis'
    
    extractor = TransitionExtractor2D(auth_token, output_path)
    
    print("=== Multi-Location Transition Analysis ===\n")
    
    # Analyze boundary layer profiles at multiple x-locations
    x_locations = [2.0, 4.0, 6.0, 8.0, 10.0]  # Streamwise positions
    
    plt.figure(figsize=(12, 8))
    
    for i, x_loc in enumerate(x_locations):
        print(f"Analyzing location x = {x_loc}")
        
        profile_data = extractor.extract_boundary_layer_profile(
            time=1.0,
            x_location=x_loc,
            ny=60,
            y_range=(0.0, 1.0),
            variables=['velocity'],
            operators=['field']
        )
        
        # Extract velocity profile
        y_coords = profile_data['velocity']['field']['y_coords']
        vel_data = np.array(profile_data['velocity']['field']['data'][0])
        u_velocity = vel_data[:, 0]  # Streamwise component
        
        # Plot profiles
        plt.subplot(2, 3, i+1)
        plt.plot(u_velocity, y_coords, 'b-', linewidth=2)
        plt.xlabel('u-velocity')
        plt.ylabel('y')
        plt.title(f'Profile at x = {x_loc}')
        plt.grid(True, alpha=0.3)
        
        # Save individual profile
        extractor.save_transition_dataset(profile_data, f'profile_x_{x_loc}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'multi_location_profiles.png'), 
               dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Multi-location analysis complete! Results saved to: {output_path}")

def example_time_series_analysis():
    """
    Example: Analyze transition evolution over time
    """
    auth_token = 'edu.jhu.pha.turbulence.testing-201406'
    output_path = './time_series_analysis'
    
    extractor = TransitionExtractor2D(auth_token, output_path)
    
    print("=== Time Series Transition Analysis ===\n")
    
    # Analyze evolution at multiple time points
    time_points = [0.5, 1.0, 1.5, 2.0]
    x_location = 6.0  # Fixed streamwise location
    y_location = 0.15  # Fixed wall-normal location
    
    plt.figure(figsize=(15, 10))
    
    for i, t in enumerate(time_points):
        print(f"Analyzing time t = {t}")
        
        # Streamwise evolution at this time
        evolution_data = extractor.extract_streamwise_evolution(
            time=t,
            y_location=y_location,
            nx=150,
            x_range=(0.0, 12.0),
            variables=['velocity'],
            operators=['field']
        )
        
        # Extract data
        x_coords = evolution_data['velocity']['field']['x_coords']
        vel_data = np.array(evolution_data['velocity']['field']['data'][0])
        u_velocity = vel_data[:, 0]
        
        # Plot evolution
        plt.subplot(2, 2, i+1)
        plt.plot(x_coords, u_velocity, 'r-', linewidth=2)
        plt.xlabel('Streamwise position (x)')
        plt.ylabel('u-velocity')
        plt.title(f'Evolution at t = {t}')
        plt.grid(True, alpha=0.3)
        
        # Save data
        extractor.save_transition_dataset(evolution_data, f'evolution_t_{t}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'time_series_evolution.png'), 
               dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Time series analysis complete! Results saved to: {output_path}")

if __name__ == "__main__":
    print("Choose analysis type:")
    print("1. Comprehensive transition analysis")
    print("2. Multi-location analysis")
    print("3. Time series analysis")
    print("4. Run all examples")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == '1':
        example_comprehensive_analysis()
    elif choice == '2':
        example_multi_location_analysis()
    elif choice == '3':
        example_time_series_analysis()
    elif choice == '4':
        print("Running all examples...\n")
        example_comprehensive_analysis()
        print("\n" + "="*50 + "\n")
        example_multi_location_analysis()
        print("\n" + "="*50 + "\n")
        example_time_series_analysis()
    else:
        print("Invalid choice. Running comprehensive analysis...")
        example_comprehensive_analysis()