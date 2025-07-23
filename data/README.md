# 2D Laminar-Turbulent Transition Data Extractor

This specialized tool extracts 2D datasets from the JHTDB transition boundary layer dataset for analyzing laminar-turbulent transition phenomena.

## Files Created

1. **`transition_2d_extractor.py`** - Main extractor class with specialized 2D analysis methods
2. **`example_transition_analysis.py`** - Example usage scripts showing different analysis workflows
3. **`turbulence_data_extractor.py`** - General-purpose turbulence data extractor (original conversion)
4. **`requirements.txt`** - Python dependencies list

## Key Features for Transition Analysis

### 2D Analysis Methods

1. **Boundary Layer Profiles** - Extract wall-normal velocity profiles at specific streamwise locations
2. **Streamwise Evolution** - Track flow development along the boundary layer at fixed heights
3. **2D X-Y Planes** - Visualize complete transition regions with contour plots
4. **Transition Indicators** - Compute shear rates, vorticity, and turbulence intensity

### Specialized Parameters

- **Dataset**: Uses `transition_bl` (transitional boundary layer) from JHTDB
- **Valid domain**: x=[30.2, 1000.0], y=[0.0, 26.5], z=[0, 240] (periodic)
- **Spatial methods**: 'none' and 'fd4noint' for robust extraction
- **Multiple variables**: velocity, pressure with field/gradient/laplacian operators
- **Transition-optimized grids**: Fine resolution in critical boundary layer regions

## Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Command Line Usage**:
```bash
# Working example - boundary layer profile analysis
python transition_2d_extractor.py --auth-token "edu.jhu.pha.turbulence.testing-201406" --output-path "./results" --mode profile --x-location 100.0

# Other analysis types (use valid coordinates)
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode evolution --y-location 2.0
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode plane
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode comprehensive
```

3. **Interactive Examples**:
```bash
python example_transition_analysis.py
# Choose from: comprehensive, multi-location, or time-series analysis
```

## Analysis Modes

### Profile Mode
- Extracts wall-normal velocity profiles
- Perfect for analyzing boundary layer shape evolution
- Key for identifying transition onset locations

### Evolution Mode  
- Tracks streamwise development at fixed wall-normal distances
- Ideal for studying transition propagation
- Shows instability growth rates

### Plane Mode
- 2D visualization of entire transition regions
- Reveals turbulent spots and transition zones
- Best for overall flow field understanding

### Comprehensive Mode
- Combines all analysis types
- Provides complete transition characterization
- Recommended for thorough studies

## Output Files

For each analysis, the extractor generates:

- **TSV files**: Raw data for each variable/operator combination
  - `*_velocity_field.tsv` - Velocity field data  
  - `*_velocity_gradient.tsv` - Velocity gradient data
- **PNG plots**: Boundary layer profiles, streamwise evolution, contour plots
  - `boundary_layer_profile.png` - Wall-normal velocity profile
  - `streamwise_evolution.png` - Streamwise development  
  - `xy_plane_contour.png` - 2D contour visualization
- **JSON metadata**: Analysis parameters and grid information
  - `*_metadata.json` - Complete analysis configuration

## Typical Workflow for Transition Studies

1. **Start with comprehensive analysis** to understand overall flow structure
2. **Use profile mode** to identify critical streamwise locations
3. **Apply evolution mode** at identified critical heights
4. **Generate plane visualizations** for publication-quality figures
5. **Compute transition indicators** for quantitative analysis

## Key Parameters for Transition Analysis

- **Streamwise range**: 30.2-1000 domain units (covers entire transition region)
- **Wall-normal range**: 0-26.5 domain units (includes entire boundary layer)
- **Grid resolution**: 50-200 points streamwise, 50-100 wall-normal
- **Critical locations**: x=100-500 for transition development, y=1-10 for boundary layer core
- **Time snapshots**: Use t=0.0 as starting point (most reliable)

## Authentication

Get your JHTDB auth token from: http://turbulence.pha.jhu.edu/  
The testing token `'edu.jhu.pha.turbulence.testing-201406'` works for basic testing, but get your own token for extended use.

## Dependencies

```bash
pip install givernylocal numpy matplotlib plotly
```

The `givernylocal` library provides JHTDB access and must be installed separately from the JHTDB website.