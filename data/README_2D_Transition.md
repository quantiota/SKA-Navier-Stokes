# 2D Laminar-Turbulent Transition Data Extractor

This specialized tool extracts 2D datasets from the JHTDB transition boundary layer dataset for analyzing laminar-turbulent transition phenomena.

## Files Created

1. **`transition_2d_extractor.py`** - Main extractor class with specialized 2D analysis methods
2. **`example_transition_analysis.py`** - Example usage scripts showing different analysis workflows
3. **`turbulence_data_extractor.py`** - General-purpose turbulence data extractor (original conversion)

## Key Features for Transition Analysis

### 2D Analysis Methods

1. **Boundary Layer Profiles** - Extract wall-normal velocity profiles at specific streamwise locations
2. **Streamwise Evolution** - Track flow development along the boundary layer at fixed heights
3. **2D X-Y Planes** - Visualize complete transition regions with contour plots
4. **Transition Indicators** - Compute shear rates, vorticity, and turbulence intensity

### Specialized Parameters

- **Dataset**: Uses `transition_bl` (transitional boundary layer) from JHTDB
- **High-order interpolation**: `lag8` method for accurate gradient calculations
- **Multiple variables**: velocity, pressure with field/gradient/laplacian operators
- **Transition-optimized grids**: Fine resolution in critical boundary layer regions

## Quick Start

1. **Command Line Usage**:
```bash
# Comprehensive analysis (recommended for first use)
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode comprehensive

# Specific analysis types
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode profile --x-location 5.0
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode evolution --y-location 0.1
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode plane
```

2. **Interactive Examples**:
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

- **TSV files**: Raw data for each variable/operator combination
- **PNG plots**: Boundary layer profiles, streamwise evolution, contour plots
- **HTML plots**: Interactive 3D visualizations (where applicable)
- **JSON metadata**: Analysis parameters and grid information

## Typical Workflow for Transition Studies

1. **Start with comprehensive analysis** to understand overall flow structure
2. **Use profile mode** to identify critical streamwise locations
3. **Apply evolution mode** at identified critical heights
4. **Generate plane visualizations** for publication-quality figures
5. **Compute transition indicators** for quantitative analysis

## Key Parameters for Transition Analysis

- **Streamwise range**: 0-15 domain units (covers pre-transition to fully turbulent)
- **Wall-normal range**: 0-2 domain units (includes entire boundary layer)
- **Grid resolution**: 100-200 points streamwise, 80-100 wall-normal
- **Critical locations**: x=5-8 for transition onset, y=0.1-0.3 for core boundary layer

## Authentication

Get your JHTDB auth token from: http://turbulence.pha.jhu.edu/
Replace `'edu.jhu.pha.turbulence.testing-201406'` with your actual token.

## Dependencies

```bash
pip install givernylocal numpy matplotlib plotly
```

The `givernylocal` library provides JHTDB access and must be installed separately from the JHTDB website.