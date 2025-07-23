# SKA Navier-Stokes Turbulence Data Extractors

Comprehensive toolkit for extracting turbulence datasets from JHTDB for Structured Knowledge Accumulation (SKA) machine learning research and laminar-turbulent transition analysis.

## Repository Contents

### Core Extractors
1. **`transition_2d_extractor.py`** - Specialized 2D laminar-turbulent transition analysis
2. **`streaming_extractor.py`** - High-resolution time series extraction for QuestDB streaming
3. **`ska_batch_extractor.py`** - Large-scale batch extraction for SKA ML training (3500+ samples)
4. **`turbulence_data_extractor.py`** - General-purpose JHTDB data extractor
5. **`multi_dataset_extractor.py`** - Combine multiple datasets (for testing token limitations)

### Utilities & Examples
6. **`check_dataset_times.py`** - Analyze time resolution across different JHTDB datasets
7. **`example_transition_analysis.py`** - Interactive examples for transition analysis workflows
8. **`requirements.txt`** - Python dependencies list

### Generated Data Examples
- **`results/`** - 2D transition analysis outputs (profiles, evolution, contours)
- **`streaming_data/`** - QuestDB-ready streaming datasets
- **`ska_training_data/`** - Large-scale training datasets for machine learning

## Key Features

### SKA Machine Learning Applications
- **Large-scale datasets**: Extract 3500+ samples for neural network training
- **High temporal resolution**: Up to 10 Hz equivalent time series
- **Multi-point monitoring**: 20+ spatial locations with 3D velocity vectors
- **QuestDB integration**: Real-time streaming data format
- **Batch processing**: Efficient extraction of massive datasets

### 2D Transition Analysis
- **Boundary Layer Profiles** - Wall-normal velocity profiles at streamwise locations
- **Streamwise Evolution** - Track transition development along boundary layer
- **2D X-Y Planes** - Complete transition region visualization
- **Transition Indicators** - Shear rates, vorticity, turbulence intensity

### Dataset Coverage
- **`transition_bl`**: Laminar-turbulent boundary layer transition
- **`isotropic1024coarse`**: High-resolution 3D isotropic turbulence (10 Hz)
- **`channel`**: Wall-bounded turbulent channel flow
- **`mixing`**: Homogeneous buoyancy-driven turbulence
- **Multi-dataset**: Combine multiple sources for comprehensive training

## Quick Start

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Get JHTDB Access Token**:
   - Visit: http://turbulence.pha.jhu.edu/
   - Register for research access (recommended for large datasets)
   - Testing token available: `edu.jhu.pha.turbulence.testing-201406`

3. **Usage Examples**:

### SKA Machine Learning Training
```bash
# Extract 3500 samples for SKA training (requires research token)
python ska_batch_extractor.py --auth-token "your-research-token" --output-path "./ska_training" --samples 3500 --points 20

# Check dataset time resolutions
python check_dataset_times.py

# High-resolution streaming data
python streaming_extractor.py --auth-token "your-token" --dataset isotropic1024coarse --output-path "./streaming" --n-points 10 --time-steps 100
```

### 2D Transition Analysis
```bash
# Comprehensive boundary layer transition analysis
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode comprehensive

# Specific analysis types
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode profile --x-location 100.0
python transition_2d_extractor.py --auth-token "your-token" --output-path "./results" --mode evolution --y-location 2.0
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

## Authentication & Research Access

### Testing vs Research Tokens
- **Testing Token**: `edu.jhu.pha.turbulence.testing-201406`
  - Limited time ranges (50-100 samples per dataset)
  - Good for initial testing and development
  - Daily query limits
  
- **Research Token**: Register at http://turbulence.pha.jhu.edu/
  - Full temporal ranges (1000s of samples available)
  - Required for SKA training datasets (3500+ samples)
  - Higher priority access and fewer limitations
  - Recommended for serious research applications

### Token Request Template
When requesting research access, mention:
- SKA (Structured Knowledge Accumulation) machine learning research
- Large-scale turbulence dataset requirements (3500+ samples)
- Institution: UJF, Quantiota
- GitHub: [Your repository URL]

## Dependencies

```bash
pip install givernylocal numpy matplotlib plotly
```

The `givernylocal` library provides JHTDB access and must be installed separately from the JHTDB website.

## Citing SKA

If you use SKA Navier-Stokes, please cite:

* Bouarfa Mahi.
  **Structured Knowledge Accumulation: An Autonomous Framework for Layer-Wise Entropy Reduction in Neural Learning**
  [arXiv:2503.13942](https://arxiv.org/abs/2503.13942)
* Bouarfa Mahi.
  **Structured Knowledge Accumulation: The Principle of Entropic Least Action in Forward-Only Neural Learning**
  [arXiv:2504.03214](https://arxiv.org/abs/2504.03214)

```
@article{mahi2025ska1,
  title={Structured Knowledge Accumulation: An Autonomous Framework for Layer-Wise Entropy Reduction in Neural Learning},
  author={Mahi, Bouarfa},
  journal={arXiv preprint arXiv:2503.13942},
  year={2025}
}
@article{mahi2025ska2,
  title={Structured Knowledge Accumulation: The Principle of Entropic Least Action in Forward-Only Neural Learning},
  author={Mahi, Bouarfa},
  journal={arXiv preprint arXiv:2504.03214},
  year={2025}
}
```