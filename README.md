# SKA-Laminar-Turbulent: Real-Time Information Geometry for Fluid Flow

**A universal tool for detecting the loss of predictability in complex systems, powered by Structured Knowledge Accumulation (SKA).**


> **Key Insight:**  
> SKA uncovers the real-time “information geometry” of fluid flows—quantifying when a system is knowable, when it becomes unpredictable, and how knowledge evolves—**directly from the data, with no need for solving the Navier–Stokes equations.**

## Overview

This repository demonstrates the use of **SKA (Structured Knowledge Accumulation)** for real-time analysis of time series from fluid flows governed by the Navier–Stokes equations.
We focus on detecting and visualizing the transition from **laminar (predictable) flow to turbulence (chaotic, unpredictable)**—a fundamental challenge in physics and engineering.

## Why SKA?

Traditional tools require modeling and simulation of the full equations of motion, often with high computational cost and expert tuning. **SKA, by contrast, is a data-driven approach:** it detects predictability loss, regime transitions, and hidden information structure by simply “watching” the data—making it universal and equation-free.

- **Laminar regime:** SKA entropy is low and periodic (high predictability)
- **Transition:** SKA entropy rises sharply (loss of predictability)
- **Turbulence:** SKA entropy remains high and irregular (chaos)

## Example Use Case

- **Input:**  
  Velocity or pressure time series from an experiment or simulation showing laminar–turbulent transition.
- **Output:**  
  Plots of SKA entropy and knowledge, phase-locked to laminar flow, and spiking as turbulence begins.

## Folder Structure

- `data/` — Example datasets (or scripts to download/simulate)
- `notebooks/` — Jupyter notebooks for SKA analysis and plotting
- `src/` — Core SKA functions (existing SKA modules)
- `figures/` — Publication-quality figures showing entropy during transitions

## Getting Started

1. Place or download a time series with a laminar-turbulent transition in the `data/` folder.
2. Open and run the notebook in `notebooks/` to compute and visualize SKA entropy and knowledge.
3. Interpret results: SKA highlights when the system is predictable, and when it loses order.

## Impact

- **First open-source demonstration** of information geometry for flow regime detection
- Enables new science in **physics, engineering, and data science**
- Generalizes to **any complex time series**: climate, market regimes, biological rhythms, and more

**Contact / Collaborate:**  
Pull requests, issues, and datasets welcome!

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