# SKA-Laminar-Turbulent: Real-Time Information Geometry for Fluid Flow

**A universal tool for detecting the loss of predictability in complex systems, powered by Structured Knowledge Accumulation (SKA).**



## Overview

This repository demonstrates the use of **SKA (Structured Knowledge Accumulation)** for real-time analysis of time series from fluid flows governed by the Navier–Stokes equations.
We focus on detecting and visualizing the transition from **laminar (predictable) flow to turbulence (chaotic, unpredictable)**—a fundamental challenge in physics and engineering.


## Why SKA?

Traditional methods reveal only the shape or statistical properties of a flow.
**SKA exposes the hidden information structure**—it tracks entropy, knowledge, and information Lagrangian in real time, directly from the data, without prior models.

* **Laminar regime:** SKA entropy is low and periodic (high predictability)
* **Transition:** SKA entropy rises sharply (loss of predictability)
* **Turbulence:** SKA entropy remains high and irregular (chaos)


## Example Use Case

* **Input:**
  Velocity or pressure time series from an experiment or simulation showing laminar–turbulent transition.
* **Output:**
  Plots of SKA entropy and knowledge, phase-locked to laminar flow, and spiking as turbulence begins.


## Folder Structure

* `data/` — Example datasets (or scripts to download/simulate)
* `notebooks/` — Jupyter notebooks for SKA analysis and plotting
* `src/` — Core SKA functions (existing SKA modules)
* `figures/` — Publication-quality figures showing entropy during transitions


## Getting Started

1. Place or download a time series with a laminar-turbulent transition in the `data/` folder.
2. Open and run the notebook in `notebooks/` to compute and visualize SKA entropy and knowledge.
3. Interpret results: SKA highlights when the system is predictable, and when it loses order.


## Impact

* **First open-source demonstration** of information geometry for flow regime detection
* Useful for physics, engineering, and anyone studying complex time series



**Contact / Collaborate:**
Pull requests, issues, and datasets welcome!

