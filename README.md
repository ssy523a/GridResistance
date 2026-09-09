# Lead-Acid Battery Grid Resistance Analyzer

Lead-Acid Battery Grid Resistance Analyzer is a Python-based engineering application for evaluating the electrical characteristics of lead-acid battery grids from 3D STEP geometry.

The program imports a grid model, generates a 3D tetrahedral mesh, and performs finite element analysis to evaluate the electrical behavior of the grid using material resistivity, operating current, and user-defined boundary conditions.

## Main Features

- Import 3D STEP files of lead-acid battery grids
- Interactive surface selection for lug and boundary conditions
- Automatic detection of grid dimensions and thickness direction
- 3D tetrahedral mesh generation using Gmsh
- Electrical conduction FEM analysis
- Grid resistance and voltage-drop evaluation
- Resistance distribution map relative to the lug
- Mid-plane visualization through the center of the grid thickness
- Current density analysis
- Joule loss density analysis
- Hotspot identification
- Material library with editable electrical resistivity
- Adjustable analysis current and reference potential
- Percentile-based contour display for improved visualization
- Separate analysis concepts for grid characterization and real-battery operating conditions
- Support for Grid and COS/Strap analysis modes

## Grid Characterization

The Grid Characterization results are intended to represent the electrical properties of the metallic grid itself and to support comparison with conventional laboratory grid measurements.

Available results include:

- Voltage Drop
- Resistance Map

The Resistance Map represents the local electrical resistance relative to the lug and is primarily determined by grid geometry, alloy resistivity, and the measurement configuration.

## Real Battery Model

The application also supports an engineering model intended to estimate electrical behavior when the grid is operating inside a lead-acid battery.

The model uses a distributed current-transfer assumption over the pasted active-material region instead of concentrating the return current at a single point.

Available results include:

- Voltage Drop in Real Battery Model
- Current Density in Real Battery Model
- Joule Loss Density in Real Battery Model

The current-transfer distribution can be approximated using a smooth Gaussian distribution, with stronger contribution near the central region of the pasted area and gradually decreasing contribution toward the edges.

This model is intended as an engineering approximation and does not directly simulate the full electrochemical reaction behavior of the positive or negative active material.

## FEM and Visualization

The electrical conduction problem is solved using a 3D finite element model.

The program provides:

- 3D surface visualization
- 3D volume visualization
- Mid-plane cross-sectional visualization
- Full-range and percentile-based contour scaling
- Linear and logarithmic display scales

The Mid-Plane view is particularly useful for examining electrical behavior inside grid frames, wires, and junctions rather than only on the external surface of the grid.

## Technologies

The application is developed in Python using:

- PySide6
- PyVista / PyVistaQt
- Gmsh
- NumPy
- scikit-fem

## Purpose

The program is intended as an engineering tool for:

- Lead-acid battery grid design evaluation
- Comparison of alternative grid geometries
- Identification of high-resistance regions
- Analysis of voltage-drop distribution
- Evaluation of current paths within the grid
- Identification of current-density and Joule-loss hotspots
- Supporting grid optimization for electrical performance

## Disclaimer

The Real Battery Model uses simplified assumptions for current transfer between the active material and the metallic grid. It should therefore be considered an engineering estimation tool rather than a complete electrochemical battery simulation.
