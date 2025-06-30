# multiAHPy: A Flexible Python Library for Multi-Criteria Decision-Making

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A flexible, object-oriented Python library for performing both classical and Fuzzy Analytic Hierarchy Process (AHP). This toolkit is designed for researchers, data scientists, and decision-makers who require a robust, reproducible, and highly extensible tool for complex multi-criteria decision problems.

Its generic, protocol-based system allows it to seamlessly handle various numeric types, from crisp numbers to different fuzzy representations like **Triangular (TFN)**, **Trapezoidal (TrFN)**, **Intuitionistic (IFN)**, and more. This makes it a powerful tool for modeling the ambiguity and subjectivity inherent in human judgment.

Its core strength is a **"plug-and-play" architecture** built on registries and protocols. This allows users to easily add their own custom logic—new number types, weight derivation algorithms, consistency indices, or aggregation methods—without modifying the library's source code. This makes it an ideal tool for academic research, enabling the replication of past studies and the exploration of new methodologies.

*This library was developed under the supervision of Selim Gündüz, as part of the doctoral thesis: "Developing a fuzzy AHP-based index for measuring corporate responsibility at the local level: Corporate Local Responsibility (COLOR)", currently being conducted in the Department of Business Administration at Adana Alparslan Türkeş Science and Technology University.*

---

## Key Features

- **Generic & Type-Safe**: Easily switch between **Classical AHP (`Crisp`)** and **Fuzzy AHP (`TFN`, `TrFN`, etc.)** by changing a single type parameter.
- **Multi-Level Hierarchy**: Build complex decision models with unlimited criteria and sub-criteria levels using an intuitive `Node`-based structure.
- **Fully Extensible Architecture**: Easily register custom algorithms and parameters. Add your own weight derivation methods, consistency indices, aggregation techniques, or even custom fuzzy number types.
- **From Advice to Action**: The built-in MethodSuggester generates a complete, ready-to-use `recipe` of parameters for the chosen research design, ensuring methodological consistency.
- **Multiple Workflows**: The library provides clear, purpose-built `pipeline` to prevent common errors for both ranking a fixed set of alternatives (classic AHP) or using AHP as a weighting engine for a performance index.
- **Centralized Configuration**: Modify global parameters like Saaty's RI values, consistency thresholds, or fuzzy scales in a single configuration object (multiAHPy.configure_parameters) for easy customization and replication of studies.
- **Group Decision Support**: Aggregate judgments from multiple experts using standard methods like **Geometric Mean** or advanced techniques like **Intuitionistic Fuzzy Weighted Averaging (IFWA)**.
- **Multiple Derivation & Consistency Methods**: Implements a wide range of academically-cited algorithms:
  - **Weight Derivation**: Geometric Mean, Chang's Extent Analysis, Mikhailov's Fuzzy Programming.
  - **Consistency Analysis**: Saaty's CR (approximation), Geometric Consistency Index (GCI), and Mikhailov's Lambda.
- **Rich Visualizations**: Generate interactive HTML diagrams of your AHP hierarchy and create insightful Matplotlib plots for weights, rankings, and sensitivity analysis.
- **User-Friendly Data Input**: Easily create comparison matrices from simple Python lists or dictionaries, with automatic handling of reciprocals.


## Installation

Currently, the library can be installed by cloning the repository. (Future: PyPI installation).

```bash
git clone https://github.com/mberkancetin/fuzzy-ahp-color.git
cd fuzzy-ahp-color
pip install -r requirements.txt
```

## Core Concepts

The library is built around a few key concepts:

### The `Hierarchy` (AHPModel)
This is the main "engine" class that manages the entire AHP model, from the goal down to the alternatives.

### The `Node`
Everything in the hierarchy (the goal, criteria, sub-criteria) is a `Node`. A `Node` can have children, forming a tree structure of any depth.

### `NumericType` Protocol
This is the heart of the library's flexibility. Any number class (like `Crisp`, `TFN`, `TrFN`, `IFN`, etc.) that follows this protocol can be used in the model, and all calculations will adapt automatically.

### `Registries` Customization
Dictionaries that map names (e.g., `geometric_mean`) to functions. This allows for the "plug-and-play" architecture for methods.

### Streamlined `Pipelines`
With the combination of `recipe` method suggestion and `Workflow` class to define an entire analysis upfront, preventing common errors like skipping steps or using incompatible methods.


### Modules
- **`model`**: Core classes (`Hierarchy`, `Node`, `Alternative`).
- **`types`**: All supported number types (`Crisp`, `TFN`, `TrFN`, `IFN`, etc.) and the underlying `NumericType` protocol.
- **`config`**: A global configuration object holding all adjustable parameters like `SAATY_RI_VALUES` and `GCI_THRESHOLDS`.
- **`pipeline`**: The main entry point to create `Workflow` for running a full analysis.
- **`suggester`**: Helps generate method recipes (`get_model_recipe`) for your workflow.
- **`matrix_factory`**: Tools for creating comparison matrices from user input, including the `FuzzyScale` class.
- **`aggregation`**: Functions for group decision-making, like `aggregate_matrices` and `aggregate_priorities`.
- **`weight_derivation`**: The algorithms for calculating priority weights from matrices.
- **`defuzzification.py`**: Handles defuzzification methods (`centroid`, `graded_mean`, `alpha_cut`, etc.) for number types.
- **`consistency`**: The classes and functions for calculating `CR`, `GCI`, and other consistency metrics.
- **`validation`**: Tools to validate the structural integrity of a model before calculation.
- **`visualization`**: Functions for generating all visual outputs (HTML diagrams and Matplotlib plots).

## Contributing

Contributions are welcome! If you find a bug, have a feature request, or want to add a new algorithm, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
