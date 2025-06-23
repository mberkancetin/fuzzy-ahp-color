# fuzzy-ahp-color: A Python Library for Multi-Criteria Decision-Making

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A flexible, object-oriented Python library for performing both classical and Fuzzy Analytic Hierarchy Process (AHP). This toolkit is designed for researchers, data scientists, and decision-makers who need a robust and extensible framework for solving complex multi-criteria decision problems, especially in group settings.

The library's core design is built on a generic, protocol-based system, allowing it to seamlessly handle various numeric types, from crisp numbers to different fuzzy representations like Triangular (TFN), Trapezoidal (TrFN), and Gaussian (GFN) fuzzy numbers.

*This library was developed under the supervision of Selim Gündüz, as part of the doctoral thesis: "Developing a fuzzy AHP-based index for measuring corporate responsibility at the local level: Corporate Local Responsibility (COLOR)", currently being conducted in the Department of Business Administration at Adana Alparslan Türkeş Science and Technology University.*

---

## Key Features

- **Generic & Flexible**: Easily switch between **Classical AHP** and **Fuzzy AHP** by simply changing the number type (`Crisp`, `TFN`, `TrFN`, etc.).
- **Object-Oriented Design**: A clear, maintainable structure with classes for the `Hierarchy`, `Node`, and `Alternative`.
- **Group Decision Support**: Includes robust methods (`geometric`, `arithmetic`) for aggregating judgments from multiple experts.
- **Multiple Weight Derivation Methods**: Supports well-established algorithms like **Fuzzy Geometric Mean** and **Chang's Extent Analysis**.
- **Comprehensive Validation**: Tools to check for matrix reciprocity, diagonal integrity, and consistency ratio (CR) to ensure data quality.
- **Rich Visualizations**: Generate intuitive HTML diagrams of your AHP hierarchy and create insightful plots with Matplotlib for weights, rankings, and sensitivity analysis.
- **User-Friendly Matrix Creation**: Easily create comparison matrices from simple Python lists or dictionaries of judgments.

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
This is the heart of the library's flexibility. Any number class (like `Crisp`, `TFN`, `TrFN`) that follows this protocol can be used in the model, and all calculations will adapt automatically.

### Modules
- **`matrix_builder.py`**: Tools for creating comparison matrices from user input.
- **`aggregation.py`**: Functions to combine judgments from multiple participants.
- **`model.py`**: Core classes (`Hierarchy`, `Node`, `Alternative`).
- **`types.py`**: Defines all number types (`Crisp`, `TFN`, etc.) and the `NumericType` protocol.
- **`defuzzification.py`**: Handles defuzzification methods (`centroid`, `graded_mean`, `alpha_cut`, etc.) for number types.
- **`weight_derivation.py`**: Algorithms for calculating priority weights.
- **`consistency.py`**: Functions to calculate and analyze the Consistency Ratio (CR).
- **`validation.py`**: Tools to validate the structure and data of the model.
- **`visualization.py`**: Functions for generating HTML diagrams and Matplotlib plots.

## Contributing

Contributions are welcome! If you find a bug, have a feature request, or want to add a new algorithm, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
