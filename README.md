# scrtt
Tools for downstream analysis of **s**ingle **c**ell **r**eal-**t**ime **t**rajectories.

This package implements methods for interfacing with trajectory models for time-resolved single-cell data. Some commonly used analysis frameworks in the field include:

* Optimal transport (OT)
* Normalizing flows (TODO)
* Recurrent Neural Networks (TODO)

So far, the functions in this module only interface with OT models, but I plan to add more soon.

Downstream analysis of single-cell real-time trajectories often involves tasks such as estimating cell densities or computing expected values at specific time points given a starting population of cells. Models may implement these functions differently, or not at all. The goal of this package is to create a consistent API for common tasks in the downstream analysis of single-cell trajectories.

# Installation

```
pip install git+https://github.com/alecplotkin/scrtt
```

# Quick start
TODO
