# CenteredRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/CenteredRBMs.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/CenteredRBMs.jl/dev)
![](https://github.com/cossio/CenteredRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/CenteredRBMs.jl/branch/master/graph/badge.svg?token=90I3AJIZIG)](https://codecov.io/gh/cossio/CenteredRBMs.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/CenteredRBMs.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/CenteredRBMs.jl)

Train and sample centered [Restricted Boltzmann machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia.
See <http://jmlr.org/papers/v17/14-237.html> for the definition of *centered*.

## Installation

This package is not registered.
Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/CenteredRBMs.jl")
```

This package does not export any symbols.

## Related

Builds upon the [RestrictedBoltzmannMachines](https://github.com/cossio/RestrictedBoltzmannMachines.jl) Julia package, which defines the `RBM` and layer types.