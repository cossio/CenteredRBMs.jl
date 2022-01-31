# CenteredRBMs.jl Documentation

A Julia package to train and simulate *centered* Restricted Boltzmann Machines.
See <http://jmlr.org/papers/v17/14-237.html> for the definition of *centered*.

This package does not export any symbols.

Most of the functions have a helpful docstring.
See [Reference](@ref) section.

## Installation

This package is not registered.
Install with:

```julia
import Pkg
Pkg.add(url="https://github.com/cossio/CenteredRBMs.jl")
```

## Related

This package is based on the [RestrictedBoltzmannMachines](https://github.com/cossio/RestrictedBoltzmannMachines.jl) Julia package, which defines the `RBM` and layer types.
We refer to `RestrictedBoltzmannMachines` by the shorter name `RBMs`, as if it were imported by the line

```julia
import RestrictedBoltzmannMachines as RBMs
```