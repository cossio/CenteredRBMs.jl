# CenteredRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/CenteredRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/CenteredRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/CenteredRBMs.jl/branch/master/graph/badge.svg?token=90I3AJIZIG)](https://codecov.io/gh/cossio/CenteredRBMs.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/CenteredRBMs.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/CenteredRBMs.jl)

Train and sample centered Restricted Boltzmann machines in Julia. See [Melchior et al] for the definition of *centered*. Consider an RBM with binary units. Then the centered variant has energy defined by:

$$
E(v,h) = -\sum_i a_i v_i - \sum_\mu b_\mu h_\mu - \sum_{i\mu} w_{i\mu} (v_i - c_i) (h_\mu - d_\mu)
$$

with offset parameters $c_i,d_\mu$. Typically $c_i,d_\mu$ are set to approximate the average activities of $v_i$ and $h_\mu$, respectively, as this seems to help training (see [Montavon et al]).

## Installation

This package is registered. Install with:

```julia
import Pkg
Pkg.add("CenteredRBMs")
```

This package does not export any symbols.

## Related

[RestrictedBoltzmannMachines](https://github.com/cossio/RestrictedBoltzmannMachines.jl), which defines `RBM` and layer types.

## References

Montavon, Grégoire, and Klaus-Robert Müller. "Deep Boltzmann machines and the centering trick." Neural networks: tricks of the trade. Springer, Berlin, Heidelberg, 2012. 621-637.

Melchior, Jan, Asja Fischer, and Laurenz Wiskott. "How to center deep Boltzmann machines." The Journal of Machine Learning Research 17.1 (2016): 3387-3447.

## Citation

If you use this package in a publication, please cite:

* Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." Physical Review X 13, 021003 (2023).

Or you can use the included [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).