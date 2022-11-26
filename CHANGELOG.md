# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v2.0.0](https://github.com/cossio/CenteredRBMs.jl/releases/tag/v1.1.0)

- BREAKING: `pcd!` is now a method of `RestrictedBoltzmannMachines.pcd!` (instead of introducing a new function in this package).
- Add regularization (options `l2_fields, l1_weights, ...` to `pcd!`).
- Gauge fix (`zerosum`, `rescale` weights).

## [v1.1.0](https://github.com/cossio/CenteredRBMs.jl/releases/tag/v1.1.0)

- Introduce the CUDA helper functions: `cpu`, `gpu`, to transfer a model between devices.

## [v1.0.0](https://github.com/cossio/CenteredRBMs.jl/releases/tag/v1.0.0)

- This CHANGELOG file.
- Release v1.0.0 and registered package in General.