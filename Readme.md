# Space-Time Multigrid Method for Space-Time Finite Element Discretizations of Parabolic and Hyperbolic PDEs

This repository contains the implementation of a space-time multigrid method based on tensor-product space-time finite element discretizations described in the paper "A Space-Time Multigrid Method for Space-Time Finite Element Discretizations of Parabolic and Hyperbolic PDEs" by Nils Margenberg and Peter Munch. The method leverages the matrix-free capabilities of the `deal.II` library and supports both high-order continuous and discontinuous variational time discretizations with spatial finite element discretizations. The implementation consistently performs well on perturbed meshes and can handle heterogeneous, discontinuous coefficients. We have demonstrated a throughput of over a billion degrees of freedom per second on problems with more than a trillion global degrees of freedom.

## Usage

The code depends on a recent version of the `deal.II` library. We recommend using version `9.5.2` or a more recent one. To run the examples and replicate the results the scripts may need to be adapted to your system.

## License

This project is licensed under the Apache License, Version 2.0 WITH LLVM-exception.
## Authors

- Nils Margenberg
- Peter Munch
