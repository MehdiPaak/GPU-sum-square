# GPU-sum-square
OpenMP parallelization of multiple GPUs; the sum of square kernel is used to test the performance.
Performance is reported as wall-time, cpu-time and Flops.
several kernels are defined (inspired by literatures on GPU computing) to perform the sum of square (SoS) operations.

$X=\{x_i\}_{i=1}^N$

$SoS(X) = \sum_{i=1}^N x_i^2
