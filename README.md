# Deterministic GPU Math for ILGPU

This is a small, low-level deterministic math helper I made while using ILGPU in 2025.

This was designed to achieve deterministic bit parity for Log and Exp functions between GPU and CPU branches. I originally developed this as part of a larger personal project involving a very complex algorithm; it would exhaustively seed millions of samples via a Monte Carlo search to identify hard gates early, but also required the CPU branch for more efficiently handling smaller volumes thereafter. In that project, a single bit difference was enough to cause severe cumulative issues, so normal epsilon-based floating point checks were not an option.

The ultimate solution was to avoid relying on separate host and accelerator implementations (e.g., XMath). This class instead uses shared lookup tables, fixed-point intermediate math, explicit IEEE-754 handling, and deterministic round-to-even behavior so both execution paths follow the same calculation. It provides matching CPU and ILGPU-kernel implementations for:

- Pow
- Log
- Log2
- Sqrt
- Exp
- Exp2

Figuring this out was more of a mindfuck than I expected and sent me fairly deep into IEEE-754, FMA behavior, correctly rounded elementary functions, subnormals, and CPU/GPU numerical differences. This was run both on my local machine and my homelab (3xL40S GPUs!). 

- explicit IEEE-754 bit handling
- shared lookup tables for logarithm and exponent operations
- Q32.32 and Q1.31 fixed-point intermediate representations
- explicit round-to-nearest, ties-to-even behavior
- deterministic table interpolation
- canonical NaN handling
- explicit handling for zero, infinity, negative inputs, and subnormals
- equivalent overloads for normal managed arrays and ILGPU `ArrayView<T>` data

## ILGPU

I am a huge fan of [ILGPU](https://github.com/m4rs-mt/ILGPU) and have followed the project and its development in their Discord community since its first release. It is one of the more interesting .NET projects I use consistently for everything. Being able to write GPU kernels directly in C# and have it basically match TPL in structure is underrated. This repository is my own personal project and is not affiliated with or endorsed by ILGPU.

## References

- https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
- https://docs.nvidia.com/cuda/floating-point/
- https://standards.ieee.org/ieee/754/6210/
