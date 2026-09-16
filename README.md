# Deterministic GPU Math for ILGPU

A small deterministic math helper I built for an ILGPU-based personal project where CPU-side calculations and GPU kernels needed to produce bit-identical results.

This was a mindfuck originally when I had to do a deep debug into the inner workings of ILGPU to figure out how to get the CPU and GPU in parity, beyond just the simple FMA. This was necessary given the exact precision I needed for a personal project that also involved a very complex computational algorithm. Even a one-bit difference was enough to break downstream comparisons, so normal epsilon-based floating-point checks were not an option.

The ultimate solution was to avoid relying on separate host and accelerator implementations (e.g., XMath). This class instead uses shared lookup tables, fixed-point intermediate math, explicit IEEE-754 handling, and deterministic round-to-even behavior so both execution paths follow the same calculation. It provides matching CPU and ILGPU-kernel implementations for:

Pow
Log
Log2
Sqrt
Exp
Exp2

Figuring this out was considerably more involved than I expected and sent me fairly deep into IEEE-754, FMA behavior, correctly rounded elementary functions, subnormals, and CPU/GPU numerical differences. This was run both in production environemnts for my local machine and my homelab (3xL40S GPUs!). 

- explicit IEEE-754 bit handling
- shared lookup tables for logarithm and exponent operations
- Q32.32 and Q1.31 fixed-point intermediate representations
- explicit round-to-nearest, ties-to-even behavior
- deterministic table interpolation
- canonical NaN handling
- explicit handling for zero, infinity, negative inputs, and subnormals
- equivalent overloads for normal managed arrays and ILGPU `ArrayView<T>` data

## ILGPU

I am a huge fan of [ILGPU](https://github.com/m4rs-mt/ILGPU) and have followed the project and its development in their Discord community since its first release. It is one of the more interesting .NET projects I use consistently for everything. Being able to write GPU kernels directly in C# and have it basically match TPL in structure is underrated.

## References

- David Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). The classic overview of floating-point representation, rounding, error, and IEEE-754 behavior.
- Nathan Whitehead and Alex Fit-Florea, NVIDIA, [Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://docs.nvidia.com/cuda/floating-point/). Particularly relevant to CPU/GPU differences, FMA, rounding modes, mathematical-function accuracy, and verification in heterogeneous systems.
- [IEEE 754-2019: Standard for Floating-Point Arithmetic](https://standards.ieee.org/ieee/754/6210/). The underlying standard for formats, arithmetic, rounding, exceptions, NaNs, infinities, and related behavior.
- Pierre Fortin, Mourad Gouicem, and Stef Graillat, [Correctly Rounding Elementary Functions on GPU](https://www-sop.inria.fr/marelle/Laurent.Thery/fast2018/mourad.pdf). Useful context for why correctly rounded `log`/`exp`-style functions on GPUs are considerably harder than they first appear.
- [CORE-MATH references](https://core-math.gitlabpages.inria.fr/references.html). A useful collection of work on correctly rounded elementary functions, including CR-LIBM, RLIBM, and related research.
