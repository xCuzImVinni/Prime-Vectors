# Prime-Vectors

## Project Documentation – Prime Factor Arithmetic System (PFS)

### Overview

This project implements a high-performance arithmetic system in Rust that reimagines how integers can be represented and manipulated. Instead of the traditional binary representation, this system encodes integers as **prime exponent vectors**—where each number is mapped to its unique prime factorization, stored as a vector of exponents.

This alternative representation enables elegant, vectorized implementations of arithmetic operations such as:

* **GCD (Greatest Common Divisor)** via componentwise `min`
* **LCM (Least Common Multiple)** via componentwise `max`
* **Multiplication** as vector addition
* **Division** as vector subtraction (when exact)

These operations become extremely efficient once numbers are already in this format.

---

### Algorithmic Design

The core idea is to compare three different GCD and LCM implementations:

1. **Native** – A standard binary GCD/LCM implementation in pure Rust.
2. **Hybrid** – A combination of native methods with prime-based optimizations.
3. **Pure (PFS)** – A fully PFS-based approach using SIMD acceleration and lookup tables.

Unfortunately, a pure PFS approach is currently limited by the overhead of integer factorization: converting a standard integer into its PFS form (i.e., factorizing) is computationally expensive and often not practical for large inputs. Many factorization algorithms lie in NP, making them inefficient for frequent runtime use.

That said, once the numbers are in PFS form, many operations (especially multiplication and division of exact multiples) become **significantly faster** and more predictable. Future advances in factorization — particularly via **Shor’s algorithm on quantum computers** — could make full adoption of PFS-based computation viable, especially for large integers.

---

### Benchmarking and Visualization

To evaluate performance, a benchmarking GUI was built using `eframe` and `egui`, including:

* Plotting runtime comparisons of all three styles (Native, Hybrid, Pure)
* Toggling visibility of individual curves
* Displaying metadata such as memory usage and cache hit rates
* A summary panel of system resources (e.g., memory consumption of LUTs, power tables, and prime lists)

The benchmarks span **bit widths from 4 to 32 bits**, each tested over **1000 iterations** per size, to provide statistically meaningful runtime data.

---

### Goals & Limitations

The main goals of the project were to:

* Explore novel number representations beyond base-2 encoding
* Implement PFS-based arithmetic in practice
* Compare hybrid algorithms against classic approaches
* Visualize empirical performance data in a clean and interactive GUI

The most significant bottleneck remains **conversion overhead**: the need to factor numbers to enter or exit PFS form. If this step could be eliminated or greatly accelerated, PFS would become a strong candidate for replacing traditional arithmetic in niche performance-critical applications.
