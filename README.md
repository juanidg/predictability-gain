# predictability-gain
This repository contains Python code to analyze temporal correlations and memory effects in discrete sequences using information-theoretic tools.
The code is organized into two independent modules.

# 1. Exact entropy and predictability gain (pred_gain_exact.py)
This module computes exact block entropies and predictability gain components for a fully specified discrete stochastic process.
You provide (or randomly generate within the code) the transition probabilities of an i.i.d. or Markov process, and the code:
- computes block entropies for increasing block lengths,
- computes predictability gain components,
- estimates the entropy-rate slope,
- and provides plotting utilities for visualization.

This module is mainly intended for theoretical analysis, validation on synthetic models, and comparison with data-driven estimates.

# 2. Memory estimation from data (PG_memory_estimator.py)

This module estimates the effective memory of a discrete sequence.
You only need to provide a sequence S. The code:
- measures predictability gain from the data,
- generates bootstrap surrogate sequences,
- tests increasing candidate memory orders,
- and returns the smallest order consistent with the data.

For convenience, the code also includes utilities to generate discrete i.i.d. and Markov sequences of known memory order.
These generators are provided for testing and illustration only and are not required to use the estimator.

# Dependencies
- Python ≥ 3.8
- NumPy
- ndd (NSB entropy estimator)
- Matplotlib (only for plotting in the exact module)

# Reference
This repository accompanies the paper:

De Gregorio, J., Sánchez, D., & Toral, R. (2025). Information-theoretic analysis of temporal dependence in discrete stochastic processes: Application to precipitation predictability. arXiv preprint arXiv:2510.11276.
