# Benchmark: Randomized vs Full SVD in scikit-learn

## Overview

This repository provides a detailed comparison of the **Singular Value Decomposition (SVD)** algorithms available in scikit-learn, specifically the **standard (full)** method and the **randomized** approach. The research focuses on their performance in terms of accuracy, stability, and computation time for large, high-dimensional datasets.

## Key Findings and Recommendations

1. **Optimal Use Cases for Randomized SVD**:
   - Use the **randomized algorithm** when the number of components is less than **20% of the input features**.
   - For scenarios where the number of components is below **5%**, the **relative error** can be exceptionally low, while retaining a significant **performance advantage** over the standard algorithm.

2. **Sparse Matrices**:
   - For sparse matrices with a density greater than **2%**, prefer the **randomized algorithm** over ARPACK.
   - For large sparse matrices (e.g., **50,000 × 4,000** or larger), the randomized algorithm is significantly faster than both the standard method and ARPACK.

3. **Stability**:
   - Always use a **normalizer** to ensure stability during computation.
   - The **LU normalizer** is recommended as it consistently provides better stability.

## Applications

- **Dimensionality reduction**: Efficiently reduce dimensions for large datasets while maintaining accuracy.
- **Feature extraction**: Ideal for identifying key features in high-dimensional datasets.
- **Sparse data analysis**: Provides speed and accuracy advantages over traditional iterative methods like ARPACK for certain matrix types.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Scikit-learn-intelex
- Matplotlib
- Scipy

Install dependencies using:

```bash
pip install -r requirements.txt
