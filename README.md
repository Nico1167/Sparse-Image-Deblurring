# Advanced Image Restoration: Sparse Deconvolution via Proximal Gradient Methods

This repository implements an inverse problem framework to solve 2D image deblurring and deconvolution using sparsity-promoting optimization. By transposing the problem into the wavelet domain, natural images can be reconstructed from heavily blurred observations. The project builds custom proximal gradient algorithms from scratch, analyzing their mathematical structures and benchmarking their practical convergence rates against optimal theoretical limits.

---

## Key Features & Methods Implemented

* **Sparsity-Promoting L1 Regularization:** Formulates image restoration as a composite optimization problem, perfectly balancing data fidelity with sparse structural priors.
* **Discrete Wavelet Transform (DWT2D):** Integrates 2D Haar wavelets to compress and sparsify image representations, shifting the optimization workspace into the wavelet coefficient domain where image signals exhibit maximum sparsity.
* **Iterative Shrinkage-Thresholding Algorithm (ISTA):** Implements the foundational proximal gradient method from scratch, utilizing a soft-thresholding operator to handle the non-smooth L1 norm penalty.
* **Fast Iterative Shrinkage-Thresholding Algorithm (FISTA):** Implements Nesterov's accelerated proximal gradient scheme to dramatically boost performance using a dynamic momentum-based step update.
* **Linear Operator Framework:** Employs high-performance linear operators for 2D Spatial Convolution and the Inverse Wavelet Transform to form the composite matrix-free operator.

---

## Key Findings & Insights

### 1. Optimization Complexity & Convergence Mechanics
* **Theoretical vs. Practical Rates:** Empirical testing verifies that the baseline proximal gradient method (ISTA) converges at a sublinear rate of $\mathcal{O}(1/k)$, whereas the accelerated variant (FISTA) successfully matches Nesterov's optimal theoretical bound of $\mathcal{O}(1/k^2)$.
* **Spectral Radius & Step Size:** The gradient step size $\alpha$ is governed by the inverse of the largest eigenvalue of the operator $A^H A$, computed via power iterations, ensuring strict numerical stability and consistent cost reduction.

### 2. Image Deblurring & Structural Restoration
* **Sparsity-Driven Denoising:** Operating within the wavelet domain proves highly effective; the soft-thresholding operator isolates high-frequency blur artifacts while recovering sharp, distinct structural boundaries.
* **Regularization Balance:** The hyperparameter $\epsilon$ directly dictates image sharpness. While an excessive $\epsilon$ forces over-smoothing and loss of fine features, a well-calibrated value yields pristine detail recovery from degraded structural matrices.

---

## Repository Structure

* `deblurring_final.py`: Core optimization engine containing data loaders, operator chains, custom ISTA/FISTA loops, and visual benchmarking charts.
* `chateau.npy`: Grayscale image data matrix representing a castle used to benchmark regularized restoration algorithms.
* `dog_rgb.npy`: Grenscale image data matrix representing a dog used to benchmark regularized restoration algorithms.
* `report_deblurring_final.pdf`: Report detailing the complete mathematical proofs, objective function analysis, and algorithmic performance evaluations (in French).

---

## Requirements & Installation

Ensure you have Python installed alongside the following scientific dependencies:

```bash
pip install numpy matplotlib pylops
