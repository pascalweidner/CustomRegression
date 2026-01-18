# Linear Models: Probabilistic Foundations and Implementation

This project explores the mathematical derivations and implementations of various linear regression models, transitioning from standard Maximum Likelihood Estimation (MLE) to Maximum A Posteriori (MAP) estimation for regularization.

## 📂 Project Structure

The project is organized into a modular Python package with specific implementations for different statistical assumptions based on the following directory structure:

* **linear_models/**:
    * `_linear.py`: Normal Linear Regression.
    * `_robust.py`: Robust Regression using Laplace distributions.
    * `_ridge.py`: Ridge Regression with Gaussian priors.
    * `_lasso.py`: Lasso Regression with Laplace priors.
    * `_l4.py`: Regression with L4-norm regularization.
    * `_poisson.py`: Poisson Regression for count data.
    * `_logistic.py`: Additional generalized models.
* **overview.ipynb**: Demonstration of model fitting and visualization.

## 🛠 Implemented Models

### 1. Robust Regression
Derived by assuming the target $y_{i}$ follows a **Laplace Distribution** centered at the model prediction.
* **Optimization**: Since the absolute value function is not differentiable at zero, the problem is reformulated as a linear program using slack variables $e_{i}$.
* **Constraint**: $e_{i} \ge |w^{\top}\phi(x_{i}) - y_{i}|$.





### 2. Regularized Regression (Ridge & Lasso)
These models address the problem of overfitting in high-degree polynomial basis functions.
***Ridge Regression**: Assumes a Gaussian Likelihood and an Isotropic Gaussian prior with mean zero.
* **Lasso Regression**: Assumes a Gaussian Likelihood and a Laplace Prior.
* **Lasso Optimization**: Solved via **Coordinate Descent** using a soft thresholding function because the $L_{1}$ penalty is non-differentiable at $w_{j}=0$.



### 3. L4 Regression
Assumes a **Generalized Normal Distribution** prior with shape parameter $p=4$
* **Loss Function**: $E_{L_{4}}(w) = \frac{1}{2}||y-Xw||_{2}^{2} + \lambda||w||_{4}^{4}$.
* **Optimization**: Solved using the **L-BFGS-B** algorithm.

### 4. Poisson Regression
Designed to predict counts where variance is not constant.
* **Link Function**: Uses a log link function to connect the linear model to the distribution: $\lambda_{i} = \exp(w^{\top}x_{i})$.
* **Optimization**: Minimized using the log-likelihood and optimized via the L-BFGS-B algorithm.