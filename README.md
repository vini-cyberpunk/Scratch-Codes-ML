# Scratch-Codes-ML

In this repository, I implement Machine Learning algorithms completely from scratch using only Python and NumPy to deeply understand their mathematical foundations, optimization procedures, and internal working mechanisms.

The goal is not just to use ML models, but to build them manually from first principles.

---

# Implemented Models

## 1. Linear Regression (Closed-Form Solution)

Implemented Ordinary Least Squares (OLS) Linear Regression using the Moore-Penrose pseudo-inverse.

### Features

- Multi-feature support
- Automatic bias/intercept handling
- Moore-Penrose pseudo-inverse based solution
- Fully vectorized NumPy implementation
- Feature validation during prediction
- Supports both 1D and 2D input arrays

### Mathematical Formulation

Closed-form solution:

$$
\theta = X^{+}y
$$

Prediction equation:

$$
\hat{y} = X\theta
$$

---

## 2. Linear Regression using Batch Gradient Descent

Implemented multivariable Linear Regression using Batch Gradient Descent optimization.

### Features

- Fully vectorized gradient descent
- Multi-feature support
- Automatic bias/intercept handling
- Configurable learning rate
- Configurable convergence tolerance
- Maximum iteration handling
- Runtime convergence warnings
- Feature validation during prediction
- Supports both 1D and 2D input arrays

### Mathematical Formulation

Prediction equation:

$$
\hat{y} = X\theta
$$

Cost function:

$$
J(\theta)=\frac{1}{2n}\|X\theta-y\|^2
$$

Gradient:

$$
\nabla J(\theta)=\frac{1}{n}X^T(X\theta-y)
$$

Gradient Descent update rule:

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

Equivalent update equation:

$$
\theta := \theta - \alpha \frac{1}{n}X^T(X\theta-y)
$$

---

## 3. Linear Regression using Stochastic Gradient Descent (SGD)

Implemented multivariable Linear Regression using Stochastic Gradient Descent optimization.

### Features

- Stochastic Gradient Descent optimization
- Per-sample weight updates
- Random shuffling every epoch
- Multi-feature support
- Automatic bias/intercept handling
- Configurable learning rate
- Configurable convergence tolerance
- Maximum iteration handling
- Runtime convergence warnings
- Feature validation during prediction
- Supports both 1D and 2D input arrays

### Mathematical Formulation

Prediction equation:

$$
\hat{y}_i = x_i^T\theta
$$

Cost function:

$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Per-sample gradient:

$$
\nabla J_i(\theta)=-(y_i-\hat{y}_i)x_i
$$

SGD update rule:

$$
\theta := \theta - \alpha \nabla J_i(\theta)
$$

Equivalent update equation:

$$
\theta := \theta + \alpha (y_i-\hat{y}_i)x_i
$$

---

## 4. Linear Regression using Mini-Batch Gradient Descent

Implemented multivariable Linear Regression using Mini-Batch Gradient Descent optimization.

### Features

- Mini-Batch Gradient Descent optimization
- Random shuffling every epoch
- Configurable mini-batch size
- Multi-feature support
- Automatic bias/intercept handling
- Configurable learning rate
- Configurable convergence tolerance
- Maximum iteration handling
- Runtime convergence warnings
- Fully vectorized batch updates
- Feature validation during prediction
- Supports both 1D and 2D input arrays

### Mathematical Formulation

Prediction equation:

$$
\hat{y} = X\theta
$$

Mini-batch cost function:

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2
$$

Mini-batch gradient:

$$
\nabla J(\theta)=\frac{1}{m}X^T(X\theta-y)
$$

Mini-Batch Gradient Descent update rule:

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

Equivalent update equation:

$$
\theta := \theta - \alpha \frac{1}{m}X^T(X\theta-y)
$$

---

# Learning Roadmap

```text
========================
REGRESSION MODELS
========================

✅ 1. Linear Regression (Closed Form)

✅ 2. Linear Regression using Batch Gradient Descent

✅ 3. Linear Regression using SGD

✅ 4. Linear Regression using Mini-Batch Gradient Descent

⬜ 5. Learning Rate Schedules

⬜ 6. Regularization
      ├── Ridge Regression (L2)
      └── Lasso Regression (L1)

⬜ 7. Kernel Regression

⬜ 8. PCA


========================
CLASSIFICATION MODELS
========================

⬜ 9. Logistic Regression

⬜ 10. Naive Bayes

⬜ 11. K-Nearest Neighbors (KNN)

⬜ 12. Support Vector Classifier (SVC)

⬜ 13. Decision Trees

⬜ 14. Random Forests
```

---

# Technologies Used

- Python
- NumPy

---

# Repository Structure

```text
Scratch-Codes-ML/
│
├── linear_regression.py
├── linear_regression_gd.py
├── sgd_regressor.py
├── mini_batch_gd_regressor.py
├── README.md
└── LICENSE
```

---

# Goals of This Repository

- Understand Machine Learning mathematically
- Implement ML algorithms without ML libraries
- Learn vectorized numerical computing
- Build intuition for optimization algorithms
- Understand gradient-based learning
- Develop strong fundamentals in ML and numerical methods

---

# Learning Philosophy

This repository focuses on:
- mathematical derivations
- vectorized implementations
- numerical optimization
- understanding how algorithms work internally
- building ML models from scratch instead of relying on high-level libraries

---

# Author

Vinit Agrawal
