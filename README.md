# Scratch-Codes-ML

In this repository, I implement Machine Learning algorithms completely from scratch using only Python and NumPy to deeply understand their mathematical foundations, optimization procedures, and internal working mechanisms.

The goal is not just to use ML models, but to build them manually from first principles.

---

# Implemented Models

## 1. Linear Regression (Closed-Form Solution)

Implemented Ordinary Least Squares (OLS) Linear Regression using the Normal Equation / Moore-Penrose pseudo-inverse.

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
\theta = X^+ y
$$

Prediction equation:

$$
\hat{y} = X\theta
$$

---

## 2. Linear Regression using Gradient Descent

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

Gradient Descent Update Rule:

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

Equivalent update equation:

$$
\theta := \theta - \alpha \frac{1}{n}X^T(X\theta-y)
$$

---

# Technologies Used

- Python
- NumPy

---

# Goals of This Repository

- Understand Machine Learning mathematically
- Implement ML algorithms without ML libraries
- Learn vectorized numerical computing
- Build intuition for optimization algorithms
- Understand gradient-based learning
- Develop strong fundamentals in ML and numerical methods

---

# Future Implementations

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- Naive Bayes
- Principal Component Analysis (PCA)
- Neural Networks
- Backpropagation from scratch
- Optimizers (SGD, Momentum, Adam)

---

# Repository Structure

```text
Scratch-Codes-ML/
│
├── LinearRegression/
│   ├── closed_form/
│   └── gradient_descent/
│
├── tests/
├── README.md
└── requirements.txt
```

---

# Learning Philosophy

This repository focuses on:

- Mathematical derivations
- Vectorized implementations
- Numerical optimization
- Understanding how algorithms actually work internally
- Writing ML code without relying on high-level ML frameworks

---

# Author

Vinit Agrawal
