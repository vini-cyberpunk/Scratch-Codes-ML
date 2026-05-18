import numpy as np
import warnings

class LassoRegression():
	def __init__(self, alpha=1, max_iter=100, tol=1e-4):
		self.alpha = alpha
		self.weights = None
		self.d = None
		self.max_iter = max_iter
		self.tol = tol
		
	@staticmethod
	def soft_thresholding(rho_j, alpha):
		if rho_j > alpha:
			return (rho_j - alpha)
		elif rho_j < -alpha:
			return (rho_j + alpha)
		else:
			return 0
	
	def fit(self, X, y):
		# reshaping the vector to array
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if y.ndim == 1:
			y = y.reshape(-1,1)
		
		n, self.d = X.shape
		
		# adding bias term in X and weights
		X_b = np.c_[np.ones((n, 1)), X]
		current_weights = np.zeros((self.d+1, 1))
		
		# sum of square of all X_j elements in each data-points
		z = np.sum(X_b ** 2, axis=0)

		# iterative method to find the weights for L1 regularization
		for i in range(self.max_iter):
			prev_weights = current_weights.copy()
			for j in range(self.d + 1):
				if z[j] == 0:
					continue
				
				y_pred = X_b @ current_weights
				r_j = y - y_pred + current_weights[j] * X_b[:, j:j+1]
				rho_j = (r_j.T @ X_b[:, j:j+1]).item()
				
				if j == 0:
					current_weights[j] = rho_j / z[j]
				else:
					current_weights[j] = self.soft_thresholding(rho_j, self.alpha) / z[j]
			
			if np.max(np.abs(current_weights - prev_weights)) < self.tol:
				break
		else:
			warnings.warn( f"Model has not converged yet, max_iter = {self.max_iter}." )
		
		self.weights = current_weights
		
	def predict(self, X_test):
		# reshaping the vector to array
		if X_test.ndim == 1:
			X_test = X_test.reshape(-1,1)
		
		n, d = X_test.shape
		
		if self.weights is None:
			raise ValueError (
				f"Model has not been fitted yet."
			)
		
		if self.d != d:
			raise ValueError (
				f"Model fitted using {self.d} features but {d} features are given."
			)
		
		# adding bias term in X_test
		X_test_b = np.c_[np.ones((n, 1)), X_test]
		y_pred = X_test_b @ self.weights
		return y_pred.ravel()
