import numpy as np

class RidgeRegression():
	def __init__(self, alpha=1):
		self.alpha = alpha
		self.weights = None
		self.d = None
		
	def fit(self, X, y):
		# reshaping the vector to array
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if y.ndim == 1:
			y = y.reshape(-1,1)
		
		n, self.d = X.shape
		
		# adding bias term in X and weights
		X_b = np.c_[np.ones((n, 1)), X]
		
		# finding Coeff using L2 penalty regularizer
		identity = np.identity(self.d+1)
		identity[0,0] = 0
		self.weights = np.linalg.inv(X_b.T @ X_b + self.alpha * identity) @ X_b.T @ y
		
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
