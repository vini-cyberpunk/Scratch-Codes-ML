import numpy as np
import warnings

class MiniBatchGDRegressor():
	def __init__(self, batch_size=0.20, lr=0.001, max_iter=100, tol=1e-4):
		self.batch_size = batch_size
		self.weights = None
		self.lr = lr
		self.max_iter = max_iter
		self.tol = tol
		self.d = None
		
	def fit(self, X, y):
		# reshaping the dataset
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if y.ndim == 1:
			y = y.reshape(-1,1)
		
		# storing shape of dataset
		n, self.d = X.shape
		
		# adding bias feature
		X_b = np.c_[np.ones((n,1)), X]
		
		# initializing the weights with bias term
		self.weights = np.zeros((self.d+1, 1))
		y_pred = X_b @ self.weights
		prev_error = np.mean((y - y_pred)**2)
		# setting up the loop for weights update
		for _ in range(self.max_iter):
			shuffled_indices = np.random.permutation(n)
			X_shuffled = X_b[shuffled_indices]
			y_shuffled = y[shuffled_indices]
			
			m = max(1, int(n * self.batch_size))		# no of data points in each batch.
			no_of_batches = int(np.ceil(n / m))
			
			for i in range(no_of_batches):
				start = i*m
				end = min( (i+1)*m, n)
				X_i = X_shuffled[ start : end, : ]
				y_i = y_shuffled[ start : end, : ]
				current_batch_size = X_i.shape[0]
				
				y_pred_i = X_i @ self.weights
				error_i = y_pred_i - y_i
				gradient = (1/current_batch_size) * X_i.T @ error_i
				self.weights = self.weights - self.lr * gradient

			y_pred = X_b @ self.weights
			error = np.mean((y - y_pred)**2)
			if np.abs(error - prev_error) < self.tol:
				break
			prev_error = error
			
		else:
			warnings.warn(
				f"failed to converge after {self.max_iter} iterations", RuntimeWarning
			)
				
	def predict(self, X_test):
		if self.weights is None:
			raise ValueError (
				"Model has not been fitted yet."
			)
			
		# reshaping the dataset
		if X_test.ndim == 1:
			X_test = X_test.reshape(-1,1)
			
		# shape of test data 
		n, d = X_test.shape

		if d != self.d:
			raise ValueError (
				f"Model fitted with {self.d} features but {d} features are given."
			)
			
		# adding bias features
		X_test_b = np.c_[np.ones((n,1)), X_test]
		
		# prediction
		y_pred_test = X_test_b @ self.weights
		
		return y_pred_test.ravel()
			
				
				
