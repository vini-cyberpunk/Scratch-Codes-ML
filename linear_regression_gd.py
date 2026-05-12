import numpy as np
import warnings

class LinearRegressionGD():
	def __init__(self, lr=0.001, niter=100, tol=1e-4):
		self.lr = lr
		self.niter = niter
		self.tol = tol
		self.weights = None
		self.d = None

	def fit(self, X, y):
		# reshaping from vector to array
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if y.ndim == 1:
			y = y.reshape(-1,1)
			
		# number of samples and features
		n = X.shape[0]
		self.d = X.shape[1]
		
		# adding bias term
		X_b = np.c_[np.ones((n,1)), X]
		
		# initializing weights with bias term and updating them using gradient descent
		self.weights = np.zeros((self.d+1,1))
		y_pred = X_b @ self.weights
		error = (1/n)*np.linalg.norm(y - y_pred)
		
		# updating weights
		i=0
		while (i < self.niter and error > self.tol):
			gradient = (1/n) * X_b.T @ (y_pred - y)
			self.weights = self.weights - self.lr * gradient
			y_pred = X_b @ self.weights
			error = (1/n)*np.linalg.norm(y - y_pred)
			i+=1
		
		# throw warning if iteration not converged
		if (i == self.niter) and (error > self.tol):
			warnings.warn("Maximum iteration reached before convergence!!", RuntimeWarning)
		
	def predict(self, X_test):
		# ensuring prediction done only after fiitting
		if self.weights is None:
			raise ValueError(
				f"Model has not been fitted yet."
			)
			
		# reshaping from vector to array
		if X_test.ndim == 1:
			X_test = X_test.reshape(-1,1)
			
		# validating test data have same features as train data.
		d = X_test.shape[1]
		if d != self.d:
			raise ValueError(
				f"Model fitted using {self.d} features but given data contains {d} features."
			)
			
		# number of samples
		n = X_test.shape[0]
		
		# adding bias term	
		X_test_b = np.c_[np.ones((n,1)), X_test]
		
		# predicting using computed weights
		y_pred = X_test_b @ self.weights
		
		# returning predcted values as 1-D vector
		return y_pred.ravel()
