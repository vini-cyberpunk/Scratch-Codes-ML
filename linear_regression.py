import numpy as np

class LinearRegression():
	def __init__(self):
		self.weights = None
		self.d = None	#number of features
		
	def fit(self, X, y):
		# reshaping from vector to array
		if X.ndim == 1:
			X = X.reshape(-1,1)
		if y.ndim == 1:
			y = y.reshape(-1,1)
			
		#number of samples
		n = X.shape[0]
		self.d = X.shape[1]
		
		# adding bias term
		X_b = np.c_[np.ones((n,1)), X]
		
		#computing closed form solution of weights including bias at index 0
		self.weights = np.linalg.pinv(X_b) @ y
		
	def predict(self, X_test):
		# ensuring prediction done only after fiitting
		if self.weights is None:
			raise ValueError(
				f"Model has not been fitted yet."
			)
			
		# reshaping from vector to array
		if X_test.ndim == 1:
			X_test = X_test.reshape(-1,1)
			
		#validating test data have same features as train data.
		d = X_test.shape[1]
		if d != self.d:
			raise ValueError(
				f"Model fitted using {self.d} features but given data contains {d} features."
			)
			
		#number of samples
		n = X_test.shape[0]
		
		#adding bias term	
		X_test_b = np.c_[np.ones((n,1)), X_test]
		
		#predicting using computed weights
		y_pred = X_test_b @ self.weights
		
		# returning predcted values as 1-D vector
		return y_pred.ravel()
