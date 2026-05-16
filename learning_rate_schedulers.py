class StepDecayScheduler():
	def __init__(self, eta0=0.1, decay_factor=0.1, step_size=10):
		self.eta0 = eta0
		self.decay_factor = decay_factor
		self.step_size = step_size
		
	def get_lr(self, epoch):
		eta = self.eta0 * (self.decay_factor ** (epoch // self.step_size))
		return eta
		
class TimeDecayScheduler():
	def __init__(self, eta0=0.1, decay_factor=0.1):
		self.eta0 = eta0
		self.decay_factor = decay_factor
		
	def get_lr(self, epoch):
		eta = self.eta0 / (1 + self.decay_factor * epoch)
		return eta
		
class ExponentDecayScheduler():
	def __init__(self, eta0=0.1, decay_factor=0.1):
		self.eta0 = eta0
		self.decay_factor = decay_factor
		
	def get_lr(self, epoch):
		eta = self.eta0 * np.exp(-self.decay_factor * epoch)
		return eta
		
class InverseScalingScheduler():
	def __init__(self, eta0=0.1, power_t=0.01):
		self.eta0 = eta0
		self.power_t = power_t
		
	def get_lr(self, epoch):
		eta = self.eta0 / (1 + epoch) ** self.power_t
		return eta
		
class AdaptiveScheduler():
	def __init__(self, eta=0.1, decay_factor=0.1, threshold=1e-4, patience=5):
		self.eta = eta
		self.decay_factor = decay_factor
		self.threshold = threshold
		self.patience = patience
		self.counter = 0
		self.best_loss = float("inf")
		
	def get_lr(self, current_loss):
		if current_loss < (self.best_loss - self.threshold):
			self.best_loss = current_loss
			self.counter = 0
		else:
			self.counter += 1
		
		if self.counter >= self.patience:
			self.eta = self.eta * self.decay_factor
			self.counter = 0
		return self.eta
