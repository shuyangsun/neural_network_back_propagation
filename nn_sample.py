import neural_network as nn
import scipy.io
import numpy as np

mat_data = scipy.io.loadmat('digits_data/hand_written_digits.mat')
X = mat_data['X']
y = mat_data['y']
augmented_matrix = np.append(X, y, axis=1)
np.random.shuffle(augmented_matrix)
X = augmented_matrix[:, :-1]
y = augmented_matrix[:, -1]
test_sample_size = 100
test_ratio = test_sample_size / np.size(X, axis=0)
# Learning rate
alpha = 0.15
# Regularization
lamb = 1
# Random Theta range
EPSILON_INIT = 0
trainer = nn.NeuralNetwork(X, y, test_ratio, alpha, lamb, EPSILON_INIT, 25)
trainer.train(iter_limit=0, time_limit=3600 * 2, grad_check=True)
print(trainer.predict(np.matrix(X[-10:-1])))
print(y[-10:-1])
