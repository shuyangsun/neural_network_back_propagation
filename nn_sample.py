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
test_ratio = 0.98
alpha = 0.001
lamb = 10
trainer = nn.NeuralNetwork(X, y, test_ratio, alpha, lamb, 400)
trainer.train(iter_limit=0, time_limit=60, grad_check=True)
print(trainer.predict(np.matrix(X[-10:-1])))
print(y[-10:-1])
