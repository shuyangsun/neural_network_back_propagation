import neural_network as nn
import scipy.io
import numpy as np

mat_data = scipy.io.loadmat('digits_data/hand_written_digits.mat')
X = np.matrix(mat_data['X'])
y = np.matrix(mat_data['y'])
test_ratio = 0.95
lamb = 0
trainer = nn.NeuralNetwork(X, y, test_ratio, lamb, 10, 10)
trainer.train(iter_limit=1000)
trainer.predict(X[-1])
