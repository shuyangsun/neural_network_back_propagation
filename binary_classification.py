import numpy as np
import neural_network as nn


X = np.matrix(np.random.rand(100, 10))
y = np.where(np.random.rand(100, 1) < 0.5, 0, 1)

trainer = nn.NeuralNetwork(X, y, 0.1, 10, 0, 25)
trainer.train(iter_limit=100, time_limit=0, grad_check=True, save_to_file=False)
