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
# Learning rate
alpha = 0.3
# Regularization
lamb = 10
# Random Theta rang
EPSILON_INIT = 0

trainer = nn.NeuralNetwork(X, y, alpha, lamb, EPSILON_INIT, 25)
trainer.train(iter_limit=100, time_limit=0, grad_check=True, save_to_file=False)
print(trainer.predict(np.matrix(X[-10:-1])))
print(y[-10:-1])

trainer.plot_training_info()
trainer.visualize_Theta()
trainer.show_plot()
