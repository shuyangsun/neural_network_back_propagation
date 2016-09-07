import neural_network as nn
import scipy.io
from scipy.interpolate import spline
import numpy as np
import matplotlib.pyplot as plt
import pylab as p

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
alpha = 0.03
# Regularization
lamb = 1
# Random Theta range
EPSILON_INIT = 0

trainer = nn.NeuralNetwork(X, y, test_ratio, alpha, lamb, EPSILON_INIT, 25)
cost_list, accuracy_list, Theta = trainer.train(iter_limit=1000,
                                                time_limit=0,
                                                grad_check=True,
                                                save_to_file=False)
print(trainer.predict(np.matrix(X[-10:-1])))
print(y[-10:-1])

plt.close('all')

# Cost plot
figure = plt.subplot(1, 2, 1)
plt.title('Cost')
plt.xlabel('iteration')
plt.ylabel('cost')
figure.plot(cost_list)

# Accuracy plot
figure = plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('iteration')
plt.ylabel('accuracy rate (%)')

x = list(range(len(accuracy_list)))
x = [num * 10 for num in x]
x_new = np.linspace(x[0], x[-1], len(x) * 3)
y_new = spline(x, accuracy_list, x_new)

figure.plot(x_new, y_new, color='blue')
p.fill_between(x_new, y_new, facecolor='blue', alpha=0.25)

plt.show()
