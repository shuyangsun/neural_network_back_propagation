import doctest
import scipy.io
import numpy as np
import nn_alg as alg
import util


def cost_of_data():
    """
    Get the cost of given data and weights.
    :return: The cost of given data.
    >>> '{0:.6f}'.format(cost_of_data())
    '0.287629'
    """
    mat_data = scipy.io.loadmat('digits_data/hand_written_digits.mat')
    X = mat_data['X']
    y = mat_data['y']
    augmented_matrix = np.append(X, y, axis=1)
    np.random.shuffle(augmented_matrix)
    X = augmented_matrix[:, :-1]
    y = augmented_matrix[:, -1]
    _, y = util.DataProcessor.get_unique_categories_and_binary_outputs(y)

    mat_weights = scipy.io.loadmat('cost_test_data/ex4weights.mat')
    Theta1 = mat_weights['Theta1']
    Theta2 = mat_weights['Theta2']

    Theta = [Theta1, Theta2]
    h_theta_x = alg.nn_forward_prop(X, Theta)[-1]

    cost = alg.nn_J_Theta(h_theta_x, y, lamb=0, Theta=Theta)
    return cost


if __name__ == '__main__':
    doctest.testmod()

