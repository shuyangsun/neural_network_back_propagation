import doctest
import numpy as np
import neural_network_back_propagation.util as util


def nn_forward_prop(x_i, Theta):
    """
    Get the all the values in the layer based on input values and Theta, using forward propagation algorithm.
    :param x_i: Input layer values.
    :param Theta: Three dimensional array containing all the Theta values.
    :return: Three dimensional list containing all values in the neural network.
    >>> x = np.matrix('1 2 3')
    >>> Theta = [np.matrix('1 1 1 1')]
    >>> result = nn_forward_prop(x, Theta)
    >>> (x == np.matrix('1 2 3')).all()
    True
    >>> len(result)
    2
    >>> result[0].size
    4
    >>> result[1].size
    1
    >>> np.sum(result[-1])
    7
    >>> x = np.matrix('1 2 3')
    >>> Theta = [np.matrix('1 1 1 1; 0 0 0 0; 2 2 2 2'), np.matrix('1 2 0 1; 1 0 0 0'), np.matrix('1 1 1')]
    >>> result = nn_forward_prop(x, Theta)
    >>> len(result)
    4
    >>> result[0].size
    4
    >>> result[1].size
    4
    >>> result[2].size
    3
    >>> result[3].size
    1
    >>> np.sum(result[-1])
    31
    >>> num_features = 400
    >>> num_classes = 10
    >>> x = np.matrix(np.random.rand(num_features))
    >>> Theta = util.rand_Theta(num_features, num_classes, 600, 600, 600, EPSILON=5)
    >>> result = nn_forward_prop(x, Theta)
    >>> np.size(result[-1])
    10
    """
    a_l = x_i
    a_l = util.add_ones(a_l)
    result = [a_l]
    for i, theta_l in enumerate(Theta):
        a_l = (theta_l @ a_l.T).T
        if i < len(Theta) - 1:
            a_l = util.add_ones(a_l)
        result.append(a_l)
    return result


def nn_J_Theta(h_theta_x, y, lamb=0, Theta=None):
    """
    Calculate the cost based on given hypothesis, output value, regularization parameter lambda and Theta.
    :param h_theta_x: Hypothesis matrix (probabilities),\
    should be a two dimensional matrix if number of class is greater than 2.
    :param y: Output values (1 or 0), shape should match hypothesis.
    :param lamb: Regularization parameter.
    :param Theta: List of weights for regularization.
    :return: The cost of current hypothesis.
    >>> import math
    >>> h_theta_x = np.matrix('0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5')
    >>> y = np.matrix('0 1 0 0; 1 0 0 0; 0 0 0 1')
    >>> cost = nn_J_Theta(h_theta_x, y)
    >>> cost == - math.log(0.5) * 4
    True
    >>> h_theta_x = np.matrix('0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5')
    >>> y = np.matrix('0 1 0 0; 1 0 0 0; 0 0 0 1')
    >>> Theta = [np.matrix('1 1 1')]
    >>> lamb = 2
    >>> cost = nn_J_Theta(h_theta_x, y, lamb, Theta)
    >>> cost == - math.log(0.5) * 4 + 1
    True
    >>> h_theta_x = np.matrix('0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5')
    >>> y = np.matrix('0 1 0 0; 1 0 0 0; 0 0 0 1')
    >>> Theta = [np.matrix('1 1 1'), np.matrix('1 1 1; 1 1 1; 1 1 1')]
    >>> lamb = 4
    >>> cost = nn_J_Theta(h_theta_x, y, lamb, Theta)
    >>> cost == - math.log(0.5) * 4 + 8
    True
    """
    m = np.size(h_theta_x, axis=0)
    cost_y_is_0 = np.multiply((1 - y), np.log(1 - h_theta_x))
    cost_y_is_1 = np.multiply(y, np.log(h_theta_x))
    cost_wo_regularization = - np.sum(cost_y_is_0 + cost_y_is_1) / m
    if Theta is None or len(Theta) == 0:
        theta_squared_sum = 0
    else:
        theta_squared_sums = [np.sum(np.power(theta_i, 2)) for theta_i in Theta]
        theta_squared_sum = sum(theta_squared_sums)
    return cost_wo_regularization + lamb / (2 * m) * theta_squared_sum


def nn_delta(neurons, Theta, y):
    """
    Calculates errors using back propagation algorithm.
    :param neurons: Two dimensional list of neurons, including bias units.
    :param Theta: Three dimensional list of Theta.
    :param y: Vector matrix of output units.
    :return: Three dimensional list of Delta.
    >>> input_layer_count = 10
    >>> hidden_layer_counts = [20, 20, 20]
    >>> output_layer_count = 5
    >>> x_i = np.matrix(np.random.rand(input_layer_count))
    >>> Theta = util.rand_Theta(input_layer_count, output_layer_count, hidden_layer_counts)
    >>> neurons = nn_forward_prop(x_i, Theta)
    >>> y = np.matrix(np.zeros(output_layer_count))
    >>> y[0] = 1
    >>> result = nn_delta(neurons, Theta, y)
    >>> len(result) == len(hidden_layer_counts) + 1
    True
    >>> all([delt.size == layer_count for delt, layer_count in zip(result, hidden_layer_counts)])
    True
    >>> result[-1].size == output_layer_count
    True
    >>> input_layer_count = 400
    >>> hidden_layer_counts = [400, 400]
    >>> output_layer_count = 400
    >>> x_i = np.matrix(np.random.rand(input_layer_count))
    >>> Theta = util.rand_Theta(input_layer_count, output_layer_count, hidden_layer_counts)
    >>> neurons = nn_forward_prop(x_i, Theta)
    >>> y = np.matrix(np.zeros(output_layer_count))
    >>> y[0] = 1
    >>> result = nn_delta(neurons, Theta, y)
    >>> len(result) == len(hidden_layer_counts) + 1
    True
    >>> all([delt.size == layer_count for delt, layer_count in zip(result, hidden_layer_counts)])
    True
    >>> result[-1].size == output_layer_count
    True
    >>> input_layer_count = 400
    >>> hidden_layer_counts = []
    >>> output_layer_count = 2
    >>> x_i = np.matrix(np.random.rand(input_layer_count))
    >>> Theta = util.rand_Theta(input_layer_count, output_layer_count, hidden_layer_counts)
    >>> neurons = nn_forward_prop(x_i, Theta)
    >>> y = np.matrix(np.zeros(output_layer_count))
    >>> y[0] = 1
    >>> result = nn_delta(neurons, Theta, y)
    >>> len(result) == len(hidden_layer_counts) + 1
    True
    >>> all([delta.size == layer_count for delta, layer_count in zip(result, hidden_layer_counts)])
    True
    >>> result[-1].size == output_layer_count
    True
    """
    assert len(Theta) == len(neurons) - 1,\
        'Layers of neural neurons count ({0}) does not fit with layers of Theta count ({1}).'.format(len(neurons),
                                                                                                     len(Theta))
    neurons_rev = list(reversed(neurons))
    Theta_rev = list(reversed(Theta))
    result = [neurons_rev[0] - y]
    for l, theta_l in enumerate(Theta_rev):
        if l < len(Theta_rev) - 1:
            delta_next_layer = result[-1]
            a_l = neurons_rev[l + 1]
            delta_cur_layer = np.multiply(delta_next_layer @ theta_l, np.multiply(a_l, (1 - a_l)))
            delta_cur_layer = np.delete(delta_cur_layer, 0, 1)
            result.append(delta_cur_layer)
    return list(reversed(result))


def nn_Delta(Delta, delta, neurons):
    """
    Compute Delta for further computing of derivative of J(Theta) - D.
    :param Delta: Original Delta from last iteration.
    :param delta: Error rates computed with back propagation algorithm.
    :param neurons: Two dimensional list of neurons, including bias units.
    :return: List of Delta, shape should match Theta.
    >>> Delta = util.zero_Delta(10, 5, 10, 10)
    >>> delta = [np.matrix(np.ones(10)), np.matrix(np.ones(10)), np.matrix(np.ones(10)), np.matrix(np.ones(5))]
    >>> neurons = [np.matrix(np.ones(11)), np.matrix(np.ones(11)), np.matrix(np.ones(11)), np.matrix(np.ones(5))]
    >>> result = nn_Delta(Delta, delta, neurons)
    >>> len(result)
    3
    >>> np.size(result[0], axis=0)
    10
    >>> np.size(result[0], axis=1)
    11
    >>> np.size(result[1], axis=0)
    10
    >>> np.size(result[1], axis=1)
    11
    >>> np.size(result[2], axis=0)
    5
    >>> np.size(result[2], axis=1)
    11
    >>> for ele in result:
    ...     (ele == 1).all()
    True
    True
    True
    >>> Delta = util.zero_Delta(5, 2)
    >>> delta = [np.matrix(np.ones(5)), np.matrix(np.ones(1))]
    >>> neurons = [np.matrix(np.ones(6)), np.matrix(np.ones(1))]
    >>> result = nn_Delta(Delta, delta, neurons)
    >>> len(result)
    1
    >>> np.size(result[0], axis=0)
    1
    >>> np.size(result[0], axis=1)
    6
    >>> for ele in result:
    ...     (ele == 1).all()
    True
    """
    result = list()
    for (l, ele) in enumerate(Delta):
        a_l = neurons[l]
        Delta_l = Delta[l] + (a_l.T @ delta[l + 1]).T
        result.append(Delta_l)
    return result


def nn_D(m, Delta, Theta, lamb):
    """
    Calculates D (derivative of cost function) for J(Theta).
    :param m: Number of training samples.
    :param Delta: Three dimensional list of Delta computed using back propagation algorithm.
    :param Theta: Three dimensional list of weights.
    :param lamb: Regularization parameter lambda.
    :return: Three dimensional list of derivative of J(Theta).
    >>> m = 5
    >>> Delta = [np.matrix('5 5 5 5')]
    >>> Theta = [np.matrix('1 1 1 1')]
    >>> lamb = 5
    >>> result = nn_D(m, Delta, Theta, lamb)
    >>> len(result)
    1
    >>> D = result[0]
    >>> D.size
    4
    >>> D[0, 0]
    1.0
    >>> (D[0, 1:0] == 6).all()
    True
    """
    result = list()
    for (l, theta_l) in enumerate(Theta):
        lambda_l = theta_l * 0 + lamb
        lambda_l[:, 0] = 0
        D_l = 1 / m * Delta[l] + np.multiply(lambda_l, theta_l)
        result.append(D_l)
    return result

def nn_grad_check(h_theta_x, y, D, Theta, lamb=0, EPSILON=0.1):
    Theta_plus = Theta.copy()
    Theta_minus = Theta.copy()
    for l in len(Theta):
        for i in np.size(Theta[l], axis=0):
            for j in np.size(Theta[l], axis=1):
                theta_l_i_j_original = Theta[l][i, j]
                Theta_plus[l][i, j] += EPSILON
                Theta_minus[l][i, j] -= EPSILON
                J_Theta_plus = nn_J_Theta(h_theta_x, y, lamb, Theta_plus)
                J_Theta_minus = nn_J_Theta(h_theta_x, y, lamb, Theta_minus)
                grad_approx = (J_Theta_plus - J_Theta_minus) / (2 * EPSILON)
                if grad_approx is not D[l][i, j]:
                    return False
                else:
                    Theta_plus[l][i, j] = theta_l_i_j_original
                    Theta_minus[l][i, j] = theta_l_i_j_original
    return True,

if __name__ == '__main__':
    doctest.testmod()
