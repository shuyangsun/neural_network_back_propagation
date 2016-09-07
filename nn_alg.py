import doctest
import numpy as np
import numexpr as ne
import util
import math


def nn_forward_prop(X, Theta):
    """
    Get the all the values in the layer based on X and θ, using forward propagation algorithm.
    :param X: Input layer values X.
    :param Theta: Three dimensional list of θ.
    :return: Three dimensional list containing all neurons in the neural network.
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
    >>> np.sum(result[-1]) > 0.5
    True
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
    >>> num_features = 400
    >>> num_classes = 10
    >>> x = np.matrix(np.random.rand(num_features))
    >>> Theta = util.rand_Theta(num_features, num_classes, 600, 600, 600, EPSILON_INIT=5)
    >>> result = nn_forward_prop(x, Theta)
    >>> np.size(result[-1])
    10
    """
    a_l = X
    a_l = util.add_ones(a_l)
    result = [a_l]
    for i, theta_l in enumerate(Theta):
        a_l = (theta_l @ a_l.T).T
        if i < len(Theta) - 1:
            a_l = util.add_ones(a_l)
        result.append(util.sigmoid(a_l))
    return result


def nn_J_Theta(h_theta_x, y, lamb=0, Theta=None):
    """
    Calculate the cost based on given hypothesis hθ(x), output value y, regularization parameter λ and θ.
    :param h_theta_x: Hypothesis matrix (probabilities),\
    should be a two dimensional matrix if number of class K is greater than 2.
    :param y: Output values (1 or 0), shape should match hypothesis.
    :param lamb: Regularization parameter λ.
    :param Theta: Three dimensional list of θ for regularization.
    :return: The cost of current hypothesis J(θ).
    >>> import math
    >>> h_theta_x = np.matrix('0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5')
    >>> y = np.matrix('0 1 0 0; 1 0 0 0; 0 0 0 1')
    >>> cost = nn_J_Theta(h_theta_x, y)
    >>> cost == - math.log(0.5) * 4
    True
    >>> h_theta_x = np.matrix('0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5')
    >>> y = np.matrix('0 1 0 0; 1 0 0 0; 0 0 0 1')
    >>> Theta = [np.matrix('1 1 1 1')]
    >>> lamb = 2
    >>> cost = nn_J_Theta(h_theta_x, y, lamb, Theta)
    >>> cost == - math.log(0.5) * 4 + 1
    True
    >>> h_theta_x = np.matrix('0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5; 0.5 0.5 0.5 0.5')
    >>> y = np.matrix('0 1 0 0; 1 0 0 0; 0 0 0 1')
    >>> Theta = [np.matrix('1 1 1 1'), np.matrix('1 1 1 1; 1 1 1 1; 1 1 1 1')]
    >>> lamb = 4
    >>> cost = nn_J_Theta(h_theta_x, y, lamb, Theta)
    >>> cost == - math.log(0.5) * 4 + 8
    True
    """
    m = np.size(h_theta_x, axis=0)
    cost_y_is_0 = np.nan_to_num(ne.evaluate('(1 - y) * log(1 - h_theta_x)'))
    cost_y_is_1 = np.nan_to_num(ne.evaluate('y * log(h_theta_x)'))
    cost_wo_regularization = -ne.evaluate('sum(cost_y_is_0 + cost_y_is_1)') / m
    if Theta is None or len(Theta) == 0:
        theta_squared_sum = 0
    else:
        theta_squared_sums = []
        for theta_l in Theta:
            theta_l_wo_0_column = theta_l[:, 1:]
            theta_squared_sums.append(ne.evaluate('sum(theta_l_wo_0_column ** 2)'))
        # theta_squared_sums = [np.sum(np.power(theta_i[:, 1:], 2)) for theta_i in Theta]
        theta_squared_sum = sum(theta_squared_sums)
    return cost_wo_regularization + lamb / (2 * m) * theta_squared_sum


def nn_delta(neurons, Theta, y):
    """
    Calculates errors using back propagation algorithm.
    :param delta: Original delta.
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
    >>> delta = util.zero_Delta(input_layer_count, output_layer_count, hidden_layer_counts)
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
            # delta_cur_layer = np.multiply(delta_next_layer @ theta_l, np.multiply(a_l, (1 - a_l)))
            delta_bp = delta_next_layer @ theta_l
            delta_cur_layer = np.matrix(ne.evaluate('delta_bp * (a_l * (1 - a_l))'))
            delta_cur_layer = np.delete(delta_cur_layer, 0, 1)
            result.append(delta_cur_layer)
    return list(reversed(result))


def nn_Delta(Delta, delta, neurons):
    """
    Compute Delta for further computing of D (derivative of J(θ)).
    :param Delta: Original Delta from last iteration.
    :param delta: Error rates computed with back propagation algorithm.
    :param neurons: Two dimensional list of neurons, including bias units.
    :return: List of Delta, shape should match Theta.
    >>> Delta = util.zero_Delta(10, 5, 10, 10)
    >>> delta = [np.matrix(np.ones(10)), np.matrix(np.ones(10)), np.matrix(np.ones(5))]
    >>> neurons = [np.matrix(np.ones(11)), np.matrix(np.ones(11)), np.matrix(np.ones(11)), np.matrix(np.ones(5))]
    >>> result = nn_Delta(Delta, delta, neurons)
    >>> len(result)
    3
    >>> np.size(result[0], axis=0)
    1
    >>> np.size(result[0], axis=1)
    10
    >>> np.size(result[0], axis=2)
    11
    >>> np.size(result[1], axis=0)
    1
    >>> np.size(result[1], axis=1)
    10
    >>> np.size(result[1], axis=2)
    11
    >>> np.size(result[2], axis=0)
    1
    >>> np.size(result[2], axis=1)
    5
    >>> np.size(result[2], axis=2)
    11
    >>> for ele in result:
    ...     (ele[0] == 1).all()
    True
    True
    True
    >>> Delta = util.zero_Delta(5, 2)
    >>> delta = [np.matrix(np.ones(1))]
    >>> neurons = [np.matrix(np.ones(6)), np.matrix(np.ones(1))]
    >>> result = nn_Delta(Delta, delta, neurons)
    >>> len(result)
    1
    >>> np.size(result[0], axis=0)
    1
    >>> np.size(result[0], axis=1)
    1
    >>> np.size(result[0], axis=2)
    6
    >>> for ele in result:
    ...     (ele[0] == 1).all()
    True
    """
    result = list()
    for (l, Delta_l) in enumerate(Delta):
        Delta_l = np.copy(Delta_l)
        for (m, Delta_l_m) in enumerate(Delta_l):
            a_l_m = np.matrix(neurons[l][m])
            # delta ^ (l) here is actually delta ^ (l + 1), because there's no delta ^ (1) on input layer.
            Delta_l[m] = Delta_l_m + delta[l][m].T @ a_l_m
        result.append(Delta_l)
    return result


def nn_D(m, Delta, Theta, lamb):
    """
    Calculates D (derivative of cost function) for J(θ).
    :param m: Number of training samples.
    :param Delta: Three dimensional list of Delta computed using back propagation algorithm.
    :param Theta: Three dimensional list of weights.
    :param lamb: Regularization parameter lambda.
    :return: Three dimensional list of derivative of J(θ).
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
    >>> m = 5
    >>> Delta = [np.matrix('5 5 5 5'), np.matrix('5 5 5 5 5 5')]
    >>> Theta = [np.matrix('1 1 1 1'), np.matrix('1 1 1 1 1 1')]
    >>> lamb = 99
    >>> result = nn_D(m, Delta, Theta, lamb)
    >>> len(result)
    2
    >>> D_1 = result[0]
    >>> D_1.size
    4
    >>> D_2 = result[1]
    >>> D_2.size
    6
    >>> D_1[0, 0]
    1.0
    >>> D_2[0, 0]
    1.0
    >>> (D_1[0, 1:0] == 100).all()
    True
    >>> (D_2[0, 1:0] == 100).all()
    True
    """
    Delta_sum = list()
    for Delta_l in Delta:
        Delta_sum.append(np.sum(Delta_l, axis=0))
    result = list()
    for (l, theta_l) in enumerate(Theta):
        lambda_l = theta_l * 0 + lamb
        lambda_l[:, 0] = 0
        D_l = 1 / m * (Delta_sum[l] + np.multiply(lambda_l, theta_l))
        result.append(D_l)
    return result


def nn_update_Theta_with_D(Theta, D, alpha=0.01):
    """
    Update θ using calculated derivative of θ.
    :param Theta: Three dimensional list of θ.
    :param D: Three dimensional list of derivative of θ.
    :param alpha: Learning rate.
    :return: Updated θ.
    >>> Theta = [np.matrix('1 2 3 4 5')]
    >>> D = [np.matrix('0 1 2 3 4')]
    >>> result = nn_update_Theta_with_D(Theta, D, alpha=1)
    >>> len(result)
    1
    >>> (result[0] == 1).all()
    True
    """
    result = list()
    for (l, theta_l) in enumerate(Theta):
        result.append((theta_l - alpha * D[l]).astype(np.float32))
    return result


def nn_grad_check(X, y, D, Theta, lamb=0, EPSILON=0.0001):
    """
    Check if the gradient approximation is the same as D (derivative of J(θ)).
    :param X: Training set inputs.
    :param y: Outputs.
    :param D: Derivative of J(θ).
    :param Theta: Three dimensional list of θ.
    :param lamb: Regularization parameter λ.
    :param EPSILON: Left and right side ε value for gradient checking.
    :return: True if gradient check passes, false otherwise.
    >>> X = np.matrix('1 2 3 5; 2 3 6 7; 5 2 11 -9')
    >>> y = np.matrix('1 0 0; 0 1 0 ; 0 0 1')
    >>> Theta = util.rand_Theta(4, 3, 10, 10, EPSILON_INIT=100)
    >>> lamb = 10
    >>> D = util.zero_Delta(4, 3, 10, 10)
    >>> result = nn_grad_check(X, y, D, Theta, lamb)
    >>> result
    False
    """
    for l in range(len(Theta)):
        for i in range(np.size(Theta[l], axis=0)):
            for j in range(np.size(Theta[l], axis=1)):
                Theta_plus = util.copy_list_of_ndarray(Theta)
                Theta_minus = util.copy_list_of_ndarray(Theta)
                perturb = np.matrix(np.zeros((np.size(Theta[l], axis=0), np.size(Theta[l], axis=1))))
                perturb[i, j] = EPSILON
                Theta_plus[l] += perturb
                Theta_minus[l] -= perturb
                h_theta_x_plus = nn_forward_prop(X, Theta_plus)[-1]
                h_theta_x_minus = nn_forward_prop(X, Theta_minus)[-1]
                J_Theta_plus = nn_J_Theta(h_theta_x_plus, y, lamb, Theta_plus)
                J_Theta_minus = nn_J_Theta(h_theta_x_minus, y, lamb, Theta_minus)
                grad_approx = (J_Theta_plus - J_Theta_minus) / (2 * EPSILON)
                D_l_i_j = D[l][i, j]
                if np.isnan(grad_approx) or math.fabs(grad_approx - D_l_i_j) > 10 ** -5:
                    return False
    return True

if __name__ == '__main__':
    doctest.testmod()
