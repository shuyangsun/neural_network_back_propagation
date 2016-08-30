import doctest
import numpy as np
import neural_network_back_propagation.util as util


def forward_prop(x_i, Theta):
    """
    Get the all the values in the layer based on input values and Theta, using forward propagation algorithm.
    :param x_i: Input layer values.
    :param Theta: Three dimensional array containing all the Theta values.
    :return: Three dimensional list containing all values in the neural network.
    >>> x = np.matrix('1 2 3').T
    >>> Theta = [np.matrix('1 1 1 1').T]
    >>> result = forward_prop(x, Theta)
    >>> len(result)
    2
    >>> result[0].size
    4
    >>> result[1].size
    1
    >>> np.sum(result[-1])
    7
    >>> x = np.matrix('1 2 3').T
    >>> Theta = [np.matrix('1 1 1 1; 0 0 0 0; 2 2 2 2').T, np.matrix('1 2 0 1; 1 0 0 0').T, np.matrix('1 1 1').T]
    >>> result = forward_prop(x, Theta)
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
    >>> x = np.matrix(np.random.rand(num_features)).T
    >>> Theta = util.rand_Theta(num_features, num_classes, 600, 600, 600, EPSILON=5)
    >>> result = forward_prop(x, Theta)
    >>> np.size(result[-1])
    10
    """
    a_i = x_i
    a_i = util.add_ones(a_i)
    result = [a_i]
    for i, theta_i in enumerate(Theta):
        a_i = theta_i.T @ a_i
        if i < len(Theta) - 1:
            a_i = util.add_ones(a_i)
        result.append(a_i)
    return result


def Delta(neurons, Theta, y):
    """
    Calculates errors using back propagation algorithm.
    :param neurons: Two dimensional list of neurons, including bias units.
    :param Theta: Three dimensional list of Theta.
    :param y: Vector matrix of output units.
    :return: Three dimensional list of Delta.
    >>> input_layer_count = 10
    >>> hidden_layer_counts = [20, 20, 20]
    >>> output_layer_count = 5
    >>> x_i = np.matrix(np.random.rand(input_layer_count)).T
    >>> Theta = util.rand_Theta(input_layer_count, output_layer_count, hidden_layer_counts)
    >>> neurons = forward_prop(x_i, Theta)
    >>> y = np.matrix(np.zeros(output_layer_count)).T
    >>> y[0] = 1
    >>> result = Delta(neurons, Theta, y)
    >>> len(result) == len(hidden_layer_counts) + 1
    True
    >>> all([delt.size == layer_count for delt, layer_count in zip(result, hidden_layer_counts)])
    True
    >>> len(result[-1]) == output_layer_count
    True
    >>> input_layer_count = 400
    >>> hidden_layer_counts = [400, 400]
    >>> output_layer_count = 400
    >>> x_i = np.matrix(np.random.rand(input_layer_count)).T
    >>> Theta = util.rand_Theta(input_layer_count, output_layer_count, hidden_layer_counts)
    >>> neurons = forward_prop(x_i, Theta)
    >>> y = np.matrix(np.zeros(output_layer_count)).T
    >>> y[0] = 1
    >>> result = Delta(neurons, Theta, y)
    >>> len(result) == len(hidden_layer_counts) + 1
    True
    >>> all([delt.size == layer_count for delt, layer_count in zip(result, hidden_layer_counts)])
    True
    >>> len(result[-1]) == output_layer_count
    True
    >>> input_layer_count = 400
    >>> hidden_layer_counts = []
    >>> output_layer_count = 2
    >>> x_i = np.matrix(np.random.rand(input_layer_count)).T
    >>> Theta = util.rand_Theta(input_layer_count, output_layer_count, hidden_layer_counts)
    >>> neurons = forward_prop(x_i, Theta)
    >>> y = np.matrix(np.zeros(output_layer_count)).T
    >>> y[0] = 1
    >>> result = Delta(neurons, Theta, y)
    >>> len(result) == len(hidden_layer_counts) + 1
    True
    >>> all([delta.size == layer_count for delta, layer_count in zip(result, hidden_layer_counts)])
    True
    >>> len(result[-1]) == output_layer_count
    True
    """
    assert len(Theta) == len(neurons) - 1,\
        'Layers of neural neurons count ({0}) does not fit with layers of Theta count ({1}).'.format(len(neurons),
                                                                                                     len(Theta))
    neurons_rev = list(reversed(neurons))
    Theta_rev = list(reversed(Theta))
    result = [neurons_rev[0] - y]
    for i, theta_i in enumerate(Theta_rev):
        if i < len(Theta_rev) - 1:
            delta_next_layer = result[-1]
            a_i = neurons_rev[i + 1]
            delta_cur_layer = np.multiply(theta_i @ delta_next_layer, np.multiply(a_i, (1 - a_i)))
            delta_cur_layer = np.delete(delta_cur_layer, 0, 0)
            result.append(delta_cur_layer)
    return list(reversed(result))


if __name__ == '__main__':
    doctest.testmod()
