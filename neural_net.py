import doctest
import numpy as np
import util


def forward_prop(x_i, Theta):
    """
    Get the output layer values based on input layer and hidden layer thetas, using forward propagation algorithm.
    :param x_i: Input layer values.
    :param Theta: Three dimensional array containing all the Theta values.
    :return: Output layer values.
    >>> x = np.matrix('1 2 3').T
    >>> Theta = [np.matrix('1 1 1 1').T]
    >>> np.sum(forward_prop(x, Theta))
    7
    >>> x = np.matrix('1 2 3').T
    >>> Theta = [np.matrix('1 1 1 1; 0 0 0 0; 2 2 2 2').T, np.matrix('1 2 0 1; 1 0 0 0').T, np.matrix('1 1 1').T]
    >>> np.sum(forward_prop(x, Theta))
    31
    """
    a_i = x_i
    for theta_i in Theta:
        a_i = util.add_ones(a_i)
        a_i = theta_i.T @ a_i
    return a_i

if __name__ == '__main__':
    doctest.testmod()
