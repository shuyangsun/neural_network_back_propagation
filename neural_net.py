import doctest
import numpy as np


def add_ones_column(matrix):
    """
    Insert a column filled with 1's at index 0 to the matrix.
    :param matrix: The matrix to add ones column to.
    :return The matrix after inserting ones column at index 0.

    >>> A = np.matrix(np.arange(5))
    >>> B = add_ones_column(A)
    >>> (B == np.matrix([1, 0, 1, 2, 3, 4])).all()
    True
    >>> A = np.matrix(np.zeros(1))
    >>> B = add_ones_column(A)
    >>> (B == np.matrix([1, 0])).all()
    True
    >>> A = np.matrix(np.zeros(0))
    >>> B = add_ones_column(A)
    >>> (B == np.matrix([1])).all()
    True
    """
    return np.insert(matrix, 0, np.array(1), axis=1)


def rand_Theta_from_neural_network_arc(num_features, num_layer_units, num_classes, EPSILON=1):
    """
    This function generates a random set of Thetas for neural network calculation.
    :param EPSILON: The EPSILON value used for range (-EPSILON < theta < EPSILON).
    :param num_features: Number of input features (without bias unit).
    :param num_layer_units: An array containing the number of units in each hidden layer (without bias unit).
    :param num_classes: Number of classification classes.
    :return: A list containing numpy matrices containing randomly generated Thetas.

    >>> Theta = rand_Theta_from_neural_network_arc(3, [3], 2)
    >>> len(Theta)
    2
    >>> np.size(Theta[0], axis=0)
    4
    >>> np.size(Theta[0], axis=1)
    3
    >>> np.size(Theta[1], axis=0)
    4
    >>> np.size(Theta[1], axis=1)
    1
    >>> Theta = rand_Theta_from_neural_network_arc(400, [600, 600, 600], 10)
    >>> len(Theta)
    4
    >>> np.size(Theta[0], axis=0)
    401
    >>> np.size(Theta[0], axis=1)
    600
    >>> np.size(Theta[1], axis=0)
    601
    >>> np.size(Theta[1], axis=1)
    600
    >>> np.size(Theta[2], axis=0)
    601
    >>> np.size(Theta[2], axis=1)
    600
    >>> np.size(Theta[3], axis=0)
    601
    >>> np.size(Theta[3], axis=1)
    10
    """
    # Convert num_layer_units to np array in case it's a generator.
    num_layer_units = np.array(num_layer_units)
    result = list()
    theta_cur = np.matrix(np.random.rand(num_features + 1, num_layer_units[0]))
    theta_cur = change_range(theta_cur, EPSILON)
    result.append(theta_cur)
    for i, num_unit_current_layer in enumerate(num_layer_units):
        if i < len(num_layer_units) - 1:
            num_unit_next_layer = num_layer_units[i + 1]
        else:
            num_unit_next_layer = num_classes if num_classes > 2 else 1
        theta_cur = np.matrix(np.random.rand(num_unit_current_layer + 1, num_unit_next_layer))
        change_range(theta_cur, EPSILON)
        result.append(theta_cur)
    return result


def change_range(Theta, EPSILON=1):
    """
    Make thetas be inside range [-EPSILON, EPSILON], assuming original values are inside range [0, 1].
    :param Theta: Numpy array containing thetas, all thetas must be in range [0, 1].
    :param EPSILON: Absolute range on each side.
    :return: Numpy array with range [-EPSILON, EPSILON].
    >>> Theta = np.random.rand(10)
    >>> Theta_ranged = change_range(Theta)
    >>> (Theta_ranged >= -1).all() and (Theta_ranged <= 1).all()
    True
    >>> Theta_ranged = change_range(Theta, 2)
    >>> (Theta_ranged >= -2).all() and (Theta_ranged <= 2).all()
    True
    """
    return Theta * 2 * EPSILON - EPSILON


if __name__ == '__main__':
    doctest.testmod()
