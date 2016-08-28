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
    """
    return np.insert(matrix, 0, np.array(1), axis=1)


def rand_Thetas_from_network_arc(num_features, num_layer_units, num_classes, EPSILON=1):
    """
    This function generates a random set of Thetas for neural network calculation.
    :param num_features: Number of input features (without bias unit).
    :param num_layer_units: An array containing the number of units in each hidden layer (without bias unit).
    :param num_classes: Number of classification classes.
    :return: A 3 dimensional numpy array containing randomly generated Thetas.
    """
    result = list()
    result.append(np.matrix(np.random.rand(num_features + 1, num_layer_units[0])))
    for i, num_unit_current_layer in enumerate(num_layer_units):
        num_unit_next_layer = num_layer_units[i + 1] if i < len(num_layer_units) - 1 else num_classes
        if num_classes <= 2:
            num_unit_next_layer = 1
        result.append(np.matrix(np.random.rand(num_unit_current_layer + 1, num_unit_next_layer)))
    result = np.array(result)
    
    return result

if __name__ == '__main__':
    doctest.testmod()
