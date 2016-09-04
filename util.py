import doctest
import numpy as np
import math


def add_ones(matrix, is_vector_column=False):
    """
    Insert a column filled with 1's at index 0 to the matrix.
    :param matrix: The matrix to add ones column to.
    :param is_vector_column: Are the vectors in matrix stored as column instead of row.
    :return The matrix after inserting ones column at index 0.

    >>> A = np.matrix(np.arange(5))
    >>> B = add_ones(A)
    >>> (B == np.matrix('1 0 1 2 3 4')).all()
    True
    >>> A = np.matrix(np.zeros(1))
    >>> B = add_ones(A)
    >>> (B == np.matrix('1 0')).all()
    True
    >>> A = np.matrix(np.zeros(0))
    >>> B = add_ones(A)
    >>> (B == np.matrix('1')).all()
    True
    >>> A = np.matrix(np.zeros(3)).T
    >>> B = add_ones(A, True)
    >>> (B == np.matrix('1 0 0 0').T).all()
    True
    """
    if matrix.ndim <= 1:
        return np.insert(matrix, 0, np.array(1))
    else:
        return np.insert(matrix, 0, np.array(1), axis=(0 if is_vector_column else 1))


def rand_Theta(num_features, num_classes, *S, EPSILON=1):
    """
    This function generates a random set of Thetas for neural network calculation.
    :param EPSILON: The EPSILON value used for range (-EPSILON < theta < EPSILON).
    :param num_features: Number of input features (without bias unit).
    :param S: An array containing the number of units in each hidden layer (without bias unit).
    :param num_classes: Number of classification classes.
    :return: A list containing numpy matrices containing randomly generated Thetas.

    >>> Theta = rand_Theta(3, 2, [3])
    >>> len(Theta)
    2
    >>> np.size(Theta[0], axis=0)
    3
    >>> np.size(Theta[0], axis=1)
    4
    >>> np.size(Theta[1], axis=0)
    1
    >>> np.size(Theta[1], axis=1)
    4
    >>> Theta = rand_Theta(400, 10, 600, 600, 600, EPSILON=2)
    >>> len(Theta)
    4
    >>> np.size(Theta[0], axis=0)
    600
    >>> np.size(Theta[0], axis=1)
    401
    >>> np.size(Theta[1], axis=0)
    600
    >>> np.size(Theta[1], axis=1)
    601
    >>> np.size(Theta[2], axis=0)
    600
    >>> np.size(Theta[2], axis=1)
    601
    >>> np.size(Theta[3], axis=0)
    10
    >>> np.size(Theta[3], axis=1)
    601
    """
    if num_classes <= 2:
        num_classes = 1

    # Convert num_layer_units to np array in case it's a generator.
    S = np.array(S)
    S = S.flatten()
    result = list()
    theta_cur = np.matrix(np.random.rand(S[0] if len(S) else num_classes, num_features + 1))
    theta_cur = change_range(theta_cur, EPSILON)
    result.append(theta_cur)
    for i, num_unit_current_layer in enumerate(S):
        if i < len(S) - 1:
            num_unit_next_layer = S[i + 1]
        else:
            num_unit_next_layer = num_classes
        theta_cur = np.matrix(np.random.rand(num_unit_next_layer, num_unit_current_layer + 1))
        theta_cur = change_range(theta_cur, EPSILON)
        result.append(theta_cur)
    return result


def zero_Delta(num_features, num_classes, *S):
    """
    Generates a three dimensional list of Deltas with zero as their value, used for back propagation calculation.
    :param num_features: Number of features.
    :param num_classes: Number of classes.
    :param S: List of number of neurons (without bias units) in each layer.
    :return: Delta with zero values that matches Theta's dimension.
    >>> result = zero_Delta(3, 2, [3])
    >>> len(result)
    2
    >>> np.size(result[0], axis=0)
    3
    >>> np.size(result[0], axis=1)
    4
    >>> np.size(result[1], axis=0)
    1
    >>> np.size(result[1], axis=1)
    4
    >>> for ele in result:
    ...     (ele == 0).all()
    True
    True
    """
    return rand_Theta(num_features, num_classes, *S, EPSILON=0)


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
    >>> Theta = np.random.rand(0)
    >>> Theta_ranged = change_range(Theta)
    >>> np.size(Theta_ranged, axis=0)
    0
    >>> Theta = np.random.rand(30, 10)
    >>> Theta_ranged = change_range(Theta)
    >>> (Theta_ranged >= -1).all() and (Theta_ranged <= 1).all()
    True
    """
    return Theta * 2 * EPSILON - EPSILON


def sigmoid(z):
    """
    Calculate sigmoid function (1/(1 + e^(-z))) output with given z value.
    :param z: Z in sigmoid function.
    :return: Sigmoid function output.
    >>> z = 0
    >>> result = sigmoid(z)
    >>> result
    0.5
    >>> z = 10
    >>> result = sigmoid(z)
    >>> result > 0.5
    True
    >>> result < 1
    True
    >>> z = -10
    >>> result = sigmoid(z)
    >>> result < 0.5
    True
    >>> result > 0
    True
    """
    return np.divide(1, (1 + np.power(math.e, -z)))


class DataProcessor:
    @staticmethod
    def add_x0_column(A):
        return np.insert(A, obj=0, values=1, axis=1)

    @staticmethod
    def augmented_to_coefficient_and_b(A):
        return A[:, :-1], A[:, -1]

    @staticmethod
    def partition(A, atInd):
        return A[:atInd], A[atInd:]

    @staticmethod
    def get_unique_categories(output, case_sensitive=True):
        if not case_sensitive:
            output = [x.lower() if isinstance(x, str) else x for x in output]
        output = output.flatten('F')
        return np.unique(output)

    @staticmethod
    def get_unique_categories_and_binary_outputs(output, case_sensitive=True):
        unique_cat = DataProcessor.get_unique_categories(output, case_sensitive)

        if np.size(unique_cat) <= 2:
            outputs_b = np.zeros(np.size(output)).T
            mask = (output == unique_cat[0]).flatten('F')
            outputs_b[mask] = 1
        else:
            outputs_b = np.zeros((np.size(output), np.size(unique_cat)))
            mask = np.repeat(np.matrix(unique_cat), np.size(outputs_b, axis=0), axis=0)
            output_2d = np.repeat(np.matrix(output).T, np.size(unique_cat), axis=1)
            mask = mask == output_2d
            outputs_b[mask] = 1

        return unique_cat, outputs_b


class FeatureNormalizer:
    def __init__(self, data, data_has_x0_column=False):
        self.__data = data.astype(np.float64)
        self.__data_has_x0_column = data_has_x0_column
        self.__scalars = np.ones(np.size(data, axis=1))
        self.__calculate_scalars()

    def normalized_feature(self):
        return self.normalize_new_feature(self.__data, self.__data_has_x0_column)

    def normalize_new_feature(self, data, input_has_x0_column=False):
        if input_has_x0_column:
            avg = np.insert(self.__avg, obj=0, values=0)
            std = np.insert(self.__std, obj=0, values=1)
        else:
            avg = self.__avg
            std = self.__std
        return np.nan_to_num((data - avg) / std)

    def __calculate_scalars(self):
        if self.__data_has_x0_column:
            self.__avg = np.average(self.__data[:, 1:], axis=0)
            self.__std = np.std(self.__data[:, 1:], axis=0)
        else:
            self.__avg = np.average(self.__data, axis=0)
            self.__std = np.std(self.__data, axis=0)
        if self.__std is 0:
            self.__std = np.max(self.__data, axis=0) - np.min(self.__data, axis=0)
            if self.__std == 0:
                self.__std = 1

if __name__ == '__main__':
    doctest.testmod()
