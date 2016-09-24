import doctest
import numpy as np
import random
import time
import os


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


def rand_Theta(num_features, num_classes, *S, EPSILON_INIT=0):
    """
    This function generates a random set of Thetas for neural network calculation.
    :param EPSILON_INIT: The EPSILON value used for range (-EPSILON < theta < EPSILON).
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
    >>> Theta = rand_Theta(400, 10, 600, 600, 600, EPSILON_INIT=2)
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
    result = list()
    optimize_eps = EPSILON_INIT is 0
    shapes = Theta_or_Delta_shapes(num_features, num_classes, *S, m=0)
    for S_l1, S_l in shapes:
        theta_cur = np.matrix(np.random.rand(S_l1, S_l))
        if optimize_eps:
            EPSILON_INIT = optimize_EPSILON_INIT(S_l, S_l1)
        theta_cur = change_range(theta_cur, EPSILON_INIT)
        result.append(theta_cur)
    return result


def zero_Delta(num_features, num_classes, *S, m=1, dtype=np.float32):
    """
    Generates a three dimensional list of Deltas with zero as their value, used for back propagation calculation.
    :param num_features: Number of features.
    :param num_classes: Number of classes.
    :param S: List of number of neurons (without bias units) in each layer.
    :param m: Number of training samples.
    :param dtype: Data type.
    :return: Delta with zero values that matches Theta's dimension.
    >>> result = zero_Delta(3, 2, [3])
    >>> len(result)
    2
    >>> np.size(result[0], axis=0)
    1
    >>> np.size(result[0], axis=1)
    3
    >>> np.size(result[0], axis=2)
    4
    >>> np.size(result[1], axis=0)
    1
    >>> np.size(result[1], axis=1)
    1
    >>> np.size(result[1], axis=2)
    4
    >>> for ele in result:
    ...     (ele == 0).all()
    True
    True
    """
    result = list()
    shapes = Theta_or_Delta_shapes(num_features, num_classes, *S, m=m)
    for m, S_l1, S_l in shapes:
        result.append(np.zeros((m, S_l1, S_l), dtype=dtype))
    return result


def Theta_or_Delta_shapes(num_features, num_classes, *S, m=0):
    """
    Return a list of shapes of Theta.
    :param num_features: Number of features.
    :param num_classes: Number of output classes.
    :param S: List of number of units in hidden layers.
    :param m: Number of training samples (for vectorized Delta only).
    :return: The shape of Theta or Delta with given parameter.
    >>> num_features = 10
    >>> S = [10, 10]
    >>> num_classes = 5
    >>> m = 0
    >>> result = Theta_or_Delta_shapes(num_features, num_classes, S, m=m)
    >>> len(result)
    3
    >>> result[0][0]
    10
    >>> result[0][1]
    11
    >>> result[1][0]
    10
    >>> result[1][1]
    11
    >>> result[2][0]
    5
    >>> result[2][1]
    11
    >>> num_features = 200
    >>> num_classes = 10
    >>> m = 15
    >>> result = Theta_or_Delta_shapes(num_features, num_classes, 200, 200, m=m)
    >>> len(result)
    3
    >>> result[0][0]
    15
    >>> result[0][1]
    200
    >>> result[0][2]
    201
    >>> result[1][0]
    15
    >>> result[1][1]
    200
    >>> result[1][2]
    201
    >>> result[2][0]
    15
    >>> result[2][1]
    10
    >>> result[2][2]
    201
    """
    if num_classes <= 2:
        num_classes = 1

    # Convert num_layer_units to np array in case it's a generator.
    S = np.array(S)
    S = S.flatten()
    result = list()

    S_l = num_features + 1
    S_l1 = S[0] if len(S) else num_classes
    result.append((m, S_l1, S_l) if m else (S_l1, S_l))
    for i, S_l in enumerate(S):
        S_l += 1
        if i < len(S) - 1:
            S_l1 = S[i + 1]
        else:
            S_l1 = num_classes
        result.append((m, S_l1, S_l) if m else (S_l1, S_l))
    return result


def optimize_EPSILON_INIT(L_in, L_out):
    """
    Calculates the most optimized EPSILON for initializing random θ.
    :param L_in: S_l.
    :param L_out: S_(l + 1)
    :return: Optimized EPSILON based on layer unit counts.
    """
    return np.sqrt(6)/(np.sqrt(L_in + L_out))


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
    return np.divide(1, (1 + np.power(np.e, -z)))


def save_training_info_to_file(directory, iteration_num=None, cost=None, accuracy=None, Theta=None):
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_wo_extension = directory + '/Theta_' + str(time.time())
    path_w_extension = path_wo_extension + '.txt'

    # Save unrolled Theta
    unrolled_list = [theta_l.flatten() for theta_l in Theta]
    unrolled = np.array([])
    for ele in unrolled_list:
        unrolled = np.append(unrolled, ele)
    np.savetxt(path_w_extension, unrolled)

    # Save other information
    theta_shapes = ''
    for theta_l in Theta:
        theta_shapes += '{0} {1}; '.format(np.size(theta_l, axis=0), np.size(theta_l, axis=1))
    theta_shapes = theta_shapes[: -2]

    path_wo_extension += '_info'
    path_w_extension = path_wo_extension + '.txt'
    with open(path_w_extension, 'w') as file:
        file.write('theta length = {0}\n'.format(len(Theta)))
        file.write('theta shapes = {0}\n'.format(theta_shapes))
        file.write('iteration number = {0}\n'.format(iteration_num))
        file.write('cost = {0}\n'.format(cost))
        file.write('accuracy = {0}%\n'.format(accuracy))
        file.close()


def copy_list_of_ndarray(lst):
    result = []
    for ele in lst:
        result.append(np.copy(ele))
    return result


def rand_int_in_range_lst(start, stop, count=1):
    """
    Generates a list with count number of random integers within specified range.
    :param start: Start of range (inclusive).
    :param stop: End of range (exclusive).
    :param count: Number of integers to generate.
    :return: List of all generated integers.
    >>> start = 0
    >>> stop = 10
    >>> count = 1
    >>> result = rand_int_in_range_lst(start, stop, count)
    >>> len(result) == count
    True
    >>> all_in_range = True
    >>> for ele in result:
    ...     if ele < start or ele >= stop:
    ...         all_in_range = False
    ...         break
    >>> all_in_range
    True
    >>> start = 0
    >>> stop = 10
    >>> count = 0
    >>> result = rand_int_in_range_lst(start, stop, count)
    >>> len(result) == count
    True
    >>> all_in_range = True
    >>> for ele in result:
    ...     if ele < start or ele >= stop:
    ...         all_in_range = False
    ...         break
    >>> all_in_range
    True
    >>> start = 0
    >>> stop = 1
    >>> count = 5
    >>> result = rand_int_in_range_lst(start, stop, count)
    >>> len(result) == count
    True
    >>> all_in_range = True
    >>> for ele in result:
    ...     if ele < start or ele >= stop:
    ...         all_in_range = False
    ...         break
    >>> all_in_range
    True
    """
    assert(stop > start)
    result = list()
    for _ in range(count):
        result.append(random.randrange(start, stop))
    return result


class DataProcessor:
    @staticmethod
    def add_x0_column(A):
        return np.insert(A, obj=0, values=1, axis=1)

    @staticmethod
    def augmented_to_coefficient_and_b(A):
        return A[:, :-1], A[:, -1]

    @staticmethod
    def partition(A, idx_1, idx_2):
        return A[:idx_1], A[idx_1:idx_2], A[idx_2:]

    @staticmethod
    def get_unique_categories(output, case_sensitive=True):
        if not case_sensitive:
            output = [x.lower() if isinstance(x, str) else x for x in output]
        output = np.hstack(output)
        return np.unique(output)

    @staticmethod
    def get_unique_categories_and_binary_outputs(output, case_sensitive=True):
        unique_cat = DataProcessor.get_unique_categories(output, case_sensitive)

        if np.size(unique_cat) <= 2:
            outputs_b = np.zeros((np.size(output), 1))
            mask = (output == unique_cat[0]).flatten('F')
            outputs_b[mask] = 1
        else:
            outputs_b = np.zeros((np.size(output), np.size(unique_cat)))
            mask = np.repeat(np.matrix(unique_cat), np.size(outputs_b, axis=0), axis=0)
            output_2d = np.repeat(np.matrix(output).T, np.size(unique_cat), axis=1)
            outputs_b = np.where(mask == output_2d, 1, 0)

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
        mask = self.__std == 0
        self.__std[mask] = 1


if __name__ == '__main__':
    doctest.testmod()
