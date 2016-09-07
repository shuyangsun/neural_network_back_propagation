import util
import nn_alg as alg
import numpy as np
import math
import time


class NeuralNetwork:
    def __init__(self, X, y, test_ratio=0.1, alpha=0.01, lamb=0, EPSILON_INIT=1, *S):
        self.__n = np.size(X, axis=1)
        training_count = math.floor(np.size(X, axis=0) * (1 - test_ratio))
        self.__X, self.__X_test = util.DataProcessor.partition(X, training_count)
        self.__y, self.__y_test = util.DataProcessor.partition(y, training_count)
        self.__m = np.size(self.__X, axis=0)
        self.__unique_cat, self.__y = util.DataProcessor.get_unique_categories_and_binary_outputs(self.__y)
        self.__K = np.size(self.__unique_cat)
        self.__S = S
        self.__lamb = lamb
        self.__alpha = alpha
        self.__EPSILON_INIT = EPSILON_INIT
        self.__Theta = util.rand_Theta(self.__n, self.__K, self.__S, EPSILON_INIT=EPSILON_INIT)
        self.__feature_normalizer = util.FeatureNormalizer(self.__X)
        self.__X = self.__feature_normalizer.normalized_feature()
        self.__neurons = None

    def train(self, iter_limit=0, time_limit=0, grad_check=True, save_to_file=False):
        print('Started training...')
        start = time.time()
        check_iter_limit = iter_limit is not 0
        cost_list = []
        accuracy_list = []
        self.__switch_to_double_precision()

        i = 0
        # Print iteration information and check iteration count or time limit.
        while (check_iter_limit and i < iter_limit) or not check_iter_limit:
            if (i + 1) % 10 is 0:
                time_elapsed = time.time() - start
                print('Iteration {0}, time passed {1}s...'.format(i + 1, np.round(time_elapsed, 2)))
                if time_limit is not 0 and time_elapsed > time_limit:
                    break

            if (i + 1) is 2 or (i + 1) % 100 is 0:
                print('-' * 50)
                print('Iteration {0} cost is {1}.'.format(i + 1, cost_list[-1]))
                print('Accuracy: {0:.2f}%.'.format(accuracy_list[-1]))
                print('-' * 50)
                if save_to_file:
                    util.save_training_info_to_file(directory='Theta_' + str(start),
                                                    iteration_num=i + 1,
                                                    cost=cost_list[-1],
                                                    accuracy=accuracy_list[-1],
                                                    Theta=self.__Theta)

            # Calculate delta, Delta and D. Then update Theta:
            Delta = util.zero_Delta(self.__n, self.__K, self.__S, m=self.__m, dtype=self.__dtype)
            self.__neurons = alg.nn_forward_prop(self.__X, self.__Theta)
            delta = alg.nn_delta(self.__neurons, self.__Theta, np.matrix(self.__y))
            Delta = alg.nn_Delta(Delta, delta, self.__neurons)
            D = alg.nn_D(self.__m, Delta, self.__Theta, lamb=0 if i is 0 else self.__lamb)

            # Do gradient check without lambda on the first iteration if turned on.
            if grad_check and i is 0:
                grad_check_result = alg.nn_grad_check(self.__X,
                                                      self.__y,
                                                      D,
                                                      self.__Theta,
                                                      lamb=0 if i is 0 else self.__lamb,
                                                      EPSILON=10 ** -4)
                if not grad_check_result:
                    print(Exception('Gradient check did not pass.'))
                else:
                    print('Passed gradient check.')

            # Update Theta after gradient checking.
            self.__Theta = alg.nn_update_Theta_with_D(self.__Theta, D, alpha=self.__alpha)

            # Record cost and accuracy change
            cost_list.append(alg.nn_J_Theta(self.__neurons[-1],
                                            self.__y,
                                            lamb=0 if i is 0 else self.__lamb,
                                            Theta=self.__Theta))
            accuracy_list.append(self.__testing_sample_accuracy())

            if i == 0:
                self.__switch_to_single_precision()

            i += 1
        print('-' * 50)
        print('Finished training, time used: {0}s.'.format(time.time() - start))
        print('Calculating error rate with test samples...')
        print('Accuracy: {0:.2f}%.'.format(accuracy_list[-1]))
        return cost_list, accuracy_list, self.__Theta

    def predict(self, X):
        input_normalized = self.__feature_normalizer.normalize_new_feature(X)
        result = alg.nn_forward_prop(input_normalized, self.__Theta)[-1]
        mask = np.argmax(result, axis=1)
        return self.__unique_cat[mask.T]

    def __testing_sample_accuracy(self):
        predict_result = self.predict(self.__X_test)
        correct_count = np.count_nonzero(predict_result == self.__y_test)
        error_rate = 1 - (correct_count / np.size(self.__y_test, axis=0))
        return (1 - error_rate) * 100

    def __switch_to_double_precision(self):
        self.__dtype = np.float64
        self.__switch_data_dtype()

    def __switch_to_single_precision(self):
        self.__dtype = np.float32
        self.__switch_data_dtype()

    def __switch_data_dtype(self):
        self.__X = self.__X.astype(self.__dtype)
        self.__y = self.__y.astype(self.__dtype)
        self.__X_test = self.__X_test.astype(self.__dtype)
        self.__y_test = self.__y_test.astype(self.__dtype)
        if self.__Theta is not None:
            for (l, theta_l) in enumerate(self.__Theta):
                self.__Theta[l] = theta_l.astype(self.__dtype)
        if self.__neurons is not None:
            for (l, a_l) in enumerate(self.__neurons):
                self.__neurons[l] = a_l.astype(self.__dtype)

