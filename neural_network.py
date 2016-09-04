import util
import nn_alg as alg
import numpy as np
import math
import time


class NeuralNetwork:
    def __init__(self, X, y, test_ratio=0.1, alpha=0.01, lamb=0, EPSILON_INIT=1, *S):
        n = np.size(X, axis=1)
        training_count = math.floor(np.size(X, axis=0) * (1 - test_ratio))
        self.__X, self.__X_test = util.DataProcessor.partition(X, training_count)
        self.__y, self.__y_test = util.DataProcessor.partition(y, training_count)
        self.__m = np.size(self.__X, axis=0)
        self.__unique_cat, self.__y = util.DataProcessor.get_unique_categories_and_binary_outputs(self.__y)
        K = np.size(self.__unique_cat)
        self.__lamb = lamb
        self.__alpha = alpha
        self.__EPSILON_INIT = EPSILON_INIT
        self.__Theta = util.rand_Theta(n, K, *S, EPSILON_INIT=EPSILON_INIT)
        self.__Delta = util.zero_Delta(n, K, *S)
        self.__feature_normalizer = util.FeatureNormalizer(self.__X)
        self.__X = self.__feature_normalizer.normalized_feature()
        self.__neurons = None
        self.__delta = None
        self.__D = None

    def train(self, iter_limit=0, time_limit=0, grad_check=True):
        print('Started training...')
        start = time.time()
        check_iter_limit = iter_limit is not 0
        i = 0
        while (check_iter_limit and i < iter_limit) or not check_iter_limit:
            if (i + 1) % 10 is 0:
                time_elapsed = time.time() - start
                print('Iteration {0}, time passed {1}s...'.format(i + 1, np.round(time_elapsed, 2)))
                if time_limit is not 0 and time_elapsed > time_limit:
                    break
            for l in range(np.size(self.__X, axis=0)):
                x_i = np.matrix(self.__X[l])
                self.__neurons = alg.nn_forward_prop(x_i, self.__Theta)
                self.__delta = alg.nn_delta(self.__neurons, self.__Theta, np.matrix(self.__y[l]))
                self.__Delta = alg.nn_Delta(self.__Delta, self.__delta, self.__neurons)
                self.__D = alg.nn_D(self.__m, self.__Delta, self.__Theta, self.__lamb)
                self.__Theta = alg.nn_update_Theta_with_D(self.__Theta, self.__D, alpha=self.__alpha)
            if grad_check and i is 0:
                grad_check_result = alg.nn_grad_check(self.__X,
                                                      self.__y,
                                                      self.__D,
                                                      self.__Theta,
                                                      lamb=self.__lamb,
                                                      EPSILON=10 ** -4)
                if not grad_check_result:
                    raise Exception('Gradient check did not pass.')
            i += 1
        print('Finished training, time used: {0}s.'.format(time.time() - start))
        print('Calculating error rate with test samples...')
        predict_result = self.predict(self.__X_test)
        correct_count = np.count_nonzero(predict_result == self.__y_test)
        error_rate = 1 - (correct_count / np.size(self.__y_test, axis=0))
        print('Error rate: {0:.2f}%.'.format(error_rate * 100))

    def predict(self, X):
        input_normalized = self.__feature_normalizer.normalize_new_feature(X)
        result = alg.nn_forward_prop(input_normalized, self.__Theta)[-1]
        mask = np.argmax(result, axis=1)
        return self.__unique_cat[mask.T]
