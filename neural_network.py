import util
import nn_alg as alg
import numpy as np
import math
import time


class NeuralNetwork:
    def __init__(self, X, y, test_ratio=0.1, lamb=0, *S):
        n = np.size(X, axis=1)
        K = np.size(y, axis=1)
        training_count = math.floor(np.size(X, axis=0) * (1 - test_ratio))
        self.__X, self.__X_test = util.DataProcessor.partition(X, training_count)
        self.__y, self.__y_test = util.DataProcessor.partition(y, training_count)
        self.__m = np.size(self.__X, axis=0)
        self.__unique_cat, self.__y = util.DataProcessor.get_unique_categories_and_binary_outputs(self.__y[:training_count])
        self.__lamb = lamb
        self.__Theta = util.rand_Theta(n, K, *S)
        self.__Delta = util.zero_Delta(n, K, *S)
        self.__feature_normalizer = util.FeatureNormalizer(self.__X)
        self.__X = self.__feature_normalizer.normalized_feature()
        self.__neurons = None
        self.__delta = None
        self.__D = None

    def train(self, iter_limit=0, grad_check=True):
        print('Started training...')
        start = time.time()
        for i in range(iter_limit):
            for x_i in self.__X:
                self.__neurons = alg.nn_forward_prop(x_i, self.__Theta)
                self.__delta = alg.nn_delta(self.__neurons, self.__Theta, self.__y)
                self.__Delta = alg.nn_Delta(self.__Delta, self.__delta, self.__neurons)
                self.__D = alg.nn_D(self.__m, self.__Delta, self.__Theta, self.__lamb)
                self.__Theta = alg.nn_update_Theta_with_D(self.__Theta, self.__D)
                if grad_check and i is 0:
                    grad_check_result = alg.nn_grad_check(self.__X,
                                                          self.__y,
                                                          self.__D,
                                                          self.__Theta,
                                                          lamb=self.__lamb,
                                                          EPSILON=0.0001)
                    if not grad_check_result:
                        raise Exception('Gradient check did not pass.')
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
        return self.__unique_cat[mask]

