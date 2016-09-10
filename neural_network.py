import util
import nn_alg as alg
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pylab as p


class NeuralNetwork:
    def __init__(self, X, y, alpha=0.01, lamb=0, EPSILON_INIT=1, *num_units_hidden_layer):
        training_set_ratio = 0.6
        num_set_samples = np.size(X, axis=0)
        self.__n = np.size(X, axis=1)

        # Partition data to training, cross validation, and testing sets.
        num_training_set_samples = math.floor(np.size(X, axis=0) * training_set_ratio)
        num_cv_set_samples = math.floor((num_set_samples - num_training_set_samples) / 2)
        self.__X, self.__X_cv, self.__X_test = util.DataProcessor.partition(X,
                                                                            num_training_set_samples,
                                                                            num_training_set_samples +
                                                                            num_cv_set_samples)
        self.__y, self.__y_cv, self.__y_test = util.DataProcessor.partition(y,
                                                                            num_training_set_samples,
                                                                            num_training_set_samples +
                                                                            num_cv_set_samples)

        self.__m = np.size(self.__X, axis=0)
        self.__unique_cat, self.__y_b = util.DataProcessor.get_unique_categories_and_binary_outputs(self.__y)
        _, self.__y_cv_b = util.DataProcessor.get_unique_categories_and_binary_outputs(self.__y_cv)
        self.__K = np.size(self.__unique_cat)
        self.__num_units_hidden_layer_lst = num_units_hidden_layer
        self.__lamb = lamb
        self.__alpha = alpha
        self.__EPSILON_INIT = EPSILON_INIT
        self.__Theta = util.rand_Theta(self.__n, self.__K, self.__num_units_hidden_layer_lst, EPSILON_INIT=EPSILON_INIT)
        self.__feature_normalizer = util.FeatureNormalizer(self.__X)
        self.__X = self.__feature_normalizer.normalized_feature()
        self.__neurons = None

        self.cost_training_list = []
        self.cost_cv_list = []
        self.accuracy_test_list = []

    def train(self, iter_limit=0, time_limit=0, grad_check=True, info_print_frequency=10, save_to_file=False):
        print('Started training...')
        start = time.time()
        check_iter_limit = iter_limit is not 0
        self.__switch_to_double_precision()

        i = 0
        # Print iteration information and check iteration count or time limit.
        while (check_iter_limit and i < iter_limit) or not check_iter_limit:
            if i is not 0 and (i is 1 or (i is not 0 and i % info_print_frequency is 0)):
                print('Iter: {0}, duration: {1:.2f}s, J(θ_train): {2}, J(θ_cv): {3}, test set accuracy: {4:.2f}%'.format(i,
                                                                                                                         time.time() - start,
                                                                                                                         self.cost_training_list[-1],
                                                                                                                         self.cost_cv_list[-1],
                                                                                                                         self.accuracy_test_list[-1]))
                if save_to_file:
                    util.save_training_info_to_file(directory='Theta_' + str(start),
                                                    iteration_num=i,
                                                    cost=self.cost_training_list[-1],
                                                    accuracy=self.accuracy_test_list[-1],
                                                    Theta=self.__Theta)

            # Calculate delta, Delta and D. Then update Theta:
            Delta = util.zero_Delta(self.__n,
                                    self.__K,
                                    self.__num_units_hidden_layer_lst,
                                    m=self.__m,
                                    dtype=self.__dtype)
            self.__neurons = alg.nn_forward_prop(self.__X, self.__Theta)
            delta = alg.nn_delta(self.__neurons, self.__Theta, np.matrix(self.__y_b))
            Delta = alg.nn_Delta(Delta, delta, self.__neurons)
            D = alg.nn_D(self.__m, Delta, self.__Theta, lamb=0 if i is 0 else self.__lamb)

            # Do gradient check without lambda on the first iteration if turned on.
            if grad_check and i is 0:
                try:
                    alg.nn_grad_check(self.__X,
                                      self.__y_b,
                                      D,
                                      self.__Theta,
                                      lamb=0 if i is 0 else self.__lamb,
                                      EPSILON=10 ** -4)
                except alg.GradientCheckingFailsException as e:
                    print(e)
                else:
                    print('-' * 50)
            elif i is 0:
                print('-' * 50)
            
            # Update Theta after gradient checking.
            self.__Theta = alg.nn_update_Theta_with_D(self.__Theta, D, alpha=self.__alpha, dtype=self.__dtype)

            # Record cost and accuracy change
            self.cost_training_list.append(alg.nn_J_Theta(self.__neurons[-1],
                                                          self.__y_b,
                                                          lamb=0 if i is 0 else self.__lamb,
                                                          Theta=self.__Theta))
            self.cost_cv_list.append(alg.nn_J_Theta(alg.nn_forward_prop(self.__X_cv, self.__Theta)[-1],
                                                    self.__y_cv_b,
                                                    lamb=0 if i is 0 else self.__lamb,
                                                    Theta=self.__Theta))
            self.accuracy_test_list.append(self.__testing_sample_accuracy())

            if i == 0:
                self.__switch_to_single_precision()

            i += 1
            
            if time_limit is not 0 and time.time() - start > time_limit:
                break

        print('Finished training.')
        print('Iter: {0}, duration: {1:.2f}s, J(θ_train): {2}, J(θ_cv): {3}, test set accuracy: {4:.2f}%'.format(i,
                                                                                                                 time.time() - start,
                                                                                                                 self.cost_training_list[-1],
                                                                                                                 self.cost_cv_list[-1],
                                                                                                                 self.accuracy_test_list[-1]))
        print('-' * 50)

    def predict(self, X):
        input_normalized = self.__feature_normalizer.normalize_new_feature(X)
        result = alg.nn_forward_prop(input_normalized, self.__Theta)[-1]
        mask = np.argmax(result, axis=1)
        return self.__unique_cat[mask.T]

    def plot_training_info(self, color='#00BFFF'):
        plt.figure('Training Info', figsize=(30, 30))
        color=color

        # Cost training set plot
        figure = plt.subplot(2, 2, 1)
        plt.title('Cost of Training Set')
        plt.xlabel('iteration')
        plt.ylabel('cost')
        figure.plot(self.cost_training_list, color=color)
        p.fill_between(range(len(self.cost_training_list)),
                       self.cost_training_list, facecolor=color,
                       alpha=0.25)

        # Cost training set plot
        figure = plt.subplot(2, 2, 2)
        plt.title('Cost of Cross Validation Set')
        plt.xlabel('iteration')
        plt.ylabel('cost')
        figure.plot(self.cost_cv_list, color=color)
        p.fill_between(range(len(self.cost_cv_list)), self.cost_cv_list, facecolor=color, alpha=0.25)

        # Accuracy plot
        figure = plt.subplot(2, 2, 3)
        plt.title('Accuracy of Testing Set')
        plt.xlabel('iteration')
        plt.ylabel('accuracy rate (%)')
        figure.plot(self.accuracy_test_list, color=color)
        p.fill_between(range(len(self.accuracy_test_list)), self.accuracy_test_list, facecolor=color, alpha=0.25)

    def visualize_Theta(self, cmap='Greys_r', invert=False):
        for l, theta_l in enumerate(self.__Theta):
            plt.figure('Theta({0})'.format(l + 1), figsize=(30, 30))
            theta_l_no_bias_units = np.delete(theta_l, obj=0, axis=1)
            num_pic = np.size(theta_l_no_bias_units, axis=0)
            num_pxl = np.size(theta_l_no_bias_units, axis=1)

            width_num = int(np.ceil(np.sqrt(num_pic)))
            height_num = int(np.ceil(num_pic / width_num))

            width_pxl = int(np.ceil(np.sqrt(num_pxl)))
            height_pxl = int(np.ceil(num_pxl / width_pxl))

            for l, theta_l_i in enumerate(theta_l_no_bias_units):
                matrix = theta_l_i.reshape(height_pxl, width_pxl)
                figure = plt.subplot(width_num, height_num, l + 1)
                figure.axes.get_xaxis().set_visible(False)
                figure.axes.get_yaxis().set_visible(False)
                plt.imshow(-matrix if invert else matrix, cmap=cmap)
            plt.tight_layout()

    def show_plot(self):
        plt.show()

    def __testing_sample_accuracy(self):
        predict_result = self.predict(self.__X_test)
        correct_count = np.count_nonzero(predict_result.flatten('C') == self.__y_test.flatten('C'))
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

