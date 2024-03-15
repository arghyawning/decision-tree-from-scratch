import numpy as np


class Question:
    def __init__(self, data, feature, threshold, impurity):
        self.data = data
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.information_gain = 0
        self.yes_data = None
        self.no_data = None

    def temp_impurity(self):
        labels = self.data[:, -1]
        if labels.size == 0:
            return 0
        yes_labels = self.yes_data[:, -1]
        no_labels = self.no_data[:, -1]
        yes_counts = np.unique(yes_labels, return_counts=True)[1]
        no_counts = np.unique(no_labels, return_counts=True)[1]
        yes_impurity = 1 - np.sum(np.square(yes_counts / yes_labels.size))
        no_impurity = 1 - np.sum(np.square(no_counts / no_labels.size))
        return yes_impurity, no_impurity

    def calc_information_gain(self):
        yes_impurity, no_impurity = self.temp_impurity()
        weighted_avg = (
            self.yes_data.shape[0] * yes_impurity + self.no_data.shape[0] * no_impurity
        ) / self.data.shape[0]
        self.information_gain = self.impurity - weighted_avg
        # return self.information_gain

    def split(self):
        self.yes_data = self.data[self.data[:, self.feature] <= self.threshold]
        self.no_data = self.data[self.data[:, self.feature] > self.threshold]
        # self.information_gain = self.information_gain()
