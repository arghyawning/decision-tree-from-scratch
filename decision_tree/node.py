from decision_tree.question import Question

import numpy as np


class Node:
    def __init__(self, data):
        self.data = data
        self.feature = -1
        self.threshold = -1
        self.yes = None
        self.no = None
        self.impurity = 0
        self.question = None
        self.leaf = False
        self.depth = 0

    def gini_impurity(self):
        labels = self.data[:, -1]
        if labels.size == 0:
            return 0
        counts = np.unique(labels, return_counts=True)[1]
        return 1 - np.sum(np.square(counts / labels.size))

    def best_question(self):
        best_gain = 0
        best_question = None
        for feature in range(self.data.shape[1] - 1):
            for threshold in np.unique(self.data[:, feature]):
                question = Question(self.data, feature, threshold, self.impurity)
                question.split()
                question.calc_information_gain()
                if question.information_gain > best_gain:
                    best_gain = question.information_gain
                    best_question = question
        return best_question

    def ini_node(self, depth, max_depth, min_samples):
        self.depth = depth
        if self.depth >= max_depth:
            self.leaf = True
            return
        if self.data.shape[0] < min_samples:
            self.leaf = True
            return
        self.impurity = self.gini_impurity()
        if self.impurity == 0:
            self.leaf = True
            return
        self.question = self.best_question()
        # if question is None:
        #     self.leaf = True
        #     return
        # self.question = question
        # self.question.split()
        self.yes = Node(self.question.yes_data)
        self.yes.ini_node(depth + 1, max_depth, min_samples)
        self.no = Node(self.question.no_data)
        self.no.ini_node(depth + 1, max_depth, min_samples)
        # return self

    def predict(self, testx):
        if self.leaf:
            labels = self.data[:, -1]
            counts = np.unique(labels, return_counts=True)[1]
            # print("Predicted:", np.argmax(counts))
            # return
            return labels[np.argmax(counts)]
        if testx[self.question.feature] <= self.question.threshold:
            return self.yes.predict(testx)
        else:
            return self.no.predict(testx)
