from decision_tree.node import Node


class Tree:
    def __init__(self, data, max_depth, min_samples):
        self.root = None
        self.data = data
        self.max_depth = max_depth
        self.min_samples = min_samples

    def build_tree(self):
        self.root = Node(self.data)
        self.root.ini_node(0, self.max_depth, self.min_samples)

    def print(self):
        self.print_node(self.root, spacing="")

    def print_node(self, node, spacing):
        # Base case: we've reached a leaf
        if node.leaf:
            predictions = {}  # a dictionary of label -> count.
            for row in node.data:
                # in our dataset format, the label is always the last column
                label = row[-1]
                if label not in predictions:
                    predictions[label] = 0
                predictions[label] += 1
            print(spacing + "Predict", predictions)
            return

        # Print the question at this node
        # print(spacing + str(node.question))
        print(
            spacing
            + str(node.question.feature)
            + " , "
            + str(node.question.data[:, node.question.feature].size)
            + ","
            + str(node.question.threshold)
        )

        # Call this function recursively on the true branch
        print(spacing + "--> True:")
        self.print_node(node.yes, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + "--> False:")
        self.print_node(node.no, spacing + "  ")

    def predict(self, testx):
        if self.root is None:
            print("Tree not built")
            return -1
        else:
            return self.root.predict(testx)
