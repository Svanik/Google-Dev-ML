from sklearn import datasets
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
"""Uses the Iris Dataset splits it up into 2 setes train and test
    test is 40% of original dataset
    Then uses Decision Tree, KNN, and Logistic Regression and makes predictions
    on the teting data and prints out the accuracy Knn and Decision tree are
    better at this task then logical Regression
    Knn is hand coded and K is 1 uses the distance fromula"""
from scipy.spatial import distance


def euc(a, b):
    """Distance between test point and closest traing point."""
    return distance


class ScrappyKnn():
    """Hand-Coded KNN."""

    def fit(self, x_train, y_train):
        """Memorizes the traing data."""
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        """Knn predictions."""
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        """Saves label of cloest point loops over all points"""
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            distance = euc(row, self.x_train[i])
            if distance < best_dist:
                best_dist = distance
                best_index = i
        return self.y_train[best_index]


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.4, random_state=7)


clftree = tree.DecisionTreeClassifier()
clftree.fit(x_train, y_train)

treepredictions = clftree.predict(x_test)
print(accuracy_score(treepredictions, y_test))

clfknn = ScrappyKnn()
clfknn.fit(x_train, y_train)

knnpreditions = clfknn.predict(x_test)
print(accuracy_score(knnpreditions, y_test))

clflr = LogisticRegression()
clflr.fit(x_train, y_train)

lrpredictions = clflr.predict(x_test)
print(accuracy_score(lrpredictions, y_test))
