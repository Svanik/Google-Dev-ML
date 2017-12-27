"""Uses the Iris Dataset splits it up into 2 setes train and test
    test is 40% of original dataset
    Then uses Decision Tree, KNN, and Logistic Regression and makes predictions
    on the teting data and prints out the accuracy Knn and Decision tree are
    better at this task then logical Regression"""

from sklearn import datasets
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.44, random_state=7)


clftree = tree.DecisionTreeClassifier()
clftree.fit(x_train, y_train)

treepredictions = clftree.predict(x_test)
print(accuracy_score(treepredictions, y_test))

clfknn = KNeighborsClassifier()
clfknn.fit(x_train, y_train)

knnpreditions = clfknn.predict(x_test)
print(accuracy_score(knnpreditions, y_test))

clflr = LogisticRegression()
clflr.fit(x_train, y_train)

lrpredictions = clflr.predict(x_test)
print(accuracy_score(lrpredictions, y_test))
