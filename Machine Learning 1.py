# Simple ML Hello World
#   1 Traing data
#   2 Train Calssifier
#   3 Test Calssifier
from sklearn import tree

# in features 0 is bumpy 1 is smooth
# in labels 0 is apple 1 is orange
features = [[140, 1], [130, 1], [150, 0], [170, 0], ]
labels = [0, 0, 1, 1]

# Calssifier for this is a decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]]))
# expected output of print is 1
