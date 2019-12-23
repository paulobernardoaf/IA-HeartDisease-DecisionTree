from sklearn import datasets, metrics, tree, svm
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import scikitplot as skplt
from sklearn.tree import export_graphviz
from subprocess import check_call
import pydot


data = pd.read_csv("./heart-disease-uci/heart.csv", sep=",")

x = data.drop(['target'], axis=1)
y = data['target']

source = data.columns.drop(['target'])

DecisionTreeClassifier = tree.DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

DecisionTreeClassifier.fit(X_train, y_train)

predictedTree = DecisionTreeClassifier.predict(X_test)

export_graphviz(
            DecisionTreeClassifier,
            out_file="myTreeName.dot",
            feature_names=source,
            class_names=["0", "1"],
            filled=True,
            rounded=True)

(graph,) = pydot.graph_from_dot_file('myTreeName.dot')
graph.write_png('tree.png')

print("Tree image saved as tree.png")