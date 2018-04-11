import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import export_graphviz
import graphviz
import warnings

warnings.filterwarnings("ignore")

# Load mobile data data.
data = pd.read_csv('../data/train.csv',index_col=0)

#Clean data
y = data.output
X = data.drop('output', axis=1)

#Model training
model = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = 1,random_state=50, max_features = "auto", min_samples_leaf=2, max_depth=6)
finalTree = model.fit(X, y)

#taking test data
testData = pd.read_csv('../data/test.csv', index_col=0)
y_test = data.output
X_test = data.drop('output', axis=1)

#generating predictions
pred = model.predict(X_test)

#printing evaluating parameters
print ("MSE = ",mean_squared_error(y_test, pred))
print ("Accuracy = ",accuracy_score(y_test, pred))

print("1000 trees are iterating!!! ")
 
for tree_in_forest in finalTree.estimators_:
	export_graphviz(tree_in_forest,feature_names=X.columns,filled=True, rounded=True)
   
print ("Generating png file ...")
os.system('dot -Tpng tree.dot -o tree.png')
with open("tree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)

print("Open Codes folder to see the tree.png file")