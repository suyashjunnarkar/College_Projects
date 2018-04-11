import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import export_graphviz
import graphviz
import warnings

warnings.filterwarnings("ignore")

 
# Load red wine data.
data = pd.read_csv('../data/Dataset.csv',index_col=0)
 
# Split data into training and test sets
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
 
# Evaluate model pipeline on test data
model = RandomForestRegressor(n_estimators = 1000, oob_score = True, n_jobs = 1,random_state =50,max_features = "auto", min_samples_leaf = 2)
finalTree = model.fit(X_train, y_train)
print (finalTree)
pred = model.predict(X_test)
print ("MSE = ",mean_squared_error(y_test, pred))

#If you don't want to see the final tree comment this section

print("1000 trees are iterating!!! ")
 
for tree_in_forest in finalTree.estimators_:
	export_graphviz(tree_in_forest,feature_names=X.columns,filled=True,rounded=True)
   
print ("Generating png file ...")
os.system('dot -Tpng tree.dot -o tree.png')
with open("tree.dot") as f:
    dot_graph=f.read()
graphviz.Source(dot_graph)

print("Open Codes folder to see the tree.png file")
