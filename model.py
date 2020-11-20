import pandas as pd
import xlrd
# Reads excel file as a DataFrame object
df = pd.read_excel("/Users/benjamin/school/big-10-covers/17_18_games.xls", parse_dates=["Date"])

# This is pulling the first n rows from the DataFrame object
df.head(2)

# Renaming columns
df.columns = ["Date", "Visitor", "V_Points", "Home", "H_Points", "OT", "Notes"]

# Prediction class
df["Home Win"] = df["V_Points"] < df["H_Points"]

print("Home Win percentage: {0:.1f}%".format(100 * df["Home Win"].sum() / df["Home Win"].count()))

y_true = df["Home Win"].values
# Array now holds class values in format that scikit learn can read


# which team is better in previous year's standings.
# https://www.sports-reference.com/cbb/conferences/big-ten/2017.html
standing = pd.read_excel("/Users/benjamin/school/big-10-covers/16_17_standings.xls")

df["Home Last Win"] = False
df["Visitor Last Win"] = False
from collections import defaultdict
won_last = defaultdict(int)
for index, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    row["Home Last Win"] = won_last[home_team]
    row["Visitor Last Win"] = won_last[visitor_team]
    df.loc[index] = row
    #We then set our dictionary with the each team's result (from this row) for the next
    #time we see these teams.
    #Set current Win
    won_last[home_team] = row["Home Win"]
    won_last[visitor_team] = not row["Home Win"]

# Which team won their last encounter
df["Home Win Streak"] = 0
df["Visitor Win Streak"] = 0
win_streak = defaultdict(int)

for index, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    row["Home Win Streak"] = win_streak[home_team]
    row["Visitor Win Streak"] = win_streak[visitor_team]
    df.loc[index] = row    
    # Set current win
    if row["Home Win"]:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1

# The standing of the team
df["Home Ranks Higher"] = 0
for index, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    home_rank = standing[standing["School"] == home_team]["Rk"].values[0]
    visitor_rank = standing[standing["School"] == visitor_team]["Rk"].values[0]
    row["Home Rank Higher"] = int(home_rank > visitor_rank)
    df.loc[index] = row

# Which team won their last encounter team regardless of playing at home
last_match_winner = defaultdict(int)
df["Home Won Last"] = 0
for index , row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    teams = tuple(sorted([home_team, visitor_team]))
    
    row["Home Won Last"] = 1 if last_match_winner[teams] == row["Home"] else 0
    df.loc[index] = row
    # Who won this one?
    winner = row["Home"] if row["Home Win"] else row["Visitor"]
    last_match_winner[teams] = winner


X_features_only = df[['Home Win Streak', 'Visitor Win Streak', 'Home Ranks Higher', 'Home Won Last', 'Home Last Win', 'Visitor Last Win']].values

import numpy as np
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_features_only, y_true, scoring='accuracy')
print(scores)
print("Using just the last result from the home and visitor teams")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoding = LabelEncoder()
# We will fit this transformer to the home teams so that it learns an integer
# representation for each team
encoding.fit(df["Home"].values)

home_teams = encoding.transform(df["Home"].values)
visitor_teams = encoding.transform(df["Visitor"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

# OneHotEncoder transformer to encode
onehot = OneHotEncoder()
# fit and transform
X_teams = onehot.fit_transform(X_teams).todense()

X_all = np.hstack([X_features_only, X_teams])

# we run the decision tree on the new dataset
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics

X_small = df[['Home Ranks Higher', 'Home Win Streak']]
pred_train, pred_test, tar_train, tar_test  =   train_test_split(X_small, y_true, test_size=.4)

# Build model on training data
classifier = DecisionTreeClassifier()
classifier = classifier.fit(pred_train, tar_train)

predictions = classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test, predictions)

sklearn.metrics.accuracy_score(tar_test, predictions)
print("Accuracy: {0:.1f}%".format(sklearn.metrics.accuracy_score(tar_test, predictions) * 100))

# Displaying the decision tree
from sklearn import tree
# from StringIO import StringIO
from io import StringIO
# from StringIO import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)

import pydotplus
graph = pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())

#Random forest classifiers
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Using full team labels is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.model_selection import GridSearchCV

parameter_space = {
"max_features": [2, 10, 'auto'],
"n_estimators": [100,],
"criterion": ["gini", "entropy"],
"min_samples_leaf": [2, 4, 6],
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))

print(grid.best_estimator_)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features=2, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=4,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False, random_state=14,
            verbose=0, warm_start=False)

# Will be the main file

# Will run all calculations/algorithms

# Heuristics:
# Kenpom

# 

# Give final output

# Sam's Testing

# from data import *

# if __name__ == '__main__':
#     indiana_test = raw_data("indianaTest", 2019, 2020, "indiana1920.csv")
#     indiana_test.print_table()


