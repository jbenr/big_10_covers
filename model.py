import pandas as pd
import xlrd
from collections import defaultdict
# Reads excel file as a DataFrame object
df = pd.read_excel("/Users/benjamin/school/big-10-covers/17_18_games.xls", parse_dates=["Date"])
df2 = pd.read_excel("/Users/benjamin/school/big-10-covers/18_19_games.xls", parse_dates=["Date"])

# Renaming columns
df.columns = ["Date", "Visitor", "V_Points", "Home", "H_Points", "OT", "Notes"]
df2.columns = ["Date", "Visitor", "V_Points", "Home", "H_Points", "OT", "Notes"]


# Prediction class
df["Home Win"] = df["V_Points"] < df["H_Points"]
df2["Home Win"] = df2["V_Points"] < df2["H_Points"]

# Percentage of the home team winning the game
print("Home Win percentage: {0:.1f}%".format(100 * df["Home Win"].sum() / df["Home Win"].count()))

y_true = df["Home Win"].values
y_true2 = df2["Home Win"].values

# Array now holds class values in format that scikit learn can read

# which team is better in previous year's standings.
# https://www.sports-reference.com/cbb/conferences/big-ten/2017.html
standing = pd.read_excel("/Users/benjamin/school/big-10-covers/16_17_standings.xls")
standing2 = pd.read_excel("/Users/benjamin/school/big-10-covers/17_18_standings.xls")

# stats
stats = pd.read_excel("/Users/benjamin/school/big-10-covers/17_18_stats.xls")
stats2 = pd.read_excel("/Users/benjamin/school/big-10-covers/18_19_stats.xls")

teams = ["Indiana", "Ohio State", "Northwestern", "Purdue", "Illinois", "Iowa", "Rutgers", "Nebraska",
 "Michigan", "Maryland", "Wisconsin,", "Michigan State", "Penn State", "Minnesota"]

#inject stats
df["H NRtg"] = 0
df["H FGP"] = 0
df["H 3PP"] = 0
df["V NRtg"] = 0
df["V FGP"] = 0
df["V 3PP"] = 0
for i, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    h_nrtg = stats[stats["School"] == home_team]["NRtg"].values[0]
    v_nrtg = stats[stats["School"] == visitor_team]["NRtg"].values[0]
    row["H NRtg"] = h_nrtg
    row["V NRtg"] = v_nrtg
    h_fgp = stats[stats["School"] == home_team]["FG%"].values[0]
    v_fgp = stats[stats["School"] == home_team]["FG%"].values[0]
    row["H FGP"] = h_fgp
    row["V FGP"] = v_fgp
    h_3pp = stats[stats["School"] == home_team]["3P%"].values[0]
    v_3pp = stats[stats["School"] == home_team]["3P%"].values[0]
    row["H 3PP"] = h_fgp
    row["V 3PP"] = v_fgp
    df.loc[i] = row

df2["H NRtg"] = 0
df2["H FGP"] = 0
df2["H 3PP"] = 0
df2["V NRtg"] = 0
df2["V FGP"] = 0
df2["V 3PP"] = 0
for i, row in df2.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    h_nrtg = stats[stats["School"] == home_team]["NRtg"].values[0]
    v_nrtg = stats[stats["School"] == visitor_team]["NRtg"].values[0]
    row["H NRtg"] = h_nrtg
    row["V NRtg"] = v_nrtg
    h_fgp = stats[stats["School"] == home_team]["FG%"].values[0]
    v_fgp = stats[stats["School"] == home_team]["FG%"].values[0]
    row["H FGP"] = h_fgp
    row["V FGP"] = v_fgp
    h_3pp = stats[stats["School"] == home_team]["3P%"].values[0]
    v_3pp = stats[stats["School"] == home_team]["3P%"].values[0]
    row["H 3PP"] = h_fgp
    row["V 3PP"] = v_fgp
    df2.loc[i] = row

# Heuristic questions:
# 1) Which team won the last time these two teams played?
df["Home Last Win"] = False
df["Visitor Last Win"] = False
won_last = defaultdict(int)
for i, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    row["Home Last Win"] = won_last[home_team]
    row["Visitor Last Win"] = won_last[visitor_team]
    df.loc[i] = row
    #We then set our dictionary with the each team's result (from this row) for the next time we see these teams.
    #Set current Win
    won_last[home_team] = row["Home Win"]
    won_last[visitor_team] = not row["Home Win"]

df2["Home Last Win"] = False
df2["Visitor Last Win"] = False
won_last = defaultdict(int)
for i, row in df2.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    row["Home Last Win"] = won_last[home_team]
    row["Visitor Last Win"] = won_last[visitor_team]
    df2.loc[i] = row
    #We then set our dictionary with the each team's result (from this row) for the next time we see these teams.
    #Set current Win
    won_last[home_team] = row["Home Win"]
    won_last[visitor_team] = not row["Home Win"]

# 2) Are either teams on a winning/losing streak? If so, how many games have they won/lost?
df["Home Win Streak"] = 0
df["Visitor Win Streak"] = 0
df["Home Lose Streak"] = 0
df["Visitor Lose Streak"] = 0
win_streak = defaultdict(int)
lose_streak = defaultdict(int)
for i, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    row["Home Win Streak"] = win_streak[home_team]
    row["Visitor Win Streak"] = win_streak[visitor_team]
    row["Home Lose Streak"] = lose_streak[home_team]
    row["Visitor Lose Streak"] = lose_streak[visitor_team]
    df.loc[i] = row
    # Set current win
    if row["Home Win"]:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
        lose_streak[home_team] = 0
        lose_streak[visitor_team] += 1
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1
        lose_streak[home_team] = 0
        lose_streak[visitor_team] += 1

df2["Home Win Streak"] = 0
df2["Visitor Win Streak"] = 0
df2["Home Lose Streak"] = 0
df2["Visitor Lose Streak"] = 0
win_streak = defaultdict(int)
lose_streak = defaultdict(int)
for i, row in df2.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    row["Home Win Streak"] = win_streak[home_team]
    row["Visitor Win Streak"] = win_streak[visitor_team]
    row["Home Lose Streak"] = lose_streak[home_team]
    row["Visitor Lose Streak"] = lose_streak[visitor_team]
    df2.loc[i] = row
    # Set current win
    if row["Home Win"]:
        win_streak[home_team] += 1
        win_streak[visitor_team] = 0
        lose_streak[home_team] = 0
        lose_streak[visitor_team] += 1
    else:
        win_streak[home_team] = 0
        win_streak[visitor_team] += 1
        lose_streak[home_team] = 0
        lose_streak[visitor_team] += 1

# 3) Who was ranked higher last year?
# The standing of the teams (previous year)
df["Home Ranks Higher"] = 0
for i, row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    home_rank = standing[standing["School"] == home_team]["Rk"].values[0]
    visitor_rank = standing[standing["School"] == visitor_team]["Rk"].values[0]
    row["Home Ranks Higher"] = int(home_rank > visitor_rank)
    df.loc[i] = row

df2["Home Ranks Higher"] = 0
for i, row in df2.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    home_rank = standing[standing["School"] == home_team]["Rk"].values[0]
    visitor_rank = standing[standing["School"] == visitor_team]["Rk"].values[0]
    row["Home Ranks Higher"] = int(home_rank > visitor_rank)
    df2.loc[i] = row

# Which team won their last encounter team regardless of playing at home
last_match_winner = defaultdict(int)
df["Home Won Last"] = 0
for i , row in df.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    teams = tuple(sorted([home_team, visitor_team]))
    
    row["Home Won Last"] = 1 if last_match_winner[teams] == row["Home"] else 0
    df.loc[i] = row
    # Who won this one?
    winner = row["Home"] if row["Home Win"] else row["Visitor"]
    last_match_winner[teams] = winner

last_match_winner = defaultdict(int)
df2["Home Won Last"] = 0
for i , row in df2.iterrows():
    home_team = row["Home"]
    visitor_team = row["Visitor"]
    teams = tuple(sorted([home_team, visitor_team]))
    
    row["Home Won Last"] = 1 if last_match_winner[teams] == row["Home"] else 0
    df2.loc[i] = row
    # Who won this one?
    winner = row["Home"] if row["Home Win"] else row["Visitor"]
    last_match_winner[teams] = winner

features_set = df[['H NRtg', 'V NRtg', 'H FGP', 'V FGP', 'H 3PP', 'V 3PP',
 'Home Win Streak', 'Visitor Win Streak', 
 'Home Lose Streak', 'Visitor Lose Streak',
 'Home Ranks Higher', 
 'Home Won Last', 
 'Home Last Win', 'Visitor Last Win']].values

import numpy as np
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=14)
from sklearn.model_selection import cross_val_score

# we use cross_val_score to test the accuracy of the model thus far using k-fold cross validation
scores = cross_val_score(clf, features_set, y_true, scoring='accuracy')
print("All features set -","Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoding = LabelEncoder()
# We will fit this transformer to the home teams so that it learns an integer representation for each team
encoding.fit(df["Home"].values)
encoding.fit(df2["Home"].values)

# transforming text data to numbers so that sklearn can read it
home_teams = encoding.transform(df["Home"].values)
visitor_teams = encoding.transform(df["Visitor"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T

home_teams = encoding.transform(df2["Home"].values)
visitor_teams = encoding.transform(df2["Visitor"].values)

# onehot turns the encoded numbers into different columns to avoid collusion of data
hot = OneHotEncoder()
X_teams = hot.fit_transform(X_teams).todense()
X_all = np.hstack([features_set, X_teams])

# we run the decision tree on the new dataset
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Adding Home Court Advantage -","Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.metrics
p_train, p_test, t_train, t_test = train_test_split(X_all, y_true, test_size=.4)

# Build model on training data
classifier = DecisionTreeClassifier()
classifier = classifier.fit(p_train, t_train)

predictions = classifier.predict(p_test)
#print(predictions)

#confusion matrix to help visualize predictions
#print(sklearn.metrics.confusion_matrix(t_test, predictions))

sklearn.metrics.accuracy_score(t_test, predictions)
print("Decision tree classifier trained model -","Accuracy: {0:.1f}%".format(sklearn.metrics.accuracy_score(t_test, predictions) * 100))

# Random forest classifiers
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Random Forest Classifier -","Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

ans = df["Home Win"]

# Hard to figure out which of these to include by guess and check, so why not have sci-kit learn do it for us
df = pd.get_dummies(df)
df2 = pd.get_dummies(df2)
# Labels are the values we want to predict
labels = np.array(df['Home Win'])
labels2 = np.array(df2['Home Win'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('H_Points', axis = 1)
df2 = df2.drop('H_Points', axis = 1)
df = df.drop('V_Points', axis = 1)
df2 = df2.drop('V_Points', axis = 1)
df= df.drop('Home Win', axis = 1)
df2= df2.drop('Home Win', axis = 1)
df= df.drop('Date', axis = 1)
df2= df2.drop('Date', axis = 1)
# Saving feature names for later use
df_list = list(df.columns)
df2_list = list(df2.columns)
# Convert to numpy array
df = np.array(df)
df2 = np.array(df2)

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(df, ans, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the RandomForest classifier
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
clf = RandomForestClassifier(n_estimators=1000)
# Train the model on training data

clf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = clf.predict(test_features)

# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(df, round(importance, 2)) for df, importance in zip(df_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the 8 most important features
important_indices = [df_list.index('V NRtg'), df_list.index('H NRtg'), df_list.index('Visitor Win Streak'),
df_list.index('Home Win Streak'), df_list.index('Home Lose Streak'), df_list.index('H FGP'), df_list.index('H 3PP'),
df_list.index('V FGP'), df_list.index('V 3PP')]

train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
