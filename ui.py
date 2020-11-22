import model
import random
import pandas as pd

dfab = pd.read_excel("/Users/benjamin/school/big-10-covers/18_19_games.xls", parse_dates=["Date"])
dfab.columns = ["Date", "Visitor", "V_Points", "Home", "H_Points", "OT", "Notes"]
dfab["Home Win"] = dfab["V_Points"] < dfab["H_Points"]

rand = random.randint(0, len(model.df2))

game = model.rf_most_important.predict([[1, 2, 1, 1, 1, 1, 1, 1, 1]])

print(game)

home = dfab.loc[rand]["Home"]
away = dfab.loc[rand]["Visitor"]

print("Randomly selected matchup:\n",home, "(Home) vs.", away, "(Away) - on", dfab.loc[rand]["Date"])
print("Who do you think will win?\n 1 for", home, "\n 0 for", away)
inp = input()

a = model.df2[rand][0]
b = model.df2[rand][1]
c = model.df2[rand][2]
d = model.df2[rand][3]
e = model.df2[rand][4]
f = model.df2[rand][5]
g = model.df2[rand][6]
h = model.df2[rand][7]
i = model.df2[rand][8]
pred = model.rf_most_important.predict([[a, b, c, d, e, f, g, h, i]])

ind = dfab.loc[rand]["Home Win"]
if dfab.loc[rand]["Home Win"] == True: 
    print("The winner was:", dfab.loc[rand]["Home"])
else:
    print("The winner was:", dfab.loc[rand]["Visitor"])

print("Our model predicted:",pred,"or",pred>0.5,"Answer:",ind)
print("And you predicted:", ind == dfab.loc[rand]["Home Win"] )
print("True means that the home team wins, False means that the visiting team wins.")

count = 0
for j in range(0, len(model.df2)):
    j = int(j)
    a = model.df2[j][0]
    b = model.df2[j][1]
    c = model.df2[j][2]
    d = model.df2[j][3]
    e = model.df2[j][4]
    f = model.df2[j][5]
    g = model.df2[j][6]
    h = model.df2[j][7]
    i = model.df2[j][8]
    pred = model.rf_most_important.predict([[a, b, c, d, e, f, g, h, i]])
    tof = dfab.loc[j]["Home Win"]
    if tof and pred>0.5:
        count = count + 1
print("Accuracy of 17-18 model on 18-19 data:",count/len(model.df2))
