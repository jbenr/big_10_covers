import model
import random
import pandas as pd

dfab = pd.read_excel("/Users/benjamin/school/big-10-covers/18_19_games.xls", parse_dates=["Date"])
dfab.columns = ["Date", "Visitor", "V_Points", "Home", "H_Points", "OT", "Notes"]
dfab["Home Win"] = dfab["V_Points"] < dfab["H_Points"]

rand = random.randint(0, len(model.df))

game = model.rf_most_important.predict([[1, 2, 1, 1, 1, 1, 1, 1, 1]])

print(game)

home = dfab.loc[rand]["Home"]
away = dfab.loc[rand]["Visitor"]
print("home win:", dfab.loc[rand]["Home Win"])

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

print(pred, dfab.loc[rand]["Home Win"])
ind = dfab.loc[rand]["Home Win"]
if ind == True and inp == 1:
    print("The winner was:", dfab.loc[rand]["Home"])
    print("Our model predicted: Correctly",(ind and pred > 0.5))
    print("And you predicted: Correctly. Nice.")
elif ind == True and inp == 0:
    print("The winner was:", dfab.loc[rand]["Home"])
    print("Our model predicted:", (ind and pred > 0.5))
    print("And you predicted: Incorrectly.")
elif not ind == True and inp == 1:
    print("The winner was:", dfab.loc[rand]["Visitor"])
    print("Our model predicted:",ind and pred <0.5)
    print("And you predicted: Inorrectly.")
else:
    print("The winner was:", dfab.loc[rand]["Visitor"])
    print("Our model predicted:",ind and pred <0.5)
    print("And you predicted: Correctly. Nice.")

