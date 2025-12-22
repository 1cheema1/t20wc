winners=[
    'England',
    'Pakistan',
    'Pakistan',
    'India',
]

wins={}

for team in winners:
    if team not in wins:
        wins[team]=0
    wins[team]= wins[team] + 1
print (wins)

matches = [
    ("India", "England"),
    ("India", 'Pakistan'),
    ("England", "Pakistan"),
    ("India", "England")
]

num_matches={}

for team1,team2 in matches:
    for team in team1,team2:
        if team not in num_matches:
            num_matches[team]=0
        num_matches[team]+=1
print(num_matches)

winp={}

for team in wins:
    winp[team]= wins[team] / num_matches[team]
print (winp)

def predwin(t1,t2):
    if winp[t1]>winp[t2]:
        return (t1 +' will win')
    elif winp[t2]==winp[t1]:
        return 'Inconclusive'
    else:
        return (t2 + 'will win')

print(predwin('Pakistan','India'))