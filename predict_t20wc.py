
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict

df = pd.read_csv("data/t20i_matches_clean.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df.dropna(subset=["winner"]).reset_index(drop=True)

print(f"Loaded {len(df)} matches with results ({df['date'].min().date()} → {df['date'].max().date()})")
print()

ELO_START = 1000
ELO_K = 32

def elo_expected(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

def elo_update(rating_winner, rating_loser, k=ELO_K):
    expected_w = elo_expected(rating_winner, rating_loser)
    new_winner = rating_winner + k * (1 - expected_w)
    new_loser = rating_loser + k * (0 - (1 - expected_w))
    return new_winner, new_loser

RECENT_N = 10

def compute_recent_form(df, n=RECENT_N):
    """Opponent-quality-weighted recent form so wins vs minnows count less."""
    team_history = defaultdict(list)  # stores (opp_elo, result)
    running_elo = defaultdict(lambda: ELO_START)
    team1_form = []
    team2_form = []

    for idx, row in df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        winner = row["winner"]

        def _recent_rate(team):
            hist = team_history[team]
            if len(hist) == 0:
                return 0.5
            recent = hist[-n:]
            total_w = 0
            weighted_sum = 0
            for opp_elo, result in recent:
                w = max(opp_elo / ELO_START, 0.3)
                weighted_sum += w * result
                total_w += w
            return weighted_sum / total_w if total_w > 0 else 0.5

        team1_form.append(_recent_rate(t1))
        team2_form.append(_recent_rate(t2))

        team_history[t1].append((running_elo[t2], 1 if winner == t1 else 0))
        team_history[t2].append((running_elo[t1], 1 if winner == t2 else 0))

        if winner == t1:
            running_elo[t1], running_elo[t2] = elo_update(running_elo[t1], running_elo[t2])
        else:
            running_elo[t2], running_elo[t1] = elo_update(running_elo[t2], running_elo[t1])

    return team1_form, team2_form


print(f"Step 36: Computing recent form (last {RECENT_N} matches)...")
df["team1_recent_form"], df["team2_recent_form"] = compute_recent_form(df)
print("  ✓ Recent form computed")
print()

def compute_features(df):
    elo = defaultdict(lambda: ELO_START)
    h2h_played = defaultdict(int)
    h2h_won = defaultdict(lambda: defaultdict(int))

    features = []

    for idx, row in df.iterrows():
        t1 = row["team1"]
        t2 = row["team2"]
        winner = row["winner"]
        matchup = tuple(sorted([t1, t2]))

        t1_elo = elo[t1]
        t2_elo = elo[t2]

        total_h2h = h2h_played[matchup]
        h2h_t1 = h2h_won[matchup][t1] / total_h2h if total_h2h > 0 else 0.5

        toss_winner_is_t1 = 1 if row["toss_winner"] == t1 else 0
        toss_decision_bat = 1 if row["toss_decision"] == "bat" else 0

        winner_is_t1 = 1 if winner == t1 else 0

        features.append({
            "team1_elo": t1_elo,
            "team2_elo": t2_elo,
            "elo_diff": t1_elo - t2_elo,
            "team1_recent_form": row["team1_recent_form"],
            "team2_recent_form": row["team2_recent_form"],
            "h2h_team1_rate": h2h_t1,
            "toss_winner_is_team1": toss_winner_is_t1,
            "toss_decision_bat": toss_decision_bat,
            "winner_is_team1": winner_is_t1,
        })

        if winner == t1:
            elo[t1], elo[t2] = elo_update(elo[t1], elo[t2])
        else:
            elo[t2], elo[t1] = elo_update(elo[t2], elo[t1])

        h2h_played[matchup] += 1
        h2h_won[matchup][winner] += 1

    return pd.DataFrame(features), elo, h2h_played, h2h_won


print("Step 37: Engineering features (with Elo ratings)...")
feat_df, final_elo, final_h2h_played, final_h2h_won = compute_features(df)

FEATURE_COLS = [
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "team1_recent_form",
    "team2_recent_form",
    "h2h_team1_rate",
    "toss_winner_is_team1",
    "toss_decision_bat",
]

X = feat_df[FEATURE_COLS]
y = feat_df["winner_is_team1"]

print(f"  ✓ Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
print(f"  ✓ Target balance: {y.mean():.3f} (team1 win rate)")
print(f"  ✓ NaN check: {X.isnull().sum().sum()} NaN values")
print()

print("  Top 15 Elo ratings after all matches:")
elo_sorted = sorted(final_elo.items(), key=lambda x: x[1], reverse=True)
for rank, (team, rating) in enumerate(elo_sorted[:15], 1):
    print(f"    {rank:2d}. {team:25s}  {rating:.0f}")
print()

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Step 38: Training models (train={len(X_train)}, test={len(X_test)})...")
print()

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

print("┌─────────────────────────────────────────────┐")
print("│         LOGISTIC REGRESSION RESULTS          │")
print("├─────────────────────────────────────────────┤")
print(f"│  Accuracy: {lr_acc:.4f}                        │")
print("└─────────────────────────────────────────────┘")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print()

gbt = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)
gbt.fit(X_train_scaled, y_train)
gbt_pred = gbt.predict(X_test_scaled)
gbt_acc = accuracy_score(y_test, gbt_pred)

print("┌─────────────────────────────────────────────┐")
print("│      GRADIENT BOOSTED TREES RESULTS          │")
print("├─────────────────────────────────────────────┤")
print(f"│  Accuracy: {gbt_acc:.4f}                        │")
print("└─────────────────────────────────────────────┘")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, gbt_pred))
print()

best_model_name = "GBT" if gbt_acc >= lr_acc else "Logistic Regression"
best_model = gbt if gbt_acc >= lr_acc else lr
print(f"→ Best model: {best_model_name} ({max(lr_acc, gbt_acc):.4f})")
print()

print("Step 39: Feature Importance")
print()

print("Logistic Regression Coefficients:")
lr_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "coefficient": lr.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
for _, row in lr_importance.iterrows():
    bar = "█" * int(abs(row["coefficient"]) * 10)
    sign = "+" if row["coefficient"] > 0 else "-"
    print(f"  {sign} {row['feature']:25s}  {row['coefficient']:+.4f}  {bar}")
print()

print("Gradient Boosted Trees Importances:")
gbt_importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": gbt.feature_importances_,
}).sort_values("importance", ascending=False)
for _, row in gbt_importance.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {row['feature']:25s}  {row['importance']:.4f}  {bar}")
print()

print("=" * 60)
print("  STEP 40: ICC T20 WORLD CUP 2026 PREDICTIONS")
print("=" * 60)
print()

ICC_RANKINGS = {
    "India": 272,
    "England": 260,
    "Australia": 258,
    "New Zealand": 250,
    "South Africa": 245,
    "Pakistan": 238,
    "West Indies": 235,
    "Sri Lanka": 227,
    "Bangladesh": 223,
    "Afghanistan": 221,
    "Zimbabwe": 200,
    "Ireland": 190,
    "Netherlands": 182,
    "Scotland": 180,
    "Namibia": 179,
    "Nepal": 176,
    "United Arab Emirates": 175,
    "United States of America": 175,
    "Canada": 152,
    "Oman": 151,
    "Italy": 120,
}

MODEL_WEIGHT = 0.60
ICC_WEIGHT = 0.40

# Build opponent-quality-weighted recent form for predictions
_running_elo_pred = defaultdict(lambda: ELO_START)
team_recent_weighted = defaultdict(list)
for _, row in df.iterrows():
    t1 = row["team1"]
    t2 = row["team2"]
    winner = row["winner"]
    team_recent_weighted[t1].append((_running_elo_pred[t2], 1 if winner == t1 else 0))
    team_recent_weighted[t2].append((_running_elo_pred[t1], 1 if winner == t2 else 0))
    if winner == t1:
        _running_elo_pred[t1], _running_elo_pred[t2] = elo_update(_running_elo_pred[t1], _running_elo_pred[t2])
    else:
        _running_elo_pred[t2], _running_elo_pred[t1] = elo_update(_running_elo_pred[t2], _running_elo_pred[t1])


def get_team_stats(team):
    elo_rating = final_elo[team]
    recent = team_recent_weighted[team][-RECENT_N:]
    if not recent:
        return elo_rating, 0.5
    total_w = 0
    weighted_sum = 0
    for opp_elo, result in recent:
        w = max(opp_elo / ELO_START, 0.3)
        weighted_sum += w * result
        total_w += w
    recent_form = weighted_sum / total_w if total_w > 0 else 0.5
    return elo_rating, recent_form


def icc_win_prob(team1, team2):
    r1 = ICC_RANKINGS.get(team1, 100)
    r2 = ICC_RANKINGS.get(team2, 100)
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 100))


def predict_match(team1, team2, toss_winner=None, toss_decision="field"):
    t1_elo, t1_rf = get_team_stats(team1)
    t2_elo, t2_rf = get_team_stats(team2)

    matchup = tuple(sorted([team1, team2]))
    total_h2h = final_h2h_played[matchup]
    h2h_t1 = final_h2h_won[matchup][team1] / total_h2h if total_h2h > 0 else 0.5

    toss_is_t1 = 1 if toss_winner == team1 else 0 if toss_winner == team2 else 0.5
    toss_bat = 1 if toss_decision == "bat" else 0

    features_raw = pd.DataFrame([{
        "team1_elo": t1_elo,
        "team2_elo": t2_elo,
        "elo_diff": t1_elo - t2_elo,
        "team1_recent_form": t1_rf,
        "team2_recent_form": t2_rf,
        "h2h_team1_rate": h2h_t1,
        "toss_winner_is_team1": toss_is_t1,
        "toss_decision_bat": toss_bat,
    }])

    features_scaled = scaler.transform(features_raw)
    model_prob = best_model.predict_proba(features_scaled)[0]
    model_t1_prob = model_prob[1]

    icc_t1_prob = icc_win_prob(team1, team2)
    t1_prob = MODEL_WEIGHT * model_t1_prob + ICC_WEIGHT * icc_t1_prob
    t2_prob = 1.0 - t1_prob

    predicted_winner = team1 if t1_prob > 0.5 else team2
    return predicted_winner, t1_prob, t2_prob


WC_GROUPS = {
    "A": ["Pakistan", "India", "United States of America", "Netherlands", "Namibia"],
    "B": ["Sri Lanka", "Australia", "Ireland", "Zimbabwe", "Oman"],
    "C": ["West Indies", "England", "Scotland", "Nepal", "Italy"],
    "D": ["New Zealand", "South Africa", "Afghanistan", "Canada", "United Arab Emirates"],
}

# Match-by-match fixtures with dates & venues
GROUP_FIXTURES = {
    "A": [
        ("Feb 7", "Pakistan", "Netherlands", "SSC, Colombo"),
        ("Feb 7", "India", "United States of America", "Mumbai"),
        ("Feb 10", "Netherlands", "Namibia", "Delhi"),
        ("Feb 10", "Pakistan", "United States of America", "SSC, Colombo"),
        ("Feb 12", "India", "Namibia", "Delhi"),
        ("Feb 13", "United States of America", "Netherlands", "Chennai"),
        ("Feb 15", "India", "Pakistan", "Premadasa, Colombo"),
        ("Feb 15", "United States of America", "Namibia", "Chennai"),
        ("Feb 18", "Pakistan", "Namibia", "SSC, Colombo"),
        ("Feb 18", "India", "Netherlands", "Ahmedabad"),
    ],
    "B": [
        ("Feb 8", "Sri Lanka", "Ireland", "Premadasa, Colombo"),
        ("Feb 9", "Zimbabwe", "Oman", "SSC, Colombo"),
        ("Feb 11", "Australia", "Ireland", "Premadasa, Colombo"),
        ("Feb 12", "Sri Lanka", "Oman", "Kandy"),
        ("Feb 13", "Australia", "Zimbabwe", "Premadasa, Colombo"),
        ("Feb 14", "Ireland", "Oman", "SSC, Colombo"),
        ("Feb 16", "Australia", "Sri Lanka", "Kandy"),
        ("Feb 17", "Ireland", "Zimbabwe", "Kandy"),
        ("Feb 19", "Sri Lanka", "Zimbabwe", "Premadasa, Colombo"),
        ("Feb 20", "Australia", "Oman", "Kandy"),
    ],
    "C": [
        ("Feb 7", "West Indies", "Scotland", "Kolkata"),
        ("Feb 8", "England", "Nepal", "Mumbai"),
        ("Feb 9", "Scotland", "Italy", "Kolkata"),
        ("Feb 11", "England", "West Indies", "Mumbai"),
        ("Feb 12", "Nepal", "Italy", "Mumbai"),
        ("Feb 14", "England", "Scotland", "Kolkata"),
        ("Feb 15", "West Indies", "Nepal", "Mumbai"),
        ("Feb 16", "England", "Italy", "Kolkata"),
        ("Feb 17", "Scotland", "Nepal", "Mumbai"),
        ("Feb 19", "West Indies", "Italy", "Kolkata"),
    ],
    "D": [
        ("Feb 8", "New Zealand", "Afghanistan", "Chennai"),
        ("Feb 9", "South Africa", "Canada", "Ahmedabad"),
        ("Feb 10", "New Zealand", "United Arab Emirates", "Chennai"),
        ("Feb 11", "South Africa", "Afghanistan", "Ahmedabad"),
        ("Feb 13", "Canada", "United Arab Emirates", "Delhi"),
        ("Feb 14", "New Zealand", "South Africa", "Ahmedabad"),
        ("Feb 16", "Afghanistan", "United Arab Emirates", "Delhi"),
        ("Feb 17", "New Zealand", "Canada", "Chennai"),
        ("Feb 18", "South Africa", "United Arab Emirates", "Delhi"),
        ("Feb 19", "Afghanistan", "Canada", "Chennai"),
    ],
}

# ICC Pre-seeded Super 8 slots
# X group: India (X1), Australia (X2), West Indies (X3), South Africa (X4)
# Y group: England (Y1), New Zealand (Y2), Pakistan (Y3), Sri Lanka (Y4)
SUPER8_SEEDS = {
    "India": "X1", "Australia": "X2", "West Indies": "X3", "South Africa": "X4",
    "England": "Y1", "New Zealand": "Y2", "Pakistan": "Y3", "Sri Lanka": "Y4",
}

SUPER8_FIXTURE_TEMPLATE = [
    ("Feb 21", "Y2", "Y3", "Premadasa, Colombo"),
    ("Feb 22", "Y1", "Y4", "Kandy"),
    ("Feb 22", "X1", "X4", "Ahmedabad"),
    ("Feb 23", "X2", "X3", "Mumbai"),
    ("Feb 24", "Y1", "Y3", "Kandy"),
    ("Feb 25", "Y2", "Y4", "Premadasa, Colombo"),
    ("Feb 26", "X3", "X4", "Ahmedabad"),
    ("Feb 26", "X1", "X2", "Chennai"),
    ("Feb 27", "Y1", "Y2", "Premadasa, Colombo"),
    ("Feb 28", "Y3", "Y4", "Kandy"),
    ("Mar 1", "X2", "X4", "Delhi"),
    ("Mar 1", "X1", "X3", "Kolkata"),
]

print("World Cup Team Elo Ratings:")
wc_teams_elo = []
for group, teams in WC_GROUPS.items():
    for team in teams:
        wc_teams_elo.append((team, final_elo[team], group))
wc_teams_elo.sort(key=lambda x: x[1], reverse=True)
for team, rating, group in wc_teams_elo:
    bar = "█" * max(1, int((rating - 800) / 15))
    print(f"  {team:25s}  Elo {rating:7.1f}  (Group {group})  {bar}")
print()

print("──── GROUP STAGE PREDICTIONS (Match by Match) ────")
print()

import json

group_points = defaultdict(lambda: defaultdict(int))
group_wins = defaultdict(lambda: defaultdict(int))
all_group_matches = {}

for group in ["A", "B", "C", "D"]:
    teams = WC_GROUPS[group]
    fixtures = GROUP_FIXTURES[group]
    all_group_matches[group] = []

    # Ensure all teams start at 0
    for t in teams:
        group_points[group][t] = 0

    print(f"╔════════════════════════════════════════════════════════╗")
    print(f"║  GROUP {group}                                                ║")
    print(f"╠════════════════════════════════════════════════════════╣")

    for date, t1, t2, venue in fixtures:
        winner, t1_prob, t2_prob = predict_match(t1, t2)
        win_prob = max(t1_prob, t2_prob)

        group_points[group][winner] += 2
        group_wins[group][winner] += 1

        all_group_matches[group].append({
            "date": date, "team1": t1, "team2": t2, "venue": venue,
            "winner": winner, "t1_prob": round(t1_prob * 100, 1), "t2_prob": round(t2_prob * 100, 1),
        })

        print(f"║  {date:6s}  {t1:20s} vs {t2:20s}  ║")
        print(f"║          → {winner} ({win_prob:.1%}){' ' * max(0, 40 - len(winner) - 8)}  ║")

    print(f"╠════════════════════════════════════════════════════════╣")
    standings = sorted(
        group_points[group].items(),
        key=lambda x: (x[1], group_wins[group].get(x[0], 0)),
        reverse=True,
    )
    for rank, (team, pts) in enumerate(standings, 1):
        wins = group_wins[group].get(team, 0)
        marker = " ★" if rank <= 2 else ""
        print(f"║  {rank}. {team:22s}  {pts} pts  ({wins}W){marker:5s}           ║")
    print(f"╚════════════════════════════════════════════════════════╝")
    print()

# ── SUPER 8s with ICC pre-seeded X/Y system ──
print("──── SUPER 8 STAGE (ICC Pre-Seeded X/Y System) ────")
print()

qualifiers = {}
for group in ["A", "B", "C", "D"]:
    standings = sorted(
        group_points[group].items(),
        key=lambda x: (x[1], group_wins[group].get(x[0], 0)),
        reverse=True,
    )
    qualifiers[group] = [standings[0][0], standings[1][0]]
    print(f"Group {group} qualifiers: {qualifiers[group][0]}, {qualifiers[group][1]}")
print()

all_qualifiers = []
for g in ["A", "B", "C", "D"]:
    all_qualifiers.extend(qualifiers[g])

# Assign pre-seeded teams to their fixed slots
slot_to_team = {}
for team in all_qualifiers:
    if team in SUPER8_SEEDS:
        slot_to_team[SUPER8_SEEDS[team]] = team

# Fill empty slots with non-seeded qualifiers
x_empty = [s for s in ["X1", "X2", "X3", "X4"] if s not in slot_to_team]
y_empty = [s for s in ["Y1", "Y2", "Y3", "Y4"] if s not in slot_to_team]
unassigned = [t for t in all_qualifiers if t not in slot_to_team.values()]
for slot in x_empty:
    if unassigned:
        slot_to_team[slot] = unassigned.pop(0)
for slot in y_empty:
    if unassigned:
        slot_to_team[slot] = unassigned.pop(0)

x_teams = [slot_to_team.get(f"X{i}", "TBD") for i in range(1, 5)]
y_teams = [slot_to_team.get(f"Y{i}", "TBD") for i in range(1, 5)]

print(f"Super 8 Group X: {x_teams}")
print(f"  Slots: " + ", ".join(f"{s}={slot_to_team[s]}" for s in ["X1", "X2", "X3", "X4"]))
print(f"Super 8 Group Y: {y_teams}")
print(f"  Slots: " + ", ".join(f"{s}={slot_to_team[s]}" for s in ["Y1", "Y2", "Y3", "Y4"]))
print()

s8_points = defaultdict(lambda: defaultdict(int))
s8_wins_track = defaultdict(lambda: defaultdict(int))
s8_all_matches = {"Super 8 Group X": [], "Super 8 Group Y": []}

for s8_name, s8_teams, prefix in [("Super 8 Group X", x_teams, "X"), ("Super 8 Group Y", y_teams, "Y")]:
    for t in s8_teams:
        s8_points[s8_name][t] = 0

    print(f"╔════════════════════════════════════════════════════════╗")
    print(f"║  {s8_name:53s} ║")
    print(f"╠════════════════════════════════════════════════════════╣")

    for date, slot1, slot2, venue in SUPER8_FIXTURE_TEMPLATE:
        if not slot1.startswith(prefix):
            continue
        t1 = slot_to_team[slot1]
        t2 = slot_to_team[slot2]
        winner, t1_prob, t2_prob = predict_match(t1, t2)
        win_prob = max(t1_prob, t2_prob)

        s8_points[s8_name][winner] += 2
        s8_wins_track[s8_name][winner] += 1

        s8_all_matches[s8_name].append({
            "date": date, "team1": t1, "team2": t2, "venue": venue,
            "winner": winner, "t1_prob": round(t1_prob * 100, 1), "t2_prob": round(t2_prob * 100, 1),
        })

        print(f"║  {date:6s}  {t1:20s} vs {t2:20s}  ║")
        print(f"║          → {winner} ({win_prob:.1%}){' ' * max(0, 40 - len(winner) - 8)}  ║")

    print(f"╠════════════════════════════════════════════════════════╣")
    standings = sorted(
        s8_points[s8_name].items(),
        key=lambda x: (x[1], s8_wins_track[s8_name].get(x[0], 0)),
        reverse=True,
    )
    for rank, (team, pts) in enumerate(standings, 1):
        wins = s8_wins_track[s8_name].get(team, 0)
        marker = " ★" if rank <= 2 else ""
        print(f"║  {rank}. {team:22s}  {pts} pts  ({wins}W){marker:5s}           ║")
    print(f"╚════════════════════════════════════════════════════════╝")
    print()

# Semi-final matchups: X1st vs Y2nd, Y1st vs X2nd
s8_x_standings = sorted(s8_points["Super 8 Group X"].items(),
    key=lambda x: (x[1], s8_wins_track["Super 8 Group X"].get(x[0], 0)), reverse=True)
s8_y_standings = sorted(s8_points["Super 8 Group Y"].items(),
    key=lambda x: (x[1], s8_wins_track["Super 8 Group Y"].get(x[0], 0)), reverse=True)

sf1_t1 = s8_x_standings[0][0]
sf1_t2 = s8_y_standings[1][0]
sf2_t1 = s8_y_standings[0][0]
sf2_t2 = s8_x_standings[1][0]

print("──── SEMI-FINALS ────")
print()

sf1_winner, sf1_p1, sf1_p2 = predict_match(sf1_t1, sf1_t2)
sf2_winner, sf2_p1, sf2_p2 = predict_match(sf2_t1, sf2_t2)

print(f"Semi-Final 1 (Mar 4): {sf1_t1} vs {sf1_t2}")
print(f"  → Winner: {sf1_winner} ({max(sf1_p1, sf1_p2):.1%})")
print()
print(f"Semi-Final 2 (Mar 5, Mumbai): {sf2_t1} vs {sf2_t2}")
print(f"  → Winner: {sf2_winner} ({max(sf2_p1, sf2_p2):.1%})")
print()

print("══════════════════════════════════════════════")
print("              FINAL (Mar 8)")
print("══════════════════════════════════════════════")
print()

final_winner, f_p1, f_p2 = predict_match(sf1_winner, sf2_winner)
print(f"  {sf1_winner} vs {sf2_winner}")
print()
print(f"  {sf1_winner}: {f_p1:.1%}")
print(f"  {sf2_winner}: {f_p2:.1%}")
print()
print(f"  🏆 PREDICTED CHAMPION: {final_winner}")
print()
print("══════════════════════════════════════════════")

# ── Export JSON for visualiser ──
print()
print("Step 41: Exporting data for visualiser...")

groups_json = {}
for group in ["A", "B", "C", "D"]:
    standings = sorted(
        group_points[group].items(),
        key=lambda x: (x[1], group_wins[group].get(x[0], 0)),
        reverse=True,
    )
    groups_json[group] = {
        "standings": [{"name": team, "pts": pts, "wins": group_wins[group].get(team, 0), "q": rank <= 2}
                      for rank, (team, pts) in enumerate(standings, 1)],
        "matches": all_group_matches[group],
    }

s8_json = []
for s8_name in ["Super 8 Group X", "Super 8 Group Y"]:
    standings = sorted(
        s8_points[s8_name].items(),
        key=lambda x: (x[1], s8_wins_track[s8_name].get(x[0], 0)),
        reverse=True,
    )
    s8_json.append({
        "title": s8_name,
        "standings": [{"name": team, "pts": pts, "wins": s8_wins_track[s8_name].get(team, 0), "q": rank <= 2}
                      for rank, (team, pts) in enumerate(standings, 1)],
        "matches": s8_all_matches[s8_name],
    })

knockout_json = {
    "sf1": {"date": "Mar 4", "venue": "TBD", "t1": sf1_t1, "t2": sf1_t2, "w": sf1_winner,
            "p1": round(sf1_p1 * 100, 1), "p2": round(sf1_p2 * 100, 1)},
    "sf2": {"date": "Mar 5", "venue": "Mumbai", "t1": sf2_t1, "t2": sf2_t2, "w": sf2_winner,
            "p1": round(sf2_p1 * 100, 1), "p2": round(sf2_p2 * 100, 1)},
    "final": {"date": "Mar 8", "venue": "Ahmedabad / Colombo", "t1": sf1_winner, "t2": sf2_winner,
              "w": final_winner, "p1": round(f_p1 * 100, 1), "p2": round(f_p2 * 100, 1)},
}

with open("data/predictions.json", "w") as f:
    json.dump({"groups": groups_json, "super8": s8_json, "knockouts": knockout_json}, f, indent=2)

print("  ✓ Exported to data/predictions.json")
print("Done!")
