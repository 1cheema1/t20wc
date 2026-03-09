
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
    team_history = defaultdict(list)
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
            return sum(w for _, w in recent) / len(recent)

        team1_form.append(_recent_rate(t1))
        team2_form.append(_recent_rate(t2))

        team_history[t1].append((idx, 1 if winner == t1 else 0))
        team_history[t2].append((idx, 1 if winner == t2 else 0))

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

MODEL_WEIGHT = 0.70
ICC_WEIGHT = 0.30

team_recent = defaultdict(list)
for _, row in df.iterrows():
    t1 = row["team1"]
    t2 = row["team2"]
    winner = row["winner"]
    team_recent[t1].append(1 if winner == t1 else 0)
    team_recent[t2].append(1 if winner == t2 else 0)


def get_team_stats(team):
    elo_rating = final_elo[team]
    recent = team_recent[team][-RECENT_N:]
    recent_form = sum(recent) / len(recent) if recent else 0.5
    return elo_rating, recent_form


def icc_win_prob(team1, team2):
    r1 = ICC_RANKINGS.get(team1, 150)
    r2 = ICC_RANKINGS.get(team2, 150)
    return 1.0 / (1.0 + 10 ** ((r2 - r1) / 150))


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
    "A": ["India", "Pakistan", "Netherlands", "Namibia", "United States of America"],
    "B": ["Australia", "Sri Lanka", "Zimbabwe", "Ireland", "Oman"],
    "C": ["England", "West Indies", "Scotland", "Nepal", "Italy"],
    "D": ["South Africa", "New Zealand", "Afghanistan", "United Arab Emirates", "Canada"],
}

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

print("──── GROUP STAGE PREDICTIONS ────")
print()

group_points = defaultdict(lambda: defaultdict(int))
group_wins = defaultdict(lambda: defaultdict(int))

for group, teams in WC_GROUPS.items():
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  GROUP {group}                                     ║")
    print(f"╠══════════════════════════════════════════════╣")

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            winner, t1_prob, t2_prob = predict_match(t1, t2)
            win_prob = max(t1_prob, t2_prob)

            group_points[group][winner] += 2
            group_wins[group][winner] += 1

            for t in [t1, t2]:
                if t not in group_points[group]:
                    group_points[group][t] = 0

            print(f"║  {t1:20s} vs {t2:20s} ║")
            print(f"║     → {winner} ({win_prob:.1%}){' ' * (38 - len(winner) - 8)}║")

    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  STANDINGS                                   ║")

    standings = sorted(
        group_points[group].items(),
        key=lambda x: (x[1], group_wins[group].get(x[0], 0)),
        reverse=True,
    )
    for rank, (team, pts) in enumerate(standings, 1):
        wins = group_wins[group].get(team, 0)
        marker = " ★" if rank <= 2 else ""
        print(f"║  {rank}. {team:20s}  {pts} pts  ({wins}W){marker:5s}       ║")

    print(f"╚══════════════════════════════════════════════╝")
    print()

print("──── SUPER 8 STAGE ────")
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

s8_group1 = [qualifiers["A"][0], qualifiers["B"][1], qualifiers["C"][0], qualifiers["D"][1]]
s8_group2 = [qualifiers["A"][1], qualifiers["B"][0], qualifiers["C"][1], qualifiers["D"][0]]

print(f"Super 8 Group 1: {s8_group1}")
print(f"Super 8 Group 2: {s8_group2}")
print()

s8_points = defaultdict(lambda: defaultdict(int))
s8_wins_track = defaultdict(lambda: defaultdict(int))

for s8_name, s8_teams in [("Super 8 Group 1", s8_group1), ("Super 8 Group 2", s8_group2)]:
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  {s8_name:43s} ║")
    print(f"╠══════════════════════════════════════════════╣")

    for i in range(len(s8_teams)):
        for j in range(i + 1, len(s8_teams)):
            t1, t2 = s8_teams[i], s8_teams[j]
            winner, t1_prob, t2_prob = predict_match(t1, t2)
            win_prob = max(t1_prob, t2_prob)

            s8_points[s8_name][winner] += 2
            s8_wins_track[s8_name][winner] += 1
            for t in [t1, t2]:
                if t not in s8_points[s8_name]:
                    s8_points[s8_name][t] = 0

            print(f"║  {t1:20s} vs {t2:20s} ║")
            print(f"║     → {winner} ({win_prob:.1%}){' ' * (38 - len(winner) - 8)}║")

    print(f"╠══════════════════════════════════════════════╣")
    standings = sorted(
        s8_points[s8_name].items(),
        key=lambda x: (x[1], s8_wins_track[s8_name].get(x[0], 0)),
        reverse=True,
    )
    for rank, (team, pts) in enumerate(standings, 1):
        wins = s8_wins_track[s8_name].get(team, 0)
        marker = " ★" if rank <= 2 else ""
        print(f"║  {rank}. {team:20s}  {pts} pts  ({wins}W){marker:5s}       ║")

    print(f"╚══════════════════════════════════════════════╝")
    print()

s8_1_standings = sorted(
    s8_points["Super 8 Group 1"].items(),
    key=lambda x: (x[1], s8_wins_track["Super 8 Group 1"].get(x[0], 0)),
    reverse=True,
)
s8_2_standings = sorted(
    s8_points["Super 8 Group 2"].items(),
    key=lambda x: (x[1], s8_wins_track["Super 8 Group 2"].get(x[0], 0)),
    reverse=True,
)

sf1_t1 = s8_1_standings[0][0]
sf1_t2 = s8_2_standings[1][0]
sf2_t1 = s8_2_standings[0][0]
sf2_t2 = s8_1_standings[1][0]

print("──── SEMI-FINALS ────")
print()

sf1_winner, sf1_p1, sf1_p2 = predict_match(sf1_t1, sf1_t2)
sf2_winner, sf2_p1, sf2_p2 = predict_match(sf2_t1, sf2_t2)

print(f"Semi-Final 1: {sf1_t1} vs {sf1_t2}")
print(f"  → Winner: {sf1_winner} ({max(sf1_p1, sf1_p2):.1%})")
print()
print(f"Semi-Final 2: {sf2_t1} vs {sf2_t2}")
print(f"  → Winner: {sf2_winner} ({max(sf2_p1, sf2_p2):.1%})")
print()

print("══════════════════════════════════════════════")
print("                   FINAL")
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
