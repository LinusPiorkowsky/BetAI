import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

###############################################################################
# 1) LOAD HISTORICAL DATA FROM MULTIPLE LEAGUES
###############################################################################
DATA_DIR = "data"
PREDICTION_DIR = "data/predictions"
FIXTURES_FILE = os.path.join(DATA_DIR, "fixtures", "fixtures.csv")
LEAGUES = ["D1", "E0", "F1", "I1", "SP1"]

os.makedirs(PREDICTION_DIR, exist_ok=True)

df_list = []
for league in LEAGUES:
    training_files = [os.path.join(DATA_DIR, season, f"{league}.csv") for season in [
        "2024_25", "2023_24", "2022_23", "2021_22", "2020_21", "2019_20", "2018_19", "2017_18"
    ]]
    
    for f in training_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(f, encoding="ISO-8859-1")
            df["League"] = league  # Track league
            df_list.append(df)

if not df_list:
    raise ValueError("No historical data found!")

df_hist = pd.concat(df_list, ignore_index=True)

# Keep only valid match results
df_hist = df_hist[df_hist["FTR"].isin(["H", "D", "A"])].copy()
df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%d/%m/%Y", errors="coerce")
df_hist.dropna(subset=["Date"], inplace=True)
df_hist.sort_values("Date", inplace=True)
df_hist.reset_index(drop=True, inplace=True)

# Encode FTR => 0=Home, 1=Draw, 2=Away
ftr_map = {"H": 0, "D": 1, "A": 2}
df_hist["FTR"] = df_hist["FTR"].map(ftr_map)

###############################################################################
# 2) ENCODE TEAMS ACROSS LEAGUES
###############################################################################
df_hist["HomeTeam"] = df_hist["HomeTeam"].str.strip().str.lower()
df_hist["AwayTeam"] = df_hist["AwayTeam"].str.strip().str.lower()

team_encoder = LabelEncoder()
all_teams = pd.concat([df_hist["HomeTeam"], df_hist["AwayTeam"]]).unique()
team_encoder.fit(all_teams)

df_hist["HomeTeamEncoded"] = team_encoder.transform(df_hist["HomeTeam"])
df_hist["AwayTeamEncoded"] = team_encoder.transform(df_hist["AwayTeam"])

###############################################################################
# 3) FEATURE ENGINEERING
###############################################################################
df_hist["HomePoints"] = df_hist["FTR"].apply(lambda x: 3 if x == 0 else (1 if x == 1 else 0))
df_hist["AwayPoints"] = df_hist["FTR"].apply(lambda x: 3 if x == 2 else (1 if x == 1 else 0))

# Expected Goals (xG, xGA)
df_hist["xG"] = df_hist["HST"].rolling(5).mean() * (df_hist["FTHG"] / df_hist["HST"]).rolling(5).mean()
df_hist["xGA"] = df_hist["AST"].rolling(5).mean() * (df_hist["FTAG"] / df_hist["AST"]).rolling(5).mean()

# Rolling recent form
def rolling_stats(df, prefix):
    df = df.sort_values("Date", ascending=False)
    df[f"{prefix}RecentPoints5"] = df["HomePoints"].shift(1).rolling(5, min_periods=1).sum()
    df[f"{prefix}RecentGF5"] = df["FTHG"].shift(1).rolling(5, min_periods=1).sum()
    df[f"{prefix}RecentGA5"] = df["FTAG"].shift(1).rolling(5, min_periods=1).sum()
    df[f"{prefix}WinRate5"] = df["HomePoints"].shift(1).rolling(5, min_periods=1).mean() / 3
    return df.sort_values("Date")

df_hist = df_hist.groupby("HomeTeamEncoded", group_keys=False).apply(lambda x: rolling_stats(x, "Home"))
df_hist = df_hist.groupby("AwayTeamEncoded", group_keys=False).apply(lambda x: rolling_stats(x, "Away"))

# Opponent Strength
df_hist["OpponentStrength"] = df_hist.groupby("AwayTeamEncoded")["HomeRecentPoints5"].transform(lambda x: x.shift(1).rolling(5).mean())
df_hist["OpponentStrength"] += df_hist.groupby("HomeTeamEncoded")["AwayRecentPoints5"].transform(lambda x: x.shift(1).rolling(5).mean())
df_hist["OpponentStrength"] /= 2  

# Home/Away Momentum
df_hist["HomeMomentum"] = df_hist.groupby("HomeTeamEncoded")["HomePoints"].transform(lambda x: x.rolling(5).sum())
df_hist["AwayMomentum"] = df_hist.groupby("AwayTeamEncoded")["AwayPoints"].transform(lambda x: x.rolling(5).sum())

###############################################################################
# 4) TRAIN MODEL ON FULL DATA
###############################################################################
features = [
    "HomeTeamEncoded", "AwayTeamEncoded", "xG", "xGA", "OpponentStrength",
    "HomeRecentPoints5", "HomeRecentGF5", "HomeRecentGA5", "HomeWinRate5",
    "AwayRecentPoints5", "AwayRecentGF5", "AwayRecentGA5", "AwayWinRate5",
    "HomeMomentum", "AwayMomentum"
]
target = "FTR"

X_train, y_train = df_hist[features], df_hist[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Model
model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight={0: 1.0, 1: 1.5, 2: 1.0})
model.fit(X_train_scaled, y_train)

###############################################################################
# 5) PREDICT UPCOMING FIXTURES
###############################################################################
df_fixtures = pd.read_csv(FIXTURES_FILE)
df_fixtures = df_fixtures[df_fixtures["Div"].isin(LEAGUES)].copy()

# Encode teams
df_fixtures["HomeTeam"] = df_fixtures["HomeTeam"].str.strip().str.lower()
df_fixtures["AwayTeam"] = df_fixtures["AwayTeam"].str.strip().str.lower()

df_fixtures["HomeTeamEncoded"] = df_fixtures["HomeTeam"].apply(lambda t: team_encoder.transform([t])[0] if t in team_encoder.classes_ else -1)
df_fixtures["AwayTeamEncoded"] = df_fixtures["AwayTeam"].apply(lambda t: team_encoder.transform([t])[0] if t in team_encoder.classes_ else -1)

# Ensure feature columns exist
for col in features:
    if col not in df_fixtures.columns:
        df_fixtures[col] = 0.0

X_fixtures_scaled = scaler.transform(df_fixtures[features])
y_pred_probs = model.predict_proba(X_fixtures_scaled)
y_pred_fixtures = model.predict(X_fixtures_scaled)

df_fixtures["Prediction"] = y_pred_fixtures
df_fixtures["Prob_HomeWin"] = y_pred_probs[:, 0]
df_fixtures["Prob_Draw"] = y_pred_probs[:, 1]
df_fixtures["Prob_AwayWin"] = y_pred_probs[:, 2]

# Convert numerical predictions to match result format
df_fixtures["Prediction"] = df_fixtures["Prediction"].apply(lambda x: "H" if x == 0 else ("D" if x == 1 else "A"))

# Decode team names
df_fixtures["HomeTeam"] = team_encoder.inverse_transform(df_fixtures["HomeTeamEncoded"])
df_fixtures["AwayTeam"] = team_encoder.inverse_transform(df_fixtures["AwayTeamEncoded"])

# Select only required columns
df_fixtures = df_fixtures[["Div", "Date", "HomeTeam", "AwayTeam", "Prediction", "Prob_HomeWin", "Prob_Draw", "Prob_AwayWin", "AvgH", "AvgD", "AvgA"]]

# Save predictions
prediction_number = len(os.listdir(PREDICTION_DIR)) + 1
prediction_file = os.path.join(PREDICTION_DIR, f"prediction{prediction_number}.csv")
df_fixtures.to_csv(prediction_file, index=False)

print(f"\n✅ Predictions saved in {prediction_file}")
