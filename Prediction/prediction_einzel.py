import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

###############################################################################
# 1) LOAD DATA & CLEAN
###############################################################################
league = "E0"  # Ändere hier die Liga: "D1", "F1", "I1", "E0", "SP1"
DATA_DIR = "data"

training_files = [
    os.path.join(DATA_DIR, "2024_25", f"{league}.csv"),
    os.path.join(DATA_DIR, "2023_24", f"{league}.csv"),
    os.path.join(DATA_DIR, "2022_23", f"{league}.csv"),
    os.path.join(DATA_DIR, "2021_22", f"{league}.csv"),
    os.path.join(DATA_DIR, "2020_21", f"{league}.csv"),
    os.path.join(DATA_DIR, "2019_20", f"{league}.csv"),
    os.path.join(DATA_DIR, "2018_19", f"{league}.csv"),
    os.path.join(DATA_DIR, "2017_18", f"{league}.csv"),
]

df_list = [pd.read_csv(f) for f in training_files if os.path.exists(f)]
if not df_list:
    raise ValueError(f"No historical CSVs found for league={league}!")

df_hist = pd.concat(df_list, ignore_index=True)

# Keep only valid match results
df_hist = df_hist[df_hist["FTR"].isin(["H", "D", "A"])].copy()
df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%d/%m/%Y", errors="coerce")
df_hist.dropna(subset=["Date"], inplace=True)
df_hist.sort_values("Date", inplace=True)
df_hist.reset_index(drop=True, inplace=True)

# Encode FTR => 0=Home,1=Draw,2=Away
ftr_map = {"H": 0, "D": 1, "A": 2}
df_hist["FTR"] = df_hist["FTR"].map(ftr_map)

# Encode teams
df_hist["HomeTeam"] = df_hist["HomeTeam"].str.strip().str.lower()
df_hist["AwayTeam"] = df_hist["AwayTeam"].str.strip().str.lower()
team_encoder = LabelEncoder()
all_teams = pd.concat([df_hist["HomeTeam"], df_hist["AwayTeam"]]).unique()
team_encoder.fit(all_teams)
df_hist["HomeTeam"] = team_encoder.transform(df_hist["HomeTeam"])
df_hist["AwayTeam"] = team_encoder.transform(df_hist["AwayTeam"])

###############################################################################
# 2) FEATURE ENGINEERING
###############################################################################

# Create points for home & away teams
df_hist["HomePoints"] = df_hist["FTR"].apply(lambda x: 3 if x == 0 else (1 if x == 1 else 0))
df_hist["AwayPoints"] = df_hist["FTR"].apply(lambda x: 3 if x == 2 else (1 if x == 1 else 0))

# Expected Goals (xG, xGA)
df_hist["xG"] = df_hist["HST"].rolling(5).mean() * (df_hist["FTHG"] / df_hist["HST"]).rolling(10).mean()
df_hist["xGA"] = df_hist["AST"].rolling(5).mean() * (df_hist["FTAG"] / df_hist["AST"]).rolling(10).mean()

# Rolling recent form
def rolling_stats(df, prefix):
    df = df.sort_values("Date", ascending=False)
    df[f"{prefix}RecentPoints5"] = df["HomePoints"].shift(1).rolling(5, min_periods=1).sum()
    df[f"{prefix}RecentGF5"] = df["FTHG"].shift(1).rolling(5, min_periods=1).sum()
    df[f"{prefix}RecentGA5"] = df["FTAG"].shift(1).rolling(5, min_periods=1).sum()
    df[f"{prefix}WinRate5"] = df["HomePoints"].shift(1).rolling(5, min_periods=1).mean() / 3
    return df.sort_values("Date")

df_hist = df_hist.groupby("HomeTeam", group_keys=False).apply(lambda x: rolling_stats(x, "Home"))
df_hist = df_hist.groupby("AwayTeam", group_keys=False).apply(lambda x: rolling_stats(x, "Away"))

# Opponent Strength
df_hist["OpponentStrength"] = df_hist.groupby("AwayTeam")["HomeRecentPoints5"].transform(lambda x: x.shift(1).rolling(5).mean())
df_hist["OpponentStrength"] += df_hist.groupby("HomeTeam")["AwayRecentPoints5"].transform(lambda x: x.shift(1).rolling(5).mean())
df_hist["OpponentStrength"] /= 2  

# Home/Away Momentum
df_hist["HomeMomentum"] = df_hist.groupby("HomeTeam")["HomePoints"].transform(lambda x: x.rolling(5).sum())
df_hist["AwayMomentum"] = df_hist.groupby("AwayTeam")["AwayPoints"].transform(lambda x: x.rolling(5).sum())

###############################################################################
# 3) TRAIN MODEL ON FULL DATA
###############################################################################
features = [
    "HomeTeam", "AwayTeam", "xG", "xGA", "OpponentStrength",
    "HomeRecentPoints5", "HomeRecentGF5", "HomeRecentGA5", "HomeWinRate5",
    "AwayRecentPoints5", "AwayRecentGF5", "AwayRecentGA5", "AwayWinRate5",
    "HomeMomentum", "AwayMomentum"
]
target = "FTR"

X_train, y_train = df_hist[features], df_hist[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

###############################################################################
# 4) TRAIN RANDOM FOREST MODEL
###############################################################################
model = RandomForestClassifier(
    n_estimators=300,  
    max_depth=6,  
    random_state=42,
    class_weight={0: 1.0, 1: 1.5, 2: 1.0}
)

model.fit(X_train_scaled, y_train)

###############################################################################
# 5) PREDICT UPCOMING FIXTURES FROM FIXTURES.CSV
###############################################################################

# Load fixtures
fixtures_path = os.path.join(DATA_DIR, "fixtures", "fixtures.csv")
df_fixtures = pd.read_csv(fixtures_path)

# Filter fixtures for the trained league
df_fixtures = df_fixtures[df_fixtures["Div"] == league].copy()

# Encode teams
df_fixtures["HomeTeam"] = df_fixtures["HomeTeam"].str.strip().str.lower()
df_fixtures["AwayTeam"] = df_fixtures["AwayTeam"].str.strip().str.lower()

df_fixtures["HomeTeam"] = df_fixtures["HomeTeam"].apply(lambda t: team_encoder.transform([t])[0] if t in team_encoder.classes_ else -1)
df_fixtures["AwayTeam"] = df_fixtures["AwayTeam"].apply(lambda t: team_encoder.transform([t])[0] if t in team_encoder.classes_ else -1)

# Ensure all feature columns exist
for col in features:
    if col not in df_fixtures.columns:
        df_fixtures[col] = 0.0

X_fixtures_scaled = scaler.transform(df_fixtures[features])

# Predict probabilities
y_pred_probs = model.predict_proba(X_fixtures_scaled)
y_pred_fixtures = model.predict(X_fixtures_scaled)

df_fixtures["Predicted Result"] = y_pred_fixtures
df_fixtures["Prob_HomeWin"] = y_pred_probs[:, 0]
df_fixtures["Prob_Draw"] = y_pred_probs[:, 1]
df_fixtures["Prob_AwayWin"] = y_pred_probs[:, 2]

# Decode Team Names
df_fixtures["HomeTeam"] = df_fixtures["HomeTeam"].apply(lambda x: team_encoder.inverse_transform([x])[0])
df_fixtures["AwayTeam"] = df_fixtures["AwayTeam"].apply(lambda x: team_encoder.inverse_transform([x])[0])

print("\n=== Upcoming Fixtures Predictions ===")
print(df_fixtures[["HomeTeam", "AwayTeam", "Predicted Result", "Prob_HomeWin", "Prob_Draw", "Prob_AwayWin"]])
