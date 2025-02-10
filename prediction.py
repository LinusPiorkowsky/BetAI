import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

###############################################################################
# 1) LOAD MULTIPLE CSVs FOR HISTORICAL D1
###############################################################################
# league can be "D1", "E0", "F1", "I1", "SP1", etc.
league = "SP1"

DATA_DIR = "data"
training_files = [
    os.path.join(DATA_DIR, "2024_25", f"{league}.csv"),
    os.path.join(DATA_DIR, "2023_24", f"{league}.csv"),
    os.path.join(DATA_DIR, "2022_23", f"{league}.csv"),
    os.path.join(DATA_DIR, "2021_22", f"{league}.csv"),
]

df_list = []
for f in training_files:
    if os.path.exists(f):
        tmp = pd.read_csv(f)
        df_list.append(tmp)
        print(f"Loaded {f}, shape={tmp.shape}")
    else:
        print(f"WARNING: file not found: {f}")

if not df_list:
    raise ValueError("No historical CSVs found!")

df_hist = pd.concat(df_list, ignore_index=True)
print("Combined historical shape:", df_hist.shape)

# Keep only rows with known FTR in [H, D, A]
df_hist = df_hist[df_hist["FTR"].isin(["H","D","A"])].copy()

###############################################################################
# 2) CLEAN & PARSE THE DATE (for recency weighting)
###############################################################################
df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%d/%m/%Y", errors="coerce")
df_hist = df_hist.dropna(subset=["Date"])  # remove rows with invalid date
df_hist = df_hist.loc[:, ~df_hist.columns.str.contains("Unnamed")]

# Sort by date ascending
df_hist = df_hist.sort_values("Date").reset_index(drop=True)

# Encode FTR => 0=Home, 1=Draw, 2=Away
ftr_map = {"H":0, "D":1, "A":2}
df_hist["FTR"] = df_hist["FTR"].map(ftr_map)

# Standardize team names & label-encode
df_hist["HomeTeam"] = df_hist["HomeTeam"].str.strip().str.lower()
df_hist["AwayTeam"] = df_hist["AwayTeam"].str.strip().str.lower()

team_encoder = LabelEncoder()
all_teams = pd.concat([df_hist["HomeTeam"], df_hist["AwayTeam"]]).unique()
team_encoder.fit(all_teams)

df_hist["HomeTeam"] = team_encoder.transform(df_hist["HomeTeam"])
df_hist["AwayTeam"] = team_encoder.transform(df_hist["AwayTeam"])

###############################################################################
# (A) CREATE HOME & AWAY POINTS COLUMNS FOR ROLLING
###############################################################################
# We'll assume your CSV has FTHG (full-time home goals), FTAG (away goals).
# If not, adjust accordingly.

def home_points_func(ftr):
    # ftr=0 => home, 1=>draw, 2=>away
    if pd.isna(ftr):
        return np.nan
    return 3 if ftr==0 else (1 if ftr==1 else 0)

def away_points_func(ftr):
    if pd.isna(ftr):
        return np.nan
    return 3 if ftr==2 else (1 if ftr==1 else 0)

df_hist["HomePoints"] = df_hist["FTR"].apply(home_points_func)
df_hist["AwayPoints"] = df_hist["FTR"].apply(away_points_func)

# If FTHG, FTAG missing, fill with 0
for c in ["FTHG","FTAG"]:
    if c not in df_hist.columns:
        df_hist[c] = 0

###############################################################################
# (B) ROLLING “RECENT FORM” FOR HOME & AWAY
###############################################################################
def rolling_home(grp):
    grp = grp.sort_values("Date")
    # shift(1) => exclude current match
    grp["HomeRecentPoints5"] = grp["HomePoints"].shift(1).rolling(5, min_periods=1).sum()
    grp["HomeRecentGF5"] = grp["FTHG"].shift(1).rolling(5, min_periods=1).sum()
    grp["HomeRecentGA5"] = grp["FTAG"].shift(1).rolling(5, min_periods=1).sum()
    return grp

df_hist = df_hist.groupby("HomeTeam", group_keys=False).apply(rolling_home)

def rolling_away(grp):
    grp = grp.sort_values("Date")
    grp["AwayRecentPoints5"] = grp["AwayPoints"].shift(1).rolling(5, min_periods=1).sum()
    grp["AwayRecentGF5"] = grp["FTAG"].shift(1).rolling(5, min_periods=1).sum()
    grp["AwayRecentGA5"] = grp["FTHG"].shift(1).rolling(5, min_periods=1).sum()
    return grp

df_hist = df_hist.groupby("AwayTeam", group_keys=False).apply(rolling_away)

###############################################################################
# (C) HEAD-TO-HEAD ROLLING
###############################################################################
df_hist["MatchUp"] = df_hist.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]), axis=1)

def rolling_h2h(grp):
    grp = grp.sort_values("Date")
    grp["H2HHomePoints5"] = grp["HomePoints"].shift(1).rolling(5, min_periods=1).sum()
    grp["H2HHomeGF5"] = grp["FTHG"].shift(1).rolling(5, min_periods=1).sum()
    grp["H2HHomeGA5"] = grp["FTAG"].shift(1).rolling(5, min_periods=1).sum()
    return grp

df_hist = df_hist.groupby("MatchUp", group_keys=False).apply(rolling_h2h)

# fill any NaNs from rolling
for c in [
    "HomeRecentPoints5","HomeRecentGF5","HomeRecentGA5",
    "AwayRecentPoints5","AwayRecentGF5","AwayRecentGA5",
    "H2HHomePoints5","H2HHomeGF5","H2HHomeGA5"
]:
    if c not in df_hist.columns:
        df_hist[c] = 0
    df_hist[c] = df_hist[c].fillna(0.0)

###############################################################################
# 3) DEFINE FEATURES + ROLLING COLUMNS & TIME-BASED WEIGHTS
###############################################################################
# Now we add the new rolling columns to your existing 'features'
features = [
    "HomeTeam","AwayTeam",
    "HS","AS","HST","AST",
    "AvgH","AvgD","AvgA",
    # Rolling form
    "HomeRecentPoints5","HomeRecentGF5","HomeRecentGA5",
    "AwayRecentPoints5","AwayRecentGF5","AwayRecentGA5",
    # Head-to-head rolling
    "H2HHomePoints5","H2HHomeGF5","H2HHomeGA5"
]
target = "FTR"

# Recency weighting
max_date = df_hist["Date"].max()
df_hist["DaysDiff"] = (max_date - df_hist["Date"]).dt.days
df_hist["RecencyWeight"] = np.power(0.99, df_hist["DaysDiff"] / 7.0)

# fill missing
for col in features:
    if col not in df_hist.columns:
        df_hist[col] = 0.0
df_hist[features] = df_hist[features].fillna(0.0)

X_train = df_hist[features].copy()
y_train = df_hist[target].astype(int).copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

###############################################################################
# 4) TRAIN LOGISTIC REGRESSION (WITH sample_weight)
###############################################################################
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    random_state=42,
    class_weight={
        0:1.0, # home
        1:0.95, # draw
        2:1.0  # away
    }
)

model.fit(X_train_scaled, y_train, sample_weight=df_hist["RecencyWeight"])
print(f"Logistic Regression training done for league={league}. Used {len(df_hist)} matches with recency weighting.")

###############################################################################
# 5) LOAD UPCOMING FIXTURES AND PREDICT
###############################################################################
FIXTURES_PATH = os.path.join(DATA_DIR, "fixtures", "fixtures.csv")
if not os.path.exists(FIXTURES_PATH):
    print(f"No fixtures file found at {FIXTURES_PATH}. Exiting.")
    exit()

df_fix = pd.read_csv(FIXTURES_PATH)
print("Fixtures shape (raw):", df_fix.shape)

# Keep only this league (the user-chosen one)
df_fix = df_fix[df_fix["Div"]==league].reset_index(drop=True)
if df_fix.empty:
    print(f"No {league} fixtures found in fixtures.csv. Done.")
    exit()

df_fix["Date"] = pd.to_datetime(df_fix["Date"], format="%d/%m/%Y", errors="coerce")
df_fix["HomeTeam"] = df_fix["HomeTeam"].str.strip().str.lower()
df_fix["AwayTeam"] = df_fix["AwayTeam"].str.strip().str.lower()

for col in features:
    if col not in df_fix.columns:
        df_fix[col] = np.nan
df_fix[features] = df_fix[features].fillna(0.0)

def safe_team_enc(t):
    if t not in team_encoder.classes_:
        print(f"WARNING: new team '{t}' not in training data. Adding it.")
        team_encoder.classes_ = np.append(team_encoder.classes_, t)
    return team_encoder.transform([t])[0]

df_fix["HomeTeam"] = df_fix["HomeTeam"].apply(safe_team_enc)
df_fix["AwayTeam"] = df_fix["AwayTeam"].apply(safe_team_enc)

X_fix = df_fix[features].copy()
X_fix_scaled = scaler.transform(X_fix)

probs = model.predict_proba(X_fix_scaled)  # shape (N,3)
preds = model.predict(X_fix_scaled)

reverse_ftr = {0:"H",1:"D",2:"A"}

# Build a list of dicts
predictions_list = []
for i, row in df_fix.iterrows():
    ph, pd, pa = probs[i]
    pred_label = preds[i]
    label_str = reverse_ftr[pred_label]

    # decode numeric IDs => original team names
    home_id = row["HomeTeam"]
    away_id = row["AwayTeam"]
    home_str = team_encoder.inverse_transform([int(home_id)])[0]
    away_str = team_encoder.inverse_transform([int(away_id)])[0]

    # convert date
    raw_date = row["Date"]
    date_str = raw_date.strftime("%Y-%m-%d")  # e.g. '2025-02-07 00:00:00' or 'NaT'
    if "nat" in date_str.lower() or "nan" in date_str.lower():
        date_str = "Unknown"

    predictions_list.append({
        "Date": date_str,
        "HomeTeam": home_str,
        "AwayTeam": away_str,
        "Prediction": label_str,
        "Prob_H": round(ph,3),
        "Prob_D": round(pd,3),
        "Prob_A": round(pa,3),
    })

print("Type of predictions_list:", type(predictions_list))
if predictions_list:
    print("Sample item in predictions_list:", predictions_list[0])

from pandas.core.frame import DataFrame as RealDataFrame

df_preds = RealDataFrame(predictions_list)
print("\n=== UPCOMING FIXTURES WITH RECENCY-WEIGHTED MODEL ===")
print(df_preds)
