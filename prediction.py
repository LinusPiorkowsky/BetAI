import time
import os
import pandas as pd
import numpy as np
import os
import hashlib
import time
import glob

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

###############################################################################
# 1) CHOOSE LEAGUE + LOAD MULTIPLE CSVs
###############################################################################
league = "F1"
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
    else:
        print(f"WARNING: file not found: {f}")

if not df_list:
    raise ValueError(f"No historical CSVs found for league={league}!")

df_hist = pd.concat(df_list, ignore_index=True)

# Keep only rows with known FTR in [H, D, A]
df_hist = df_hist[df_hist["FTR"].isin(["H","D","A"])].copy()

###############################################################################
# 2) PARSE DATES & CLEAN
###############################################################################
df_hist["Date"] = pd.to_datetime(df_hist["Date"], format="%d/%m/%Y", errors="coerce")
df_hist = df_hist.dropna(subset=["Date"])  # remove invalid date
df_hist = df_hist.loc[:, ~df_hist.columns.str.contains("Unnamed")]
df_hist = df_hist.sort_values("Date").reset_index(drop=True)

# Encode FTR => 0=Home,1=Draw,2=Away
ftr_map = {"H":0,"D":1,"A":2}
df_hist["FTR"] = df_hist["FTR"].map(ftr_map)

# Standardize team names & label encode
df_hist["HomeTeam"] = df_hist["HomeTeam"].str.strip().str.lower()
df_hist["AwayTeam"] = df_hist["AwayTeam"].str.strip().str.lower()

team_encoder = LabelEncoder()
all_teams = pd.concat([df_hist["HomeTeam"], df_hist["AwayTeam"]]).unique()
team_encoder.fit(all_teams)

df_hist["HomeTeam"] = team_encoder.transform(df_hist["HomeTeam"])
df_hist["AwayTeam"] = team_encoder.transform(df_hist["AwayTeam"])

###############################################################################
# (A) CREATE HOME & AWAY POINTS
###############################################################################
def home_points_func(ftr):
    if pd.isna(ftr):
        return np.nan
    return 3 if ftr==0 else (1 if ftr==1 else 0)

def away_points_func(ftr):
    if pd.isna(ftr):
        return np.nan
    return 3 if ftr==2 else (1 if ftr==1 else 0)

df_hist["HomePoints"] = df_hist["FTR"].apply(home_points_func)
df_hist["AwayPoints"] = df_hist["FTR"].apply(away_points_func)

# Ensure we have FTHG/FTAG
for c in ["FTHG","FTAG"]:
    if c not in df_hist.columns:
        df_hist[c] = 0

###############################################################################
# (B) ROLLING RECENT FORM FOR HOME & AWAY
###############################################################################
def rolling_home(grp):
    grp = grp.sort_values("Date")
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
# (C) HEAD-TO-HEAD
###############################################################################
df_hist["MatchUp"] = df_hist.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]), axis=1)

def rolling_h2h(grp):
    grp = grp.sort_values("Date")
    grp["H2HHomePoints5"] = grp["HomePoints"].shift(1).rolling(5, min_periods=1).sum()
    grp["H2HHomeGF5"] = grp["FTHG"].shift(1).rolling(5, min_periods=1).sum()
    grp["H2HHomeGA5"] = grp["FTAG"].shift(1).rolling(5, min_periods=1).sum()
    return grp

df_hist = df_hist.groupby("MatchUp", group_keys=False).apply(rolling_h2h)

# fill rolling NaNs
roll_cols = [
    "HomeRecentPoints5","HomeRecentGF5","HomeRecentGA5",
    "AwayRecentPoints5","AwayRecentGF5","AwayRecentGA5",
    "H2HHomePoints5","H2HHomeGF5","H2HHomeGA5"
]
df_hist[roll_cols] = df_hist[roll_cols].fillna(0.0)

###############################################################################
# 3) DEFINE FEATURES & RECENCY WEIGHT
###############################################################################
features = [
    "HomeTeam","AwayTeam",
    "HS","AS","HST","AST",
    "AvgH","AvgD","AvgA",
    "HomeRecentPoints5","HomeRecentGF5","HomeRecentGA5",
    "AwayRecentPoints5","AwayRecentGF5","AwayRecentGA5",
    "H2HHomePoints5","H2HHomeGF5","H2HHomeGA5"
]
target = "FTR"

max_date = df_hist["Date"].max()
df_hist["DaysDiff"] = (max_date - df_hist["Date"]).dt.days
df_hist["RecencyWeight"] = np.power(0.99, df_hist["DaysDiff"] / 7.0)

# fill missing
for col in features:
    if col not in df_hist.columns:
        df_hist[col] = 0.0
df_hist[features] = df_hist[features].fillna(0.0)

###############################################################################
# 4) BUILD X_train, THEN STANDARD-SCALE, THEN MULTIPLY ROLLING COLUMNS
###############################################################################
X_train = df_hist[features].copy()
y_train = df_hist[target].astype(int).copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# indices of rolling columns in the final 'features' list
roll_col_names = [
    "HomeRecentPoints5","HomeRecentGF5","HomeRecentGA5",
    "AwayRecentPoints5","AwayRecentGF5","AwayRecentGA5",
    "H2HHomePoints5","H2HHomeGF5","H2HHomeGA5"
]
roll_indices = [features.index(c) for c in roll_col_names]

# multiply them by e.g. 100 => or 1000 if you want a bigger weighting
mult_factor = 20.0
for idx in roll_indices:
    X_train_scaled[:, idx] *= mult_factor

###############################################################################
# TRAIN MODEL
###############################################################################
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    random_state=42,
    class_weight={
        0:1.0, # home
        1:0.7, # draw
        2:1.0  # away
    }
)

model.fit(X_train_scaled, y_train, sample_weight=df_hist["RecencyWeight"])
print(f"Logistic Regression done for league = {league}. Used {len(df_hist)} matches with recency weighting + rolling weighting.")


###############################################################################
# 5) LOAD FIXTURES, APPLY SAME SCALING & MULTIPLICATION
###############################################################################
FIXTURES_PATH = os.path.join(DATA_DIR, "fixtures", "fixtures.csv")
if not os.path.exists(FIXTURES_PATH):
    print(f"No fixtures file found at {FIXTURES_PATH}. Exiting.")
    exit()

df_fix = pd.read_csv(FIXTURES_PATH)

df_fix = df_fix[df_fix["Div"]==league].reset_index(drop=True)
if df_fix.empty:
    print(f"No {league} fixtures found in fixtures.csv.")
    exit()

df_fix["Date"] = pd.to_datetime(df_fix["Date"], format="%d/%m/%Y", errors="coerce")
df_fix["HomeTeam"] = df_fix["HomeTeam"].str.strip().str.lower()
df_fix["AwayTeam"] = df_fix["AwayTeam"].str.strip().str.lower()

# ensure all columns exist
for col in features:
    if col not in df_fix.columns:
        df_fix[col] = 0.0

def safe_team_enc(t):
    if t not in team_encoder.classes_:
        print(f"WARNING: new team '{t}' not in training data. Adding it.")
        team_encoder.classes_ = np.append(team_encoder.classes_, t)
    return team_encoder.transform([t])[0]

df_fix["HomeTeam"] = df_fix["HomeTeam"].apply(safe_team_enc)
df_fix["AwayTeam"] = df_fix["AwayTeam"].apply(safe_team_enc)

# fill fixture missing
df_fix[features] = df_fix[features].fillna(0.0)

X_fix = df_fix[features].copy()
X_fix_scaled = scaler.transform(X_fix)

# multiply the same rolling columns in the test set too
for idx in roll_indices:
    X_fix_scaled[:, idx] *= mult_factor

# predict
probs = model.predict_proba(X_fix_scaled)
preds = model.predict(X_fix_scaled)

reverse_ftr = {0:"H",1:"D",2:"A"}

# Build a list of dicts
predictions_list = []
for i, row in df_fix.iterrows():
    ph, pd, pa = probs[i]
    pred_label = preds[i]
    label_str = reverse_ftr[pred_label]

        # Override #1: If originally "H", but (ph - pd <= 0.06), then change label to "D"
    if pred_label == 0 and (ph <= 0.45):
        label_str = "D"

    # Override #2: If originally "A", but (pa - pd <= 0.025), then change label to "D"
    if pred_label == 2 and (pa <= 0.4):
        label_str = "D"

    home_id = row["HomeTeam"]
    away_id = row["AwayTeam"]
    home_str = team_encoder.inverse_transform([int(home_id)])[0]
    away_str = team_encoder.inverse_transform([int(away_id)])[0]

    raw_date = row["Date"]
    date_str = raw_date.strftime("%Y-%m-%d") if hasattr(raw_date, "strftime") else "Unknown"
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

if predictions_list:
    print("Sample item in predictions_list:", predictions_list[0])

# final DF
from pandas.core.frame import DataFrame as RealDataFrame
df_preds = RealDataFrame(predictions_list)
print(f"\n=== UPCOMING FIXTURES FOR {league} (ROLLED + Weighted Rolling Columns) ===")
print(df_preds)

def hash_dataframe(df):
    # Convert DataFrame to CSV string (without index) and compute MD5 hash.
    csv_string = df.to_csv(index=False)
    return hashlib.md5(csv_string.encode('utf-8')).hexdigest()

# Use the current date as a base for the filename.
today_str = time.strftime("%Y-%m-%d")
base_filename = f"predictions_{league}_{today_str}.csv"
save_dir = os.path.join("data", "prediction_history")

# Create the directory if it doesn't exist.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, base_filename)

# Check if a file for today already exists.
if os.path.exists(save_path):
    # Load the existing predictions file.
    df_existing = pd.read_csv(save_path)
    # Compute hashes for the existing file and the new predictions.
    existing_hash = hash_dataframe(df_existing)
    new_hash = hash_dataframe(df_preds)
    if new_hash == existing_hash:
        print("Vorhersagen haben sich seit dem letzten Speichern nicht geändert. Kein Duplikat wird gespeichert.")
    else:
        # If the predictions differ, add an incremental suffix.
        counter = 1
        new_filename = f"predictions_{league}_{today_str}_{counter}.csv"
        new_save_path = os.path.join(save_dir, new_filename)
        while os.path.exists(new_save_path):
            counter += 1
            new_filename = f"predictions_{league}_{today_str}_{counter}.csv"
            new_save_path = os.path.join(save_dir, new_filename)
        df_preds.to_csv(new_save_path, index=False)
        print(f"Neue Vorhersagen wurden gespeichert als: {new_save_path}")
else:
    # If no file exists for today, save the new predictions.
    df_preds.to_csv(save_path, index=False)
    print(f"Vorhersagen wurden gespeichert als: {save_path}")