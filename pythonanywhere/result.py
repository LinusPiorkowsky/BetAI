#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import re
import requests
import zipfile
import pandas as pd

# ------------------------------------------------------------------------------
# 1) GLOBAL CONFIG
# ------------------------------------------------------------------------------
BASE_DIR = "/home/MachineLearningBets/BetAI"

SAVE_DIR = os.path.join(BASE_DIR, "data", "2024_25")  # Where fresh league CSVs are stored
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ALLOWED_LEAGUES = {
    "D1.csv","D2.csv","E0.csv","E1.csv","F1.csv","F2.csv",
    "I1.csv","I2.csv","SP1.csv","SP2.csv"
}

# Example Football-Data URL for the 2024/25 season
DATASET_URL = "https://www.football-data.co.uk/mmz4281/2425/data.zip"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) (OPTIONAL) DOWNLOAD ZIP & EXTRACT
# ------------------------------------------------------------------------------
def download_dataset(url: str, zip_filename: str) -> None:
    """
    Download a ZIP from `url`, extract only ALLOWED_LEAGUES into SAVE_DIR.
    """
    temp_zip_path = os.path.join(SAVE_DIR, zip_filename)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        if not resp.content:
            print("❌ The downloaded dataset is empty.")
            return

        with open(temp_zip_path, 'wb') as f:
            f.write(resp.content)

        with zipfile.ZipFile(temp_zip_path, 'r') as zf:
            for member in zf.namelist():
                if any(member.endswith(league) for league in ALLOWED_LEAGUES):
                    zf.extract(member, SAVE_DIR)

        os.remove(temp_zip_path)
        print(f"✅ Downloaded & extracted dataset into: {SAVE_DIR}")

    except requests.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")
    except zipfile.BadZipFile:
        print("❌ Error: Not a valid ZIP file.")

# ------------------------------------------------------------------------------
# 3) READ FRESH ACTUAL RESULTS
# ------------------------------------------------------------------------------
def read_fresh_actuals(data_directory: str) -> pd.DataFrame:
    """
    Reads any .csv in `data_directory` matching ALLOWED_LEAGUES.
    Keeps essential columns:
       Div, Date, Time, HomeTeam, AwayTeam, FTHG→FTHG_h, FTAG→FTAG_h, FTR→FTR_h
    Returns a combined DataFrame.
    """
    league_files = [
        f for f in os.listdir(data_directory)
        if any(f.endswith(league) for league in ALLOWED_LEAGUES)
    ]
    if not league_files:
        print("❌ No allowed league files found. Returning empty.")
        return pd.DataFrame(columns=["Div","Date","Time","HomeTeam","AwayTeam","FTHG_h","FTAG_h","FTR_h"])

    df_list = []
    for csv_file in league_files:
        path = os.path.join(data_directory, csv_file)
        print(f"Reading actual data file: {csv_file}")
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")

        # Rename columns
        rename_map = {}
        if "FTHG" in df.columns: rename_map["FTHG"] = "FTHG_h"
        if "FTAG" in df.columns: rename_map["FTAG"] = "FTAG_h"
        if "FTR"  in df.columns: rename_map["FTR"]  = "FTR_h"
        df.rename(columns=rename_map, inplace=True)

        keep_cols = {"Div","Date","Time","HomeTeam","AwayTeam","FTHG_h","FTAG_h","FTR_h"}
        df = df[list(keep_cols.intersection(df.columns))]
        # ✅ Fix Time: Add 1 hour if it exists
        if "Time" in df.columns and df["Time"].notna().all():
            df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce") + datetime.timedelta(hours=1)
            df["Time"] = df["Time"].dt.strftime("%H:%M")  # Convert back to string format

        df_list.append(df)


    return pd.concat(df_list, ignore_index=True).drop_duplicates()

# ------------------------------------------------------------------------------
# 4) READ HISTORICAL RESULTS (result_*.csv)
# ------------------------------------------------------------------------------
def read_all_results(results_directory: str) -> pd.DataFrame:
    """
    Reads each 'result_*.csv', combining them into a single DataFrame.
    We assume final output columns might include 'Actual_Result','Score','FTHG_h','FTAG_h','FTR_h' etc.
    We'll store at least [Date,HomeTeam,AwayTeam] to avoid duplicates.
    """
    all_files = sorted(
        f for f in os.listdir(results_directory)
        if re.match(r"result_(\d+)\.csv$", f)
    )
    if not all_files:
        print("No existing result_*.csv found; starting fresh.")
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam"])

    df_list = []
    for fn in all_files:
        path = os.path.join(results_directory, fn)
        print(f"Reading historical file: {fn}")
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")
        df_list.append(df)

    combined = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["Date","HomeTeam","AwayTeam"])
    return combined

# ------------------------------------------------------------------------------
# 5) READ ALL PREDICTIONS (predictions_*.csv)
# ------------------------------------------------------------------------------
def read_all_predictions(pred_dir: str) -> pd.DataFrame:
    """
    Reads all 'predictions_(N).csv' into a single DataFrame.
    Must have columns like: Div,Date,Weekday,Time,HomeTeam,AwayTeam,Prediction,...
    """
    all_preds = sorted(
        f for f in os.listdir(pred_dir)
        if re.match(r"predictions_(\d+)\.csv$", f)
    )
    if not all_preds:
        print("No predictions found.")
        return pd.DataFrame()

    df_list = []
    for fn in all_preds:
        path = os.path.join(pred_dir, fn)
        print(f"Reading predictions file: {fn}")
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True).drop_duplicates()

# ------------------------------------------------------------------------------
# 6) GET NEXT RESULT NUMBER
# ------------------------------------------------------------------------------
def get_next_result_number(results_directory: str) -> int:
    existing = [
        f for f in os.listdir(results_directory)
        if re.match(r"result_(\d+)\.csv$", f)
    ]
    if not existing:
        return 1

    def extract_num(fname):
        m = re.search(r"result_(\d+)\.csv$", fname)
        return int(m.group(1)) if m else 0

    return max(extract_num(x) for x in existing) + 1

# ------------------------------------------------------------------------------
# 7) MAIN LOGIC
# ------------------------------------------------------------------------------
def main():
    """
    Steps:
      1) Download & unzip data (optional).
      2) Read fresh actual data => FTHG_h, FTAG_h, FTR_h
      3) Read existing results => historical
      4) Read predictions => predictions
      5) Merge => only keep matches with a final result
      6) Exclude duplicates => only new
      7) Reformat columns => save as result_X.csv
    """

    # 1) Download step (comment out if not needed):
    download_dataset(DATASET_URL, "freshdata.zip")

    # 2) Fresh actual data
    fresh_df = read_fresh_actuals(SAVE_DIR)
    if fresh_df.empty:
        print("No fresh actual data. Exiting.")
        return

    # 3) Historical
    historical_df = read_all_results(RESULTS_DIR)

    # 4) Predictions
    preds_df = read_all_predictions(PREDICTIONS_DIR)
    if preds_df.empty:
        print("No predictions found. Exiting.")
        return

    # 5) Merge predictions with fresh actual results (on [Date,HomeTeam,AwayTeam])
    print("Merging predictions with actual results to find completed matches...")
    preds_df["Date"] = pd.to_datetime(preds_df["Date"], errors="coerce")
    fresh_df["Date"] = pd.to_datetime(fresh_df["Date"], errors="coerce")

    merged = pd.merge(
        preds_df,
        fresh_df,
        on=["Date","HomeTeam","AwayTeam"],
        how="left",
        suffixes=("_pred", "_act")
    )
    print(f"Merged shape={merged.shape}")

    # Ensure 'Div' is properly retained from predictions
    if "Div_pred" in merged.columns:
        merged.rename(columns={"Div_pred": "Div"}, inplace=True)
    elif "Div" in preds_df.columns:
        merged["Div"] = preds_df["Div"]

    # Only keep rows that have a final result => FTR_h not null
    completed = merged[merged["FTR_h"].notna()].copy()
    print(f"{len(completed)} matches are completed and in predictions.")

    if completed.empty:
        print("No new completed matches found.")
        return

    # 6) Exclude matches already in historical
    if not historical_df.empty:
        hist_idx = historical_df.set_index(["Date","HomeTeam","AwayTeam"])
        comp_idx = completed.set_index(["Date","HomeTeam","AwayTeam"])
        duplicates_mask = comp_idx.index.isin(hist_idx.index)
        truly_new = completed[~duplicates_mask].reset_index(drop=True)
    else:
        truly_new = completed

    print(f"{len(truly_new)} matches are truly new (not in historical).")
    if truly_new.empty:
        print("All these completed matches already exist in historical. Nothing to add.")
        return

    # 7) Reformat columns for final CSV

    # (A) Rename FTR_h -> Actual_Result
    truly_new["Actual_Result"] = truly_new["FTR_h"]

    # (B) Create Score from FTHG_h, FTAG_h (e.g., "2:1")
    def make_score(row):
        if pd.notna(row.get("FTHG_h")) and pd.notna(row.get("FTAG_h")):
            return f"{int(row['FTHG_h'])}:{int(row['FTAG_h'])}"
        return None
    truly_new["Score"] = truly_new.apply(make_score, axis=1)

    # (C) Prediction_Correct
    if "Prediction" in truly_new.columns:
        truly_new["Prediction_Correct"] = truly_new["Prediction"] == truly_new["Actual_Result"]
    else:
        truly_new["Prediction_Correct"] = None

    # (D) DC_1X_Correct / DC_X2_Correct
    truly_new["DC_1X_Correct"] = truly_new.apply(
        lambda x: (x["Actual_Result"] in ["H","D"]) if pd.notna(x.get("1X_odds")) else None,
        axis=1
    )
    truly_new["DC_X2_Correct"] = truly_new.apply(
        lambda x: (x["Actual_Result"] in ["D","A"]) if pd.notna(x.get("X2_odds")) else None,
        axis=1
    )

    # (E) Final columns in your desired order
    final_cols = [
        "Div","Date","Weekday","Time","HomeTeam","AwayTeam","Prediction","Actual_Result",
        "Score","Prediction_Correct","B365H","B365D","B365A","Prob_H","Prob_D","Prob_A",
        "double_chance","1X_odds","X2_odds","DC_1X_Correct","DC_X2_Correct",
        "High_conf","High_conf_dc"
    ]

    # Reindex to these columns if they exist in the data
    existing_cols = [c for c in final_cols if c in truly_new.columns]
    final_df = truly_new.reindex(columns=existing_cols)

    # Time HH:MM format
    if "Time" in final_df.columns:
        final_df["Time"] = final_df["Time"].str.replace(":00", "", regex=False)

    # Save to result_{N}.csv
    next_num = get_next_result_number(RESULTS_DIR)
    out_name = f"result_{next_num}.csv"
    out_path = os.path.join(RESULTS_DIR, out_name)
    final_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"✅ Wrote {len(final_df)} new matches to {out_name}.")

# ------------------------------------------------------------------------------
# 8) ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
