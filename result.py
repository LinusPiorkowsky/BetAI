#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import requests
import zipfile
import pandas as pd

# ------------------------------------------------------------------------------
# 1) GLOBAL CONFIG
# ------------------------------------------------------------------------------
BASE_DIR = "/home/MachineLearningBets/BetAI"

SAVE_DIR = os.path.join(BASE_DIR, "data", "2024_25")   # Where we store newly extracted league CSVs
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

ALLOWED_LEAGUES = {
    "D1.csv","D2.csv","E0.csv","E1.csv","F1.csv","F2.csv",
    "I1.csv","I2.csv","SP1.csv","SP2.csv"
}

# URL from football-data.co.uk (update if desired)
DATASET_URL = "https://www.football-data.co.uk/mmz4281/2425/data.zip"

# Ensure directories exist
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
# 3) READ FRESH ACTUALS (from data/2024_25)
# ------------------------------------------------------------------------------
def read_fresh_actuals(data_directory: str) -> pd.DataFrame:
    """
    Reads any CSV in data_directory that ends with one of ALLOWED_LEAGUES,
    merges them, renames FTHG->FTHG_h, FTAG->FTAG_h, FTR->FTR_h, etc.
    Typical football-data.co.uk columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, ...
    We'll keep [Date, HomeTeam, AwayTeam, FTHG_h, FTAG_h, FTR_h] for merging with predictions.
    """
    print(f"🔎 Reading fresh actual results in: {data_directory}")
    all_files = os.listdir(data_directory)

    # Identify relevant CSVs
    league_files = [
        f for f in all_files
        if any(f.endswith(league) for league in ALLOWED_LEAGUES)
    ]

    if not league_files:
        print("  => No league files found in data/2024_25. Returning empty DF.")
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam","FTHG_h","FTAG_h","FTR_h"])

    df_list = []
    for csv_file in league_files:
        path = os.path.join(data_directory, csv_file)
        print(f"  Reading fresh league file: {csv_file}")

        # Removed errors="ignore" because it's invalid for read_csv()
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")

        # Rename standard columns
        rename_map = {}
        if "FTHG" in df.columns:
            rename_map["FTHG"] = "FTHG_h"
        if "FTAG" in df.columns:
            rename_map["FTAG"] = "FTAG_h"
        if "FTR" in df.columns:
            rename_map["FTR"] = "FTR_h"

        df.rename(columns=rename_map, inplace=True)

        # Keep only relevant columns
        keep_cols = {"Date","HomeTeam","AwayTeam","FTHG_h","FTAG_h","FTR_h"}
        existing = list(keep_cols.intersection(df.columns))
        df = df[existing]
        df_list.append(df)

    combined = pd.concat(df_list, ignore_index=True).drop_duplicates()
    print(f"🔎 Fresh actuals shape={combined.shape}, columns={list(combined.columns)}")
    return combined

# ------------------------------------------------------------------------------
# 4) READ HISTORICAL RESULTS (result_*.csv)
# ------------------------------------------------------------------------------
def read_all_results(results_directory: str) -> pd.DataFrame:
    """
    Reads each 'result_*.csv', renames 'Actual_Result' -> 'FTR_h' if present,
    parses 'Score' -> FTHG_h, FTAG_h if present, merges them all.
    """
    print(f"🔎 Checking for 'result_*.csv' in {results_directory}")
    all_results = sorted(
        f for f in os.listdir(results_directory)
        if re.match(r"result_(\d+)\.csv$", f)
    )
    if not all_results:
        print("  => No result_*.csv found, returning empty.")
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam","FTR_h","FTHG_h","FTAG_h"])

    df_list = []
    for fname in all_results:
        path = os.path.join(results_directory, fname)
        print(f"  Reading historical result file: {fname}")

        # Removed errors="ignore"
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")

        # If 'Actual_Result' is present, rename to 'FTR_h'
        if "Actual_Result" in df.columns:
            df.rename(columns={"Actual_Result":"FTR_h"}, inplace=True)

        # If 'Score' is present, parse it => FTHG_h, FTAG_h
        if "Score" in df.columns:
            fthg_vals = []
            ftag_vals = []
            for val in df["Score"]:
                if isinstance(val, str) and ":" in val:
                    try:
                        hg_str, ag_str = val.split(":",1)
                        fthg_vals.append(int(hg_str))
                        ftag_vals.append(int(ag_str))
                    except ValueError:
                        fthg_vals.append(None)
                        ftag_vals.append(None)
                else:
                    fthg_vals.append(None)
                    ftag_vals.append(None)
            df["FTHG_h"] = fthg_vals
            df["FTAG_h"] = ftag_vals

        df_list.append(df)

    combined = pd.concat(df_list, ignore_index=True).drop_duplicates()
    print(f"🔎 Combined historical shape={combined.shape}, columns={list(combined.columns)}")
    return combined

# ------------------------------------------------------------------------------
# 5) READ ALL PREDICTIONS (predictions_*.csv)
# ------------------------------------------------------------------------------
def read_all_predictions(pred_dir: str) -> pd.DataFrame:
    """
    Reads all 'predictions_(N).csv' into one big DataFrame, ignoring duplicates.
    """
    print(f"🔎 Checking for 'predictions_*.csv' in {pred_dir}")
    all_preds = sorted(
        f for f in os.listdir(pred_dir)
        if re.match(r"predictions_(\d+)\.csv$", f)
    )
    if not all_preds:
        print("  => No prediction files found.")
        return pd.DataFrame()

    df_list = []
    for fname in all_preds:
        path = os.path.join(pred_dir, fname)
        print(f"  Reading {fname}")

        # Removed errors="ignore"
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=True, encoding="utf-8")
        df_list.append(df)

    combined_pred = pd.concat(df_list, ignore_index=True).drop_duplicates()
    print(f"🔎 Combined predictions shape={combined_pred.shape}, columns={list(combined_pred.columns)}")
    return combined_pred

# ------------------------------------------------------------------------------
# 6) GET NEXT RESULT NUMBER
# ------------------------------------------------------------------------------
def get_next_result_number(results_directory: str) -> int:
    """
    Return 1 + max of the existing 'result_(N).csv' or 1 if none exist.
    """
    all_results = [
        f for f in os.listdir(results_directory)
        if re.match(r"result_(\d+)\.csv$", f)
    ]
    if not all_results:
        return 1

    def extract_num(fname):
        m = re.search(r"result_(\d+)\.csv$", fname)
        return int(m.group(1)) if m else 0

    return max(extract_num(x) for x in all_results) + 1

# ------------------------------------------------------------------------------
# 7) MAIN LOGIC
# ------------------------------------------------------------------------------
def main():
    """
    Steps:
      1) Download & unzip new data (optional).
      2) Read fresh "actuals" from data/2024_25 => fresh_actuals_df
      3) Read existing results => historical_df
      4) Read all predictions => predictions_df
      5) Compare predictions with fresh_actuals => newly completed matches
      6) Exclude any matches already in historical => truly new
      7) Write them to a single result_{N}.csv if any exist
    """

    # (1) Download step (comment out if not needed)
    download_dataset(DATASET_URL, "freshdata.zip")

    # (2) Fresh actual results from your newly extracted CSV files
    fresh_actuals_df = read_fresh_actuals(SAVE_DIR)
    if fresh_actuals_df.empty:
        print("❌ No fresh actual data found. Exiting.")
        return

    # (3) Existing historical data from result_*.csv
    historical_df = read_all_results(RESULTS_DIR)

    # (4) All predictions from predictions_*.csv
    predictions_df = read_all_predictions(PREDICTIONS_DIR)
    if predictions_df.empty:
        print("❌ No predictions found. Exiting.")
        return

    # (5) Merge predictions with fresh actuals
    print("🔎 Merging predictions with fresh actuals to see which matches are completed...")
    predictions_df["Date"] = pd.to_datetime(predictions_df["Date"], errors="coerce")
    fresh_actuals_df["Date"] = pd.to_datetime(fresh_actuals_df["Date"], errors="coerce")

    merged = pd.merge(
        predictions_df,
        fresh_actuals_df,
        on=["Date","HomeTeam","AwayTeam"],
        how="left"
    )
    print(f"   => merged shape={merged.shape}, columns={list(merged.columns)}")

    # Keep only matches that have a known final result => 'FTR_h' not null
    completed = merged[merged["FTR_h"].notna()].copy()
    print(f"   => {len(completed)} matches are completed (have final result).")

    if completed.empty:
        print("✅ No newly completed matches found in predictions. Done.")
        return

    # (6) Exclude matches already in historical => truly new
    hist_idx = historical_df.set_index(["Date","HomeTeam","AwayTeam"])
    comp_idx = completed.set_index(["Date","HomeTeam","AwayTeam"])

    duplicates_mask = comp_idx.index.isin(hist_idx.index)
    truly_new = completed[~duplicates_mask].reset_index(drop=True)
    print(f"   => {len(truly_new)} truly new matches (not in historical).")

    if truly_new.empty:
        print("✅ All these completed matches are already in historical. Nothing to add.")
        return

    # (7) Write them out to a new result_{N}.csv
    next_num = get_next_result_number(RESULTS_DIR)
    out_name = f"result_{next_num}.csv"
    out_path = os.path.join(RESULTS_DIR, out_name)
    truly_new.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Wrote {len(truly_new)} newly completed matches to {out_name}.")
    print("=== DONE. ===")

# ------------------------------------------------------------------------------
# 8) ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
