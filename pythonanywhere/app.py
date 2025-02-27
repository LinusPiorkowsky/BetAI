import os
import glob
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request

app = Flask(__name__, static_folder="static")

# Directory where your CSV prediction files are stored
PREDICTION_DIR = "/home/MachineLearningBets/BetAI/predictions" # /home/MachineLearningBets/BetAI/predictions
RESULT_DIR = "/home/MachineLearningBets/BetAI/results" # /home/MachineLearningBets/BetAI/results

# Mapping of league codes to their names
LEAGUE_NAMES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "I1": "Serie A",
    "SP1": "La Liga"
}

# Letzte Prediction-Datei laden
def get_latest_prediction():
    """Find the latest prediction CSV file and load it into a DataFrame."""
    prediction_files = sorted(
        [f for f in os.listdir(PREDICTION_DIR)
         if f.startswith("prediction") and f.endswith(".csv")],
        key=lambda x: int(''.join(filter(str.isdigit, x.replace("prediction", ""))))
                      if any(char.isdigit() for char in x) else 0,
        reverse=True  # Sort descending to get the newest file first
    )

    if not prediction_files:
        return None

    latest_file = os.path.join(PREDICTION_DIR, prediction_files[0])
    df = pd.read_csv(latest_file)

    # Convert Date and Time columns to a proper datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["Date"] = df["Date"].astype(str) + " " + df["Time"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Example of removing matches that started more than 2.5 hours ago
    # (Uncomment if you want to filter out past matches)
    now = datetime.now()
    df = df[df["Date"] + timedelta(hours=1.0) > now]

    # Capitalize team names
    df["HomeTeam"] = df["HomeTeam"].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))
    df["AwayTeam"] = df["AwayTeam"].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))

    return df

def load_all_result_files():
    """
    Reads ALL result_{i}.csv files in the results directory into a single DataFrame.
    Returns the combined DataFrame (or None if none are found).
    """
    pattern = os.path.join(RESULT_DIR, "result_*.csv")
    file_list = glob.glob(pattern)

    if not file_list:
        return None

    frames = []
    for file_path in file_list:
        df_temp = pd.read_csv(file_path)

        # Convert 'Date' column to datetime (coerce errors to NaT)
        df_temp["Date"] = pd.to_datetime(df_temp["Date"], format="%Y-%m-%d", errors="coerce")

        # Capitalize team names (optional)
        if "HomeTeam" in df_temp.columns:
            df_temp["HomeTeam"] = df_temp["HomeTeam"].apply(
                lambda x: ' '.join([word.capitalize() for word in str(x).split()])
            )
        if "AwayTeam" in df_temp.columns:
            df_temp["AwayTeam"] = df_temp["AwayTeam"].apply(
                lambda x: ' '.join([word.capitalize() for word in str(x).split()])
            )

        frames.append(df_temp)

    # Concatenate all DataFrames
    all_results = pd.concat(frames, ignore_index=True)

    # Sort by Date and (if present) Time
    if "Time" in all_results.columns:
        all_results.sort_values(by=["Date", "Time"], inplace=True)
    else:
        all_results.sort_values(by="Date", inplace=True)

    # Reset index to clean up after sort
    all_results.reset_index(drop=True, inplace=True)

    return all_results if not all_results.empty else None


def get_latest_result():
    """Find the latest result CSV file and load it into a DataFrame."""
    # Sort by the numeric part in the filename descending
    prediction_files = sorted(
        [f for f in os.listdir(RESULT_DIR)
         if f.startswith("result") and f.endswith(".csv")],
        key=lambda x: int(''.join(filter(str.isdigit, x.replace("result", ""))))
                      if any(char.isdigit() for char in x) else 0,
        reverse=True
    )

    if not prediction_files:
        return None

    latest_file = os.path.join(RESULT_DIR, prediction_files[0])
    df = pd.read_csv(latest_file)

    # Convert Date and Time columns to a proper datetime
    if "Date" in df.columns and "Time" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        # Combine Date + Time into a single datetime column if you prefer
        df["Date"] = df["Date"].astype(str) + " " + df["Time"]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Capitalize team names
    if "HomeTeam" in df.columns:
        df["HomeTeam"] = df["HomeTeam"].apply(lambda x: ' '.join([word.capitalize() for word in str(x).split()]))
    if "AwayTeam" in df.columns:
        df["AwayTeam"] = df["AwayTeam"].apply(lambda x: ' '.join([word.capitalize() for word in str(x).split()]))

    return df


@app.route("/")
def index():
    """
    Main route that:
      - Loads all results from CSV
      - Determines final bet based on your logic
      - Checks correctness
      - Calculates total & last-week performance for high-confidence vs. all bets
      - Passes data to the template
    """
    all_results = load_all_result_files()
    if all_results is None or all_results.empty:
        # No data found, just pass zeros or placeholders
        return render_template(
            "index.html",
            # High-confidence overall
            high_conf_bets_total=0,
            high_conf_correct_total=0,
            high_conf_accuracy_total=0.0,
            # All bets overall
            all_bets_total=0,
            all_correct_total=0,
            all_accuracy_total=0.0,
            # High-confidence last week
            high_conf_bets_week=0,
            high_conf_correct_week=0,
            high_conf_accuracy_week=0.0,
            # All bets last week
            all_bets_week=0,
            all_correct_week=0,
            all_accuracy_week=0.0,
            # (If you want to pass the DataFrame to the template)
            all_results=None
        )

    # ------------------------------------------------------------------
    # 1) Convert "Date" to datetime if it isn't already
    # ------------------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(all_results["Date"]):
        all_results["Date"] = pd.to_datetime(all_results["Date"], errors="coerce")

    # ------------------------------------------------------------------
    # 2) Determine the "FinalBet" based on your custom rules
    #    (the same logic you showed for Jinja, but here in Python)
    # ------------------------------------------------------------------
    def get_final_bet(row):
        """
        Replicates your conditional logic for 'Prediction' + probabilities + High_conf
        to decide if the final bet is 'H', '1X', 'X2', etc.
        """
        prediction = row["Prediction"]  # 'H','D','A'
        prob_h = row.get("Prob_H", 0.0)  # default to 0 if missing
        prob_d = row.get("Prob_D", 0.0)
        prob_a = row.get("Prob_A", 0.0)
        high_conf = row.get("High_conf", 0)
        b365h = row.get("B365H", 0.0)
        b365a = row.get("B365A", 0.0)

        if prediction == "H":
            # If Prob_H <= 0.6 => '1X'; else 'H'
            return "1X" if prob_h <= 0.65 else "H"

        elif prediction == "D":
            # If Prob_H + Prob_D > Prob_D + Prob_A => '1X' else 'X2'
            if (b365h) < (b365a):
                return "1X"
            else:
                return "X2"

        elif prediction == "A":
            if (b365a) < 2.21 and prob_a > 0.65:
                return "A"
            else:
                return "X2"

        # If no condition matched, fallback
        return None

    all_results["FinalBet"] = all_results.apply(get_final_bet, axis=1)

    # ------------------------------------------------------------------
    # 3) Determine correctness of that "FinalBet"
    #    (or you can override your existing "Prediction_Correct" if you want)
    # ------------------------------------------------------------------
    def is_bet_correct(actual_result, final_bet):
        """
        actual_result: 'H','D','A'
        final_bet:     'H','D','A','1X','X2', etc.

        Rules:
          - If actual == H => correct if final_bet in [H, 1X]
          - If actual == D => correct if final_bet in [D, 1X, X2]
          - If actual == A => correct if final_bet in [A, X2]
        """
        if pd.isna(actual_result) or pd.isna(final_bet):
            return False  # No data, can't be correct
        if actual_result == "H":
            return final_bet in ["H", "1X"]
        elif actual_result == "D":
            return final_bet in ["D", "1X", "X2"]
        elif actual_result == "A":
            return final_bet in ["A", "X2"]
        return False

    all_results["BetCorrect"] = all_results.apply(
        lambda row: is_bet_correct(row["Actual_Result"], row["FinalBet"]),
        axis=1
    )

    # ------------------------------------------------------------------
    # 4) Calculate stats for total / last-week
    # ------------------------------------------------------------------
    # 4a) Convert correctness to 0/1 for easy sum
    all_results["BetCorrectInt"] = all_results["BetCorrect"].astype(int)

    # 4b) High-confidence (where High_conf == 1)
    df_high_conf = all_results[all_results["High_conf"] == 1]
    high_conf_bets_total = len(df_high_conf)
    high_conf_correct_total = df_high_conf["BetCorrectInt"].sum()
    high_conf_accuracy_total = (
        round(high_conf_correct_total / high_conf_bets_total * 100, 2)
        if high_conf_bets_total > 0 else 0.0
    )

    # 4c) All bets
    all_bets_total = len(all_results)
    all_correct_total = all_results["BetCorrectInt"].sum()
    all_accuracy_total = (
        round(all_correct_total / all_bets_total * 100, 2)
        if all_bets_total > 0 else 0.0
    )

    # Ensure Date column is in datetime format
    all_results["Date"] = pd.to_datetime(all_results["Date"]).dt.normalize()

    # Get today's date
    today = pd.Timestamp.now().normalize()

    # Get the current ISO year and week number
    current_year, current_week, _ = today.isocalendar()

    # Assign year and week to each row in the DataFrame
    all_results["Year"] = all_results["Date"].dt.isocalendar().year
    all_results["Week"] = all_results["Date"].dt.isocalendar().week

    # Determine last week's year and week number
    if current_week == 1:  # Handle cases where it's the first week of the year
        last_week_year = current_year - 1
        last_week_number = 52  # Assuming standard 52-week year (can be 53 in some cases)
    else:
        last_week_year = current_year
        last_week_number = current_week - 1

    # Filter only last week's data (Monday to Sunday)
    df_last_week = all_results[
        (all_results["Year"] == last_week_year) & (all_results["Week"] == last_week_number)
    ]

    # High confidence last week
    df_high_conf_week = df_last_week[df_last_week["High_conf"] == 1]
    high_conf_bets_week = len(df_high_conf_week)
    high_conf_correct_week = df_high_conf_week["BetCorrectInt"].sum()
    high_conf_accuracy_week = (
        round(high_conf_correct_week / high_conf_bets_week * 100, 2)
        if high_conf_bets_week > 0 else 0.0
    )

    # All bets last week
    all_bets_week = len(df_last_week)
    all_correct_week = df_last_week["BetCorrectInt"].sum()
    all_accuracy_week = (
        round(all_correct_week / all_bets_week * 100, 2)
        if all_bets_week > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 5) Render the template, passing stats + entire DF (if needed)
    # ------------------------------------------------------------------
    return render_template(
        "index.html",
        all_results=all_results,  # for display/filter in your template if desired

        # High Confidence - total
        high_conf_bets_total=high_conf_bets_total,
        high_conf_correct_total=high_conf_correct_total,
        high_conf_accuracy_total=high_conf_accuracy_total,

        # All Bets - total
        all_bets_total=all_bets_total,
        all_correct_total=all_correct_total,
        all_accuracy_total=all_accuracy_total,

        # High Confidence - last week
        high_conf_bets_week=high_conf_bets_week,
        high_conf_correct_week=high_conf_correct_week,
        high_conf_accuracy_week=high_conf_accuracy_week,

        # All Bets - last week
        all_bets_week=all_bets_week,
        all_correct_week=all_correct_week,
        all_accuracy_week=all_accuracy_week
    )


@app.route("/predictions")
def predictions():
    df = get_latest_prediction()

    if df is None or df.empty:
        return render_template(
            "predictions.html",
            predictions=None,
            date_range="",
            leagues=[],
            leagues_dict=LEAGUE_NAMES
        )

    df = df.sort_values(by=["Date", "Time"])
    first_date = df["Date"].min().strftime("%d/%m/%Y")
    last_date = df["Date"].max().strftime("%d/%m/%Y")
    date_range = f"{first_date} - {last_date}" if first_date != last_date else first_date

    # Convert DataFrame rows to a list of dictionaries
    predictions_list = df.to_dict(orient="records")

    # Unique leagues
    leagues = df["Div"].unique().tolist()

    return render_template(
        "predictions.html",
        predictions=predictions_list,
        date_range=date_range,
        leagues=leagues,
        leagues_dict=LEAGUE_NAMES
    )


@app.route("/results")
def results():
    """
    Display Past Results in 'card' style, with final bet logic & correctness.
    """
    df = load_all_result_files()
    if df is None or df.empty:
        return render_template("results.html", results=None, leagues=[], weeks=[])

    # -------------------------------------------------------------------------
    # APPLY YOUR get_final_bet & is_bet_correct LOGIC HERE, so we don't modify
    # load_all_result_files() at all.
    # -------------------------------------------------------------------------
    # 1) Standardize 'Prediction'/'Actual_Result' if needed
    if "Prediction" in df.columns:
        df["Prediction"] = df["Prediction"].astype(str).str.strip().str.upper()
    if "Actual_Result" in df.columns:
        df["Actual_Result"] = df["Actual_Result"].astype(str).str.strip().str.upper()

    # 2) Convert High_conf to numeric if it exists
    if "High_conf" in df.columns:
        df["High_conf"] = pd.to_numeric(df["High_conf"], errors="coerce").fillna(0).astype(int)
    else:
        df["High_conf"] = 0

    # 3) Define your logic for final bet
    def get_final_bet(row):
        """
        Same logic as provided:
        - If 'Prediction' == 'H' => '1X' if prob_h <= 0.6 else 'H'
        - If 'Prediction' == 'D' => '1X' if (prob_h+prob_d)>(prob_d+prob_a), else 'X2'
        - If 'Prediction' == 'A' => 'A' if high_conf == 1, else 'X2'
        """
        prediction = row.get("Prediction", "")
        prob_h = row.get("Prob_H", 0.0)
        prob_d = row.get("Prob_D", 0.0)
        prob_a = row.get("Prob_A", 0.0)
        high_conf = row.get("High_conf", 0)
        b365h = row.get("B365H", 0.0)
        b365a = row.get("B365A", 0.0)

        if prediction == "H":
            return "1X" if prob_h <= 0.65 else "H"
        elif prediction == "D":
            if (b365h) < (b365a):
                return "1X"
            else:
                return "X2"
        elif prediction == "A":
            if (b365a) < 2.21 and prob_a > 0.65:
                return "A"
            else:
                return "X2"

        return None

    # 4) Apply the final bet logic
    df["FinalBet"] = df.apply(get_final_bet, axis=1)

    # 5) Correctness logic
    def is_bet_correct(actual_result, final_bet):
        """
        - If actual == H => correct if final_bet in [H, 1X]
        - If actual == D => correct if final_bet in [D, 1X, X2]
        - If actual == A => correct if final_bet in [A, X2]
        """
        if pd.isna(actual_result) or pd.isna(final_bet):
            return False
        if actual_result == "H":
            return final_bet in ["H", "1X"]
        elif actual_result == "D":
            return final_bet in ["D", "1X", "X2"]
        elif actual_result == "A":
            return final_bet in ["A", "X2"]
        return False

    df["BetCorrect"] = df.apply(
        lambda row: is_bet_correct(row.get("Actual_Result"), row.get("FinalBet")),
        axis=1
    )

    # 6) Create a 'Week' column if you want to filter
    df["Week"] = df["Date"].dt.isocalendar().week

    # sort by Date and Time backwards (most recent first)
    df = df.sort_values(by=["Date", "Time"], ascending=False)

    # 7) Gather unique leagues/weeks for filtering
    leagues = sorted(df["Div"].unique().tolist())
    weeks = sorted(df["Week"].unique().tolist())

    # Finally, pass the updated DataFrame to your template
    return render_template(
        "results.html",
        results=df,
        leagues=leagues,
        weeks=weeks,
        league_names=LEAGUE_NAMES
    )

@app.route("/informations")
def informations():
    """
    FAQ / Informations page about the ML models, data sources, etc.
    """
    return render_template("informations.html")

if __name__ == "__main__":
    app.run(debug=True)
