import os
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template

app = Flask(__name__)

PREDICTION_DIR = "data/predictions"

LEAGUE_NAMES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "I1": "Serie A",
    "SP1": "La Liga"
}

def get_latest_prediction():
    """Find the latest prediction file and load it."""
    prediction_files = sorted(
        [f for f in os.listdir(PREDICTION_DIR) if f.startswith("prediction") and f.endswith(".csv")],
        key=lambda x: int(''.join(filter(str.isdigit, x.replace("prediction", "")))) if any(char.isdigit() for char in x) else 0,
        reverse=True  # Sort in descending order, so we get the latest one
    )

    if not prediction_files:
        return None

    latest_file = os.path.join(PREDICTION_DIR, prediction_files[0])  # Get the most recent file
    df = pd.read_csv(latest_file)

    # Convert Date & Time for filtering and sorting
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df["MatchDateTime"] = df["Date"].astype(str) + " " + df["Time"]
    df["MatchDateTime"] = pd.to_datetime(df["MatchDateTime"], errors="coerce")

    # Remove matches that passed 2.5 hours ago
    now = datetime.now()
    df = df[df["MatchDateTime"] + timedelta(hours=2.5) > now]

    return df

@app.route("/")
def index():
    df = get_latest_prediction()
    if df is None or df.empty:
        return render_template("index.html", predictions=None, date_range="", leagues=[], leagues_dict=LEAGUE_NAMES)

    # Sort matches by date and time
    df = df.sort_values(by=["Date", "Time"])

    # Format the date range for the header
    first_date = df["Date"].min().strftime("%d/%m/%Y")
    last_date = df["Date"].max().strftime("%d/%m/%Y")
    date_range = f"{first_date} - {last_date}" if first_date != last_date else first_date

    leagues = df["Div"].unique().tolist()

    return render_template("index.html", predictions=df.to_dict(orient="records"), date_range=date_range, leagues=leagues, leagues_dict=LEAGUE_NAMES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
