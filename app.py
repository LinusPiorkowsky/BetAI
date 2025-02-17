import os
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template

app = Flask(__name__)

# Directory where your CSV prediction files are stored
PREDICTION_DIR = "predictions"

# Mapping of league codes to their names
LEAGUE_NAMES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "I1": "Serie A",
    "SP1": "La Liga"
}

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
    # now = datetime.now()
    # df = df[df["Date"] + timedelta(hours=2.5) > now]

    # Capitalize team names
    df["HomeTeam"] = df["HomeTeam"].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))
    df["AwayTeam"] = df["AwayTeam"].apply(lambda x: ' '.join([word.capitalize() for word in x.split()]))

    return df

@app.route("/")
def index():
    """
    Landing page: can show a welcome message, site info, etc.
    If you want to display some basic info, pass data to index.html as needed.
    """
    return render_template("index.html")

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
    """Placeholder for your results page."""
    return render_template("results.html")

@app.route("/informations")
def informations():
    """Placeholder for your informations page."""
    return render_template("informations.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
