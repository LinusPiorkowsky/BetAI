import os
import pandas as pd
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
        key=lambda x: int(x.replace("prediction", "").replace(".csv", ""))
    )

    if not prediction_files:
        return None

    latest_file = os.path.join(PREDICTION_DIR, prediction_files[-1])
    df = pd.read_csv(latest_file)
    return df

@app.route("/")
def index():
    """Display the latest prediction."""
    df = get_latest_prediction()
    if df is None or df.empty:
        return render_template("index.html", predictions=None, leagues=[], leagues_dict=LEAGUE_NAMES)

    df["Max_Prob"] = df[["Prob_HomeWin", "Prob_Draw", "Prob_AwayWin"]].max(axis=1)
    df = df.sort_values(by="Max_Prob", ascending=False)

    leagues = df["Div"].unique().tolist()

    return render_template("index.html", predictions=df.to_dict(orient="records"), leagues=leagues, leagues_dict=LEAGUE_NAMES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
