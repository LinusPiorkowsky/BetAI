import os
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

# Verzeichnis der Predictions
PREDICTION_DIR = "data/predictions"

def get_latest_prediction():
    """Findet die neueste Prediction-Datei und lädt sie als DataFrame."""
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
    """Zeigt die neueste Prediction."""
    df = get_latest_prediction()
    if df is None or df.empty:
        return render_template("index.html", predictions=None)

    # Sortiere nach Wahrscheinlichkeit eines Sieges (egal welches)
    df["Max_Prob"] = df[["Prob_HomeWin", "Prob_Draw", "Prob_AwayWin"]].max(axis=1)
    df = df.sort_values(by="Max_Prob", ascending=False)

    return render_template("index.html", predictions=df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
