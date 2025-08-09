# app.py - Main Flask Application
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Configuration for PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTION_DIR = os.path.join(BASE_DIR, "predictions")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create directories if they don't exist
for directory in [PREDICTION_DIR, RESULT_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mapping of league codes to their names
LEAGUE_NAMES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "I1": "Serie A",
    "SP1": "La Liga",
    "E1": "Championship",
    "D2": "2. Bundesliga",
    "F2": "Ligue 2",
    "I2": "Serie B",
    "SP2": "Segunda División"
}

def get_latest_prediction():
    """Find the latest prediction CSV file and load it into a DataFrame."""
    try:
        prediction_files = sorted(
            [f for f in os.listdir(PREDICTION_DIR)
             if f.startswith("predictions_") and f.endswith(".csv")],
            key=lambda x: int(''.join(filter(str.isdigit, x.replace("predictions_", ""))))
                          if any(char.isdigit() for char in x) else 0,
            reverse=True
        )

        if not prediction_files:
            logger.warning("No prediction files found")
            return None

        latest_file = os.path.join(PREDICTION_DIR, prediction_files[0])
        df = pd.read_csv(latest_file)

        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors='coerce')

        # Filter out matches that started more than 2 hours ago
        now = datetime.now()
        if "Time" in df.columns:
            df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"], errors='coerce')
            df = df[df["DateTime"] + timedelta(hours=2) > now]

        # Capitalize team names
        for col in ["HomeTeam", "AwayTeam"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ' '.join([word.capitalize() for word in str(x).split()]))

        return df.sort_values(["Date", "Time"]) if not df.empty else None

    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return None

def load_all_result_files():
    """Reads ALL result_{i}.csv files in the results directory into a single DataFrame."""
    try:
        pattern = os.path.join(RESULT_DIR, "result_*.csv")
        file_list = glob.glob(pattern)

        if not file_list:
            logger.warning("No result files found")
            return None

        frames = []
        for file_path in file_list:
            df_temp = pd.read_csv(file_path)
            df_temp["Date"] = pd.to_datetime(df_temp["Date"], format="%Y-%m-%d", errors="coerce")

            # Capitalize team names
            for col in ["HomeTeam", "AwayTeam"]:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].apply(
                        lambda x: ' '.join([word.capitalize() for word in str(x).split()])
                    )

            frames.append(df_temp)

        all_results = pd.concat(frames, ignore_index=True)
        all_results.sort_values(by=["Date", "Time"], inplace=True)
        all_results.reset_index(drop=True, inplace=True)

        return all_results if not all_results.empty else None

    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

def get_final_bet(row):
    """Enhanced betting logic based on prediction and probabilities."""
    prediction = row.get("Prediction", "")
    prob_h = row.get("Prob_H", 0.0)
    prob_d = row.get("Prob_D", 0.0)
    prob_a = row.get("Prob_A", 0.0)
    b365h = row.get("B365H", 0.0)
    b365d = row.get("B365D", 0.0)
    b365a = row.get("B365A", 0.0)

    if prediction == "H":
        # More conservative approach for home wins
        if prob_h <= 0.60 or b365h > 2.5:
            return "1X"
        return "H"
    elif prediction == "D":
        # Choose based on odds value
        if b365h < b365a:
            return "1X"
        else:
            return "X2"
    elif prediction == "A":
        # More conservative for away wins
        if b365a < 2.0 and prob_a > 0.70:
            return "A"
        else:
            return "X2"

    return prediction

def is_bet_correct(actual_result, final_bet):
    """Check if the bet was correct based on actual result."""
    if pd.isna(actual_result) or pd.isna(final_bet):
        return False
    
    actual_result = str(actual_result).upper()
    final_bet = str(final_bet).upper()
    
    if actual_result == "H":
        return final_bet in ["H", "1X"]
    elif actual_result == "D":
        return final_bet in ["D", "1X", "X2"]
    elif actual_result == "A":
        return final_bet in ["A", "X2"]
    return False

def calculate_statistics(df):
    """Calculate comprehensive betting statistics."""
    if df.empty:
        return {
            'total_bets': 0, 'correct_bets': 0, 'accuracy': 0.0,
            'high_conf_bets': 0, 'high_conf_correct': 0, 'high_conf_accuracy': 0.0
        }

    # Apply betting logic
    df["FinalBet"] = df.apply(get_final_bet, axis=1)
    df["BetCorrect"] = df.apply(
        lambda row: is_bet_correct(row.get("Actual_Result"), row.get("FinalBet")),
        axis=1
    )

    # Overall statistics
    total_bets = len(df)
    correct_bets = df["BetCorrect"].sum()
    accuracy = round(correct_bets / total_bets * 100, 2) if total_bets > 0 else 0.0

    # High confidence statistics
    high_conf_df = df[df.get("High_conf", 0) == 1]
    high_conf_bets = len(high_conf_df)
    high_conf_correct = high_conf_df["BetCorrect"].sum() if not high_conf_df.empty else 0
    high_conf_accuracy = round(high_conf_correct / high_conf_bets * 100, 2) if high_conf_bets > 0 else 0.0

    return {
        'total_bets': total_bets,
        'correct_bets': correct_bets,
        'accuracy': accuracy,
        'high_conf_bets': high_conf_bets,
        'high_conf_correct': high_conf_correct,
        'high_conf_accuracy': high_conf_accuracy
    }

@app.route("/")
def index():
    """Dashboard with overall statistics."""
    try:
        all_results = load_all_result_files()
        
        if all_results is None or all_results.empty:
            return render_template("index.html", stats={
                'overall': calculate_statistics(pd.DataFrame()),
                'last_week': calculate_statistics(pd.DataFrame())
            })

        # Overall statistics
        overall_stats = calculate_statistics(all_results)

        # Last week statistics
        today = pd.Timestamp.now().normalize()
        current_year, current_week, _ = today.isocalendar()
        
        all_results["Year"] = all_results["Date"].dt.isocalendar().year
        all_results["Week"] = all_results["Date"].dt.isocalendar().week

        last_week_year = current_year if current_week > 1 else current_year - 1
        last_week_number = current_week - 1 if current_week > 1 else 52

        last_week_df = all_results[
            (all_results["Year"] == last_week_year) & 
            (all_results["Week"] == last_week_number)
        ]
        
        last_week_stats = calculate_statistics(last_week_df)

        return render_template("index.html", stats={
            'overall': overall_stats,
            'last_week': last_week_stats
        })

    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template("error.html", error="Failed to load dashboard")

@app.route("/predictions")
def predictions():
    """Display upcoming predictions with filtering options."""
    try:
        df = get_latest_prediction()
        
        # Get filter parameters
        selected_league = request.args.get('league', '')
        selected_confidence = request.args.get('confidence', '')
        selected_date = request.args.get('date', '')

        if df is None or df.empty:
            return render_template("predictions.html", 
                                 predictions=[], 
                                 leagues=[], 
                                 dates=[],
                                 league_names=LEAGUE_NAMES,
                                 filters={'league': selected_league, 'confidence': selected_confidence, 'date': selected_date})

        # Apply filters
        filtered_df = df.copy()
        
        if selected_league:
            filtered_df = filtered_df[filtered_df["Div"] == selected_league]
        
        if selected_confidence == 'high':
            filtered_df = filtered_df[filtered_df.get("High_conf", 0) == 1]
        
        if selected_date:
            selected_date_dt = pd.to_datetime(selected_date)
            filtered_df = filtered_df[filtered_df["Date"].dt.date == selected_date_dt.date()]

        # Prepare data for template
        predictions_list = filtered_df.to_dict(orient="records")
        leagues = sorted(df["Div"].unique().tolist())
        dates = sorted(df["Date"].dt.strftime('%Y-%m-%d').unique().tolist())

        return render_template("predictions.html",
                             predictions=predictions_list,
                             leagues=leagues,
                             dates=dates,
                             league_names=LEAGUE_NAMES,
                             filters={'league': selected_league, 'confidence': selected_confidence, 'date': selected_date})

    except Exception as e:
        logger.error(f"Error in predictions route: {e}")
        return render_template("error.html", error="Failed to load predictions")

@app.route("/results")
def results():
    """Display past results with comprehensive filtering."""
    try:
        # Get filter parameters
        selected_league = request.args.get('league', '')
        selected_week = request.args.get('week', '')
        selected_month = request.args.get('month', '')
        high_conf_only = request.args.get('high_conf', '') == 'true'
        correct_only = request.args.get('correct_only', '') == 'true'

        df = load_all_result_files()
        if df is None or df.empty:
            return render_template("results.html", 
                                 results=[], 
                                 leagues=[], 
                                 weeks=[], 
                                 months=[],
                                 stats=calculate_statistics(pd.DataFrame()),
                                 league_names=LEAGUE_NAMES)

        # Apply betting logic
        df["FinalBet"] = df.apply(get_final_bet, axis=1)
        df["BetCorrect"] = df.apply(
            lambda row: is_bet_correct(row.get("Actual_Result"), row.get("FinalBet")),
            axis=1
        )

        # Create additional columns for filtering
        df["Week"] = df["Date"].dt.isocalendar().week
        df["Year"] = df["Date"].dt.isocalendar().year
        df["Month"] = df["Date"].dt.strftime('%Y-%m')

        # Apply filters
        filtered_df = df.copy()

        if selected_league:
            filtered_df = filtered_df[filtered_df["Div"] == selected_league]

        if selected_week and selected_week.isdigit():
            filtered_df = filtered_df[filtered_df["Week"] == int(selected_week)]

        if selected_month:
            filtered_df = filtered_df[filtered_df["Month"] == selected_month]

        if high_conf_only:
            filtered_df = filtered_df[filtered_df.get("High_conf", 0) == 1]

        if correct_only:
            filtered_df = filtered_df[filtered_df["BetCorrect"] == True]

        # Calculate statistics for filtered data
        stats = calculate_statistics(filtered_df)

        # Sort by date (most recent first)
        filtered_df = filtered_df.sort_values(by=["Date", "Time"], ascending=False)

        # Prepare filter options
        leagues = sorted(df["Div"].unique().tolist())
        weeks = sorted(df["Week"].unique().tolist())
        months = sorted(df["Month"].unique().tolist(), reverse=True)

        return render_template("results.html",
                             results=filtered_df.to_dict(orient="records"),
                             leagues=leagues,
                             weeks=weeks,
                             months=months,
                             stats=stats,
                             league_names=LEAGUE_NAMES,
                             filters={
                                 'league': selected_league,
                                 'week': selected_week,
                                 'month': selected_month,
                                 'high_conf': high_conf_only,
                                 'correct_only': correct_only
                             })

    except Exception as e:
        logger.error(f"Error in results route: {e}")
        return render_template("error.html", error="Failed to load results")

@app.route("/api/stats")
def api_stats():
    """API endpoint for statistics (for AJAX updates)."""
    try:
        all_results = load_all_result_files()
        if all_results is None or all_results.empty:
            return jsonify({'error': 'No data available'})

        overall_stats = calculate_statistics(all_results)
        return jsonify(overall_stats)

    except Exception as e:
        logger.error(f"Error in API stats: {e}")
        return jsonify({'error': str(e)})

@app.route("/about")
def about():
    """Information page about the ML models and methodology."""
    return render_template("about.html")

@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="Internal server error"), 500

if __name__ == "__main__":
    # For PythonAnywhere, this won't be used, but useful for local testing
    app.run(debug=False)