import requests
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta

# Basisverzeichnis für gespeicherte Dateien
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_dataset(url, filename):
    """Lädt einen ZIP-Datensatz herunter, entpackt ihn, verarbeitet die Daten und ersetzt die alte Datei."""
    filepath = os.path.join(SAVE_DIR, filename)
    temp_filename = filepath + ".tmp.zip"
    prepared_filepath = os.path.join(SAVE_DIR, "dataset_prepared.csv")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Prüfen, ob die Datei nicht leer ist
        if not response.content:
            print("Fehler: Der heruntergeladene Datensatz ist leer.")
            return

        # Temporär speichern
        with open(temp_filename, 'wb') as file:
            file.write(response.content)

        # Entpacken und alten Datensatz ersetzen
        extract_path = os.path.splitext(filepath)[0]  # Ordnername ohne .zip
        if os.path.exists(extract_path):
            os.system(f'rm -r {extract_path}')  # Löscht alten Ordner rekursiv
        
        with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
            zip_ref.extractall(SAVE_DIR)
        
        os.remove(temp_filename)  # Entfernt die temporäre ZIP-Datei
        
        # Annahme: Die entpackte Datei ist eine CSV-Datei
        csv_filename = [f for f in os.listdir(SAVE_DIR) if f.endswith(".csv")][0]
        csv_filepath = os.path.join(SAVE_DIR, csv_filename)
    

    except requests.RequestException as e:
        print(f"Fehler beim Herunterladen des Datensatzes: {e}")
    except zipfile.BadZipFile:
        print("Fehler: Die heruntergeladene Datei ist keine gültige ZIP-Datei.")

def download_fixtures(url, filename):
    """Lädt die Spielansetzungen als CSV herunter, verarbeitet sie und speichert nur relevante Spalten."""
    temp_filename = os.path.join(SAVE_DIR, filename + ".tmp")
    prepared_filepath = os.path.join(SAVE_DIR, "fixtures_prepared.csv")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Prüfen, ob die Datei nicht leer ist
        if not response.content:
            print("Fehler: Die heruntergeladene Datei ist leer.")
            return

        # Temporär speichern
        with open(temp_filename, 'wb') as file:
            file.write(response.content)
        
        # Daten vorbereiten
        prepare_fixtures(temp_filename, prepared_filepath)
        os.remove(temp_filename)  # Entfernt die temporäre Datei nach Verarbeitung

    except requests.RequestException as e:
        print(f"Fehler beim Herunterladen der Spielansetzungen: {e}")

def prepare_fixtures(input_filepath, output_filepath):
    """Bereitet die heruntergeladene CSV-Datei auf und speichert nur relevante Spalten."""
    try:
        df = pd.read_csv(input_filepath)
        
        # Relevante Spalten auswählen
        columns_needed = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"]
        df = df[columns_needed]
        
        # Zeit um +1 Stunde anpassen
        df["Time"] = df["Time"].apply(lambda x: (datetime.strptime(x, "%H:%M") + timedelta(hours=1)).strftime("%H:%M"))
        
        # Gesäuberte Datei speichern
        df.to_csv(output_filepath, index=False)
        print(f"Fixtures erfolgreich vorbereitet und gespeichert: {output_filepath}")

    except Exception as e:
        print(f"Fehler bei der Datenaufbereitung: {e}")

download_dataset('https://www.football-data.co.uk/mmz4281/2425/data.zip', 'dataset.csv')
download_fixtures('https://www.football-data.co.uk/fixtures.csv', 'fixtures.csv')
