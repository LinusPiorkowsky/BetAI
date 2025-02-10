import requests
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta

# Basisverzeichnis für gespeicherte Dateien
SAVE_DIR = "data/2024_25"
FIXTURES_DIR = "data/fixtures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIXTURES_DIR, exist_ok=True)

def download_dataset(url, filename):
    """Lädt einen ZIP-Datensatz herunter, entpackt ihn, verarbeitet die Daten und ersetzt die alte Datei."""
    filepath = os.path.join(SAVE_DIR, filename)
    temp_filename = filepath + ".tmp.zip"

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
        
        print(f"✅ Datensatz erfolgreich heruntergeladen und entpackt: {SAVE_DIR}")

    except requests.RequestException as e:
        print(f"❌ Fehler beim Herunterladen des Datensatzes: {e}")
    except zipfile.BadZipFile:
        print("❌ Fehler: Die heruntergeladene Datei ist keine gültige ZIP-Datei.")

def download_fixtures(url, filename):
    """Lädt die Spielansetzungen als CSV herunter und speichert sie direkt in 'data/fixtures.csv'."""
    filepath = os.path.join(FIXTURES_DIR, filename)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Prüfen, ob die Datei nicht leer ist
        if not response.content:
            print("❌ Fehler: Die heruntergeladene Datei ist leer.")
            return

        # Speichern ohne Modifikationen
        with open(filepath, 'wb') as file:
            file.write(response.content)
        
        print(f"✅ Fixtures erfolgreich heruntergeladen und gespeichert: {filepath}")

    except requests.RequestException as e:
        print(f"❌ Fehler beim Herunterladen der Spielansetzungen: {e}")

# Aufruf der Funktionen
download_dataset('https://www.football-data.co.uk/mmz4281/2425/data.zip', 'dataset.csv')
download_fixtures('https://www.football-data.co.uk/fixtures.csv', 'fixtures.csv')
