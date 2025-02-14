import requests
import os
import zipfile
import pandas as pd
import io  

# Basisverzeichnisse für gespeicherte Dateien
SAVE_DIR = "data/2024_25"
FIXTURES_DIR = "data/fixtures"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIXTURES_DIR, exist_ok=True)

# Erlaubte Ligen
ALLOWED_LEAGUES = {"D1.csv", "D2.csv", "E0.csv", "E1.csv", "F1.csv", "F2.csv", "I1.csv", "I2.csv", "SP1.csv", "SP2.csv"}

def download_dataset(url, filename):
    """Lädt einen ZIP-Datensatz herunter, entpackt nur die relevanten Ligen und ersetzt alte Dateien."""
    temp_zip_path = os.path.join(SAVE_DIR, "temp_dataset.zip")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Prüfen, ob die Datei nicht leer ist
        if not response.content:
            print("❌ Fehler: Der heruntergeladene Datensatz ist leer.")
            return

        # ZIP-Datei temporär speichern
        with open(temp_zip_path, 'wb') as file:
            file.write(response.content)

        # ZIP-Datei entpacken und nur relevante Ligen speichern
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            
            # Filtere nur die erlaubten Dateien
            for file in extracted_files:
                if any(file.endswith(league) for league in ALLOWED_LEAGUES):
                    zip_ref.extract(file, SAVE_DIR)
                    # print(f"✅ Gespeichert: {file}")
                # else:
                    # print(f"❌ Übersprungen: {file}")  # Nicht erlaubte Datei wird ignoriert

        # Entferne die temporäre ZIP-Datei
        os.remove(temp_zip_path)
        print(f"✅ Datensatz erfolgreich heruntergeladen und entpackt: {SAVE_DIR}")

    except requests.RequestException as e:
        print(f"❌ Fehler beim Herunterladen des Datensatzes: {e}")
    except zipfile.BadZipFile:
        print("❌ Fehler: Die heruntergeladene Datei ist keine gültige ZIP-Datei.")

def download_fixtures(url, filename):
    """Lädt die Spielansetzungen als CSV herunter, filtert nur die gewünschten Ligen und speichert sie."""
    filepath = os.path.join(FIXTURES_DIR, filename)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Prüfen, ob die Datei nicht leer ist
        if not response.content:
            print("❌ Fehler: Die heruntergeladene Datei ist leer.")
            return

        # ✅ CSV-Daten einlesen mit UTF-8 und BOM entfernen
        df_fixtures = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")

        # 🛠 **Spaltennamen bereinigen**
        df_fixtures.columns = df_fixtures.columns.str.replace("ï»¿", "", regex=True)

        # 🎯 Falls "Div" trotzdem nicht vorhanden ist, abbrechen
        if "Div" not in df_fixtures.columns:
            raise ValueError(f"❌ Fehler: Spalte 'Div' fehlt weiterhin. Verfügbare Spalten: {df_fixtures.columns.tolist()}")

        # 🎯 Nur die gewünschten Ligen behalten
        allowed_leagues = ["D1", "E0", "F1", "I1", "SP1"]
        df_fixtures = df_fixtures[df_fixtures["Div"].isin(allowed_leagues)]
        # 🎯 Nur die gewünschten Spalten behalten
        columns_to_keep = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]
        df_fixtures = df_fixtures[columns_to_keep]

        # Speichern
        df_fixtures.to_csv(filepath, index=False)
        
        print(f"✅ Fixtures erfolgreich gefiltert und gespeichert: {filepath}")

    except requests.RequestException as e:
        print(f"❌ Fehler beim Herunterladen der Spielansetzungen: {e}")
    except Exception as e:
        print(f"❌ Fehler bei der Verarbeitung der Fixtures: {e}")

# Aufruf der Funktion
download_dataset('https://www.football-data.co.uk/mmz4281/2425/data.zip', 'dataset.csv')
download_fixtures('https://www.football-data.co.uk/fixtures.csv', 'fixtures.csv')
