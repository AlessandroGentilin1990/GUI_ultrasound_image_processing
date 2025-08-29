import requests

# URL del file raw su GitHub (sostituisci branch/file se serve)
url = "https://raw.githubusercontent.com/AlessandroGentilin1990/GUI_ultrasound_image_processing/main/ultrasound_roi_gui.py"

try:
    # Scarica il contenuto del file
    response = requests.get(url)
    response.raise_for_status()  # controlla eventuali errori HTTP

    code = response.text

    # Esegue il codice direttamente in memoria
    exec(code, globals())

except requests.exceptions.RequestException as e:
    print(f"Errore nel download del file: {e}")
except Exception as e:
    print(f"Errore durante l'esecuzione del codice: {e}")
