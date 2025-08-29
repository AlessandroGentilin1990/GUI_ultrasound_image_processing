import requests
import base64

# URL del file raw Base64 su GitHub
url = "https://raw.githubusercontent.com/AlessandroGentilin1990/GUI_ultrasound_image_processing/main/ultrasound_roi_gui.b64"

try:
    # Scarica il contenuto del file codificato
    response = requests.get(url)
    response.raise_for_status()  # controlla eventuali errori HTTP

    # Decodifica Base64 in memoria
    code_bytes = base64.b64decode(response.content)
    code_str = code_bytes.decode()  # converte da byte a stringa

    # Esegue il codice direttamente in memoria
    exec(code_str, globals())

except requests.exceptions.RequestException as e:
    print(f"Errore nel download del file: {e}")
except Exception as e:
    print(f"Errore durante l'esecuzione del codice: {e}")
