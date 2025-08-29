import cv2
import os
from tkinter import Tk, filedialog

def main():
    # Nascondo la finestra principale di Tkinter
    root = Tk()
    root.withdraw()

    # Finestra per scegliere il file video
    video_path = filedialog.askopenfilename(
        title="Seleziona un file video MP4",
        filetypes=[("File MP4", "*.mp4")]
    )

    if not video_path:
        print("Nessun file selezionato, uscita.")
        return

    # Ricavo directory e nome del file senza estensione
    base_dir = os.path.dirname(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]

    # Creo la cartella con lo stesso nome del video
    output_dir = os.path.join(base_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    # Apro il video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # fine del video

        frame_count += 1
        # Nome file progressivo
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"Estrazione completata. {frame_count} frame salvati in: {output_dir}")

if __name__ == "__main__":
    main()
