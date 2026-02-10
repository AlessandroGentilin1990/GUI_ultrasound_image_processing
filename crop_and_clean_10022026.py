# pip install pillow opencv-python numpy matplotlib
import tkinter as tk
from tkinter import filedialog, Frame, Button, Label, Tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------
# FINESTRA DI CROP
# -------------------
WINDOW_W = 900
WINDOW_H = 600
DOT_R = 4

class CropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Crop")

        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.btn_close = tk.Button(
            root,
            text="✕",
            command=self.close_app,
            bg="red",
            fg="white",
            font=("Arial", 12, "bold"),
            bd=0,
            padx=10,
            pady=5
        )
        self.btn_close.place(relx=1.0, x=-10, y=10, anchor="ne")

        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack()

        self.btn_crop = tk.Button(root, text="Crop & Postprocess", command=self.crop)
        self.btn_crop.pack(pady=5)

        self.image_original = None
        self.image_display = None
        self.tk_img = None
        self.scale = 1.0
        self.points = []
        self.rect = None
        self.dots = []

        self.load_image()
        self.canvas.bind("<Button-1>", self.click)

    def close_app(self):
        self.root.destroy()

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            self.root.destroy()
            return

        self.path = path
        self.image_original = Image.open(path)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight() - 100

        ow, oh = self.image_original.size
        scale_w = screen_w / ow
        scale_h = screen_h / oh
        self.scale = min(scale_w, scale_h, 1)

        nw, nh = int(ow * self.scale), int(oh * self.scale)
        self.image_display = self.image_original.resize((nw, nh), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(self.image_display)

        self.canvas.config(width=nw, height=nh)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def click(self, event):
        if len(self.points) == 2:
            self.canvas.delete(self.rect)
            for d in self.dots:
                self.canvas.delete(d)
            self.points = []
            self.dots = []

        self.points.append((event.x, event.y))

        dot = self.canvas.create_oval(
            event.x - DOT_R, event.y - DOT_R,
            event.x + DOT_R, event.y + DOT_R,
            fill="red", outline=""
        )
        self.dots.append(dot)

        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            self.rect = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="red", width=2
            )

    def crop(self):
        if len(self.points) != 2:
            return

        x1, y1 = self.points[0]
        x2, y2 = self.points[1]

        left = int(min(x1, x2) / self.scale)
        top = int(min(y1, y2) / self.scale)
        right = int(max(x1, x2) / self.scale)
        bottom = int(max(y1, y2) / self.scale)

        cropped = self.image_original.crop((left, top, right, bottom))
        self.image_original = cropped
        self.root.destroy()  # chiude il crop e apre la finestra di post-processing

        PostProcessApp(cropped)  # apre la seconda finestra per i filtri

# -------------------
# FINESTRA DI POST-PROCESSING
# -------------------
class PostProcessApp:
    def __init__(self, pil_image):
        self.root = Tk()
        self.root.title("Cleaning and Post-processing")
        self.root.geometry("1100x650")

        self.original_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        self.img = self.original_img.copy()

        Label(self.root,
              text="Doppler Image Cleanup & Envelope Reconstruction",
              font=("Helvetica", 12, "bold")).pack(pady=5)

        viewer_frame = Frame(self.root)
        viewer_frame.pack(fill="both", expand=True)
        self.init_viewer(viewer_frame)

        controls = Frame(self.root)
        controls.pack(pady=5)

        Button(controls, text="Reset", command=self.reset_image).grid(row=0, column=0, padx=4)
        Button(controls, text="Save", command=self.save_image).grid(row=0, column=1, padx=4)

        Button(controls, text="Median", command=self.apply_median_filter).grid(row=1, column=0, padx=4)
        Button(controls, text="Gaussian", command=self.apply_gaussian_filter).grid(row=1, column=1, padx=4)
        Button(controls, text="Bilateral", command=self.apply_bilateral_filter).grid(row=1, column=2, padx=4)
        Button(controls, text="Contrast", command=self.enhance_contrast).grid(row=1, column=3, padx=4)
        Button(controls, text="Threshold", command=self.apply_threshold).grid(row=1, column=4, padx=4)
        Button(controls, text="Morphology", command=self.apply_morphology).grid(row=1, column=5, padx=4)
        Button(controls, text="Fill Envelope", command=self.fill_envelope_holes).grid(row=1, column=6, padx=4)

        self.update_viewer(self.img, "Original Image")
        self.root.mainloop()

    # -------------------
    # VISUALIZZATORE MATPLOTLIB
    # -------------------
    def init_viewer(self, parent):
        self.fig = Figure(figsize=(10, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.img_plot = self.ax.imshow(np.zeros((10, 10)), cmap="gray")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_viewer(self, image, title=""):
        self.img_plot.set_data(image)
        self.img_plot.set_clim(vmin=image.min(), vmax=image.max())
        self.ax.set_title(title)
        self.canvas.draw_idle()

    # -------------------
    # FUNZIONI BASE
    # -------------------
    def reset_image(self):
        self.img = self.original_img.copy()
        self.update_viewer(self.img, "Image restored")

    def save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if path:
            cv2.imwrite(path, self.img)

    # -------------------
    # FILTRI
    # -------------------
    def apply_median_filter(self):
        self.img = cv2.medianBlur(self.img, 3)
        self.update_viewer(self.img, "Median Filter")

    def apply_gaussian_filter(self):
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        self.update_viewer(self.img, "Gaussian Filter")

    def apply_bilateral_filter(self):
        self.img = cv2.bilateralFilter(self.img, 9, 75, 75)
        self.update_viewer(self.img, "Bilateral Filter")

    def enhance_contrast(self):
        self.img = cv2.equalizeHist(self.img)
        self.update_viewer(self.img, "Improved contrast")

    def apply_threshold(self):
        _, self.img = cv2.threshold(self.img, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.update_viewer(self.img, "Threshold (Otsu)")

    def apply_morphology(self):
        kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        self.update_viewer(self.img, "Morphological cleaning")

    def fill_envelope_holes(self):
        if len(np.unique(self.img)) > 2:
            _, self.img = cv2.threshold(self.img, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h, w = self.img.shape
        flood = self.img.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        self.img = self.img | flood_inv

        self.update_viewer(self.img, "Reconstructed envelope")


if __name__ == "__main__":
    root = tk.Tk()
    CropApp(root)
    root.mainloop()
