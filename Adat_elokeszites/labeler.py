import os
import glob
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

CSV_FILE = 'labels_phone.csv'
IMG_DIR = './phone_images'

class CarcassonneLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Carcassonne Címkéző")
        self.root.geometry("400x550")
        
        # Korábbi címkék betöltése, hogy ne kelljen elölről kezdeni
        if os.path.exists(CSV_FILE):
            self.df = pd.read_csv(CSV_FILE)
            self.labeled_images = set(self.df['image_name'].tolist())
        else:
            self.df = pd.DataFrame(columns=['image_name', 'top', 'right', 'bottom', 'left'])
            self.labeled_images = set()

        # Csak azokat a képeket töltjük be, amik még nincsenek a CSV-ben
        all_images = glob.glob(os.path.join(IMG_DIR, '*.png'))
        self.images_to_label = [os.path.basename(img) for img in all_images if os.path.basename(img) not in self.labeled_images]
        
        self.current_idx = 0

        # Felhasználói felület (UI)
        instructions = "Sorrend: FELSŐ, JOBB, ALSÓ, BAL\n\n0 = Mező\n1 = Út\n2 = Város\n3 = Folyó"
        self.info_label = tk.Label(root, text=instructions, font=("Arial", 12, "bold"), fg="blue")
        self.info_label.pack(pady=10)

        self.progress_label = tk.Label(root, text="", font=("Arial", 10))
        self.progress_label.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # Ide lehet gépelni a 4 számot (pl. "0123")
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Arial", 24), width=6, justify='center')
        self.entry.pack(pady=10)
        self.entry.bind('<Return>', self.save_label)
        self.entry.focus()

        self.load_image()

    def load_image(self):
        if self.current_idx >= len(self.images_to_label):
            messagebox.showinfo("Kész!", "Az összes képet felcímkézted!")
            self.root.quit()
            return

        img_name = self.images_to_label[self.current_idx]
        self.progress_label.config(text=f"Kép: {img_name} ({self.current_idx + 1} / {len(self.images_to_label)} hátralévő)")
        
        img_path = os.path.join(IMG_DIR, img_name)
        img = Image.open(img_path).resize((250, 250))
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)
        
        self.entry_var.set("")

    def save_label(self, event):
        val = self.entry_var.get().strip()
        
        # Ellenőrzés: Pontosan 4 karakter és csak 0, 1, 2, 3 lehet benne
        if len(val) != 4 or not all(c in '0123' for c in val):
            messagebox.showerror("Hiba", "Pontosan 4 számjegyet adj meg (0, 1, 2 vagy 3)!\nPélda: 0130")
            self.entry_var.set("")
            return

        img_name = self.images_to_label[self.current_idx]
        
        # Hozzáadjuk a táblázathoz és azonnal elmentjük
        new_row = pd.DataFrame([{
            'image_name': img_name,
            'top': int(val[0]),
            'right': int(val[1]),
            'bottom': int(val[2]),
            'left': int(val[3])
        }])
        
        # Pandas frissítések miatti figyelmeztetések elkerülése
        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            
        self.df.to_csv(CSV_FILE, index=False)

        self.current_idx += 1
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarcassonneLabeler(root)
    root.mainloop()
    