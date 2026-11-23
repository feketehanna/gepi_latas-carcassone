import sys
import os
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox
from torchvision import transforms
from PIL import Image, ImageTk
from model import CarcassonneClassifier

class CarcassonneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Carcassonne Elemző")
        self.root.geometry("650x400")
        
        # Modern, sötét színek és betűtípusok
        self.bg_color = "#121212"      # Nagyon sötét szürke/fekete
        self.panel_color = "#1e1e1e"   # Kicsit világosabb panel szín
        self.fg_color = "#ffffff"      # Fehér szöveg
        self.accent_color = "#3a3a3a"  # Gomb színe
        self.font_title = ("Segoe UI", 16, "bold")
        self.font_text = ("Segoe UI", 13)
        
        self.root.configure(bg=self.bg_color)
        
        # --- PyTorch Modell Inicializálása ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CarcassonneClassifier().to(self.device)
        self.classes = {0: 'Mező', 1: 'Út', 2: 'Város', 3: 'Folyó'}
        
        try:
            self.model.load_state_dict(torch.load('carcassonne_model.pth', map_location=self.device, weights_only=True))
            self.model.eval()
        except Exception as e:
            messagebox.showwarning("Modell hiba", f"Nem sikerült betölteni a modellt. Ellenőrizd a 'carcassonne_model.pth' fájlt!\n{e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.setup_ui()

    def setup_ui(self):
        # Fő konténer
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Bal oldali panel (Kép helye)
        self.image_frame = tk.Frame(main_frame, bg=self.panel_color, width=250, height=250, highlightbackground="#333", highlightthickness=1)
        self.image_frame.pack(side="left", padx=(0, 20))
        self.image_frame.pack_propagate(False) # Hogy ne zsugorodjon össze

        self.image_label = tk.Label(self.image_frame, text="Nincs kép betöltve", bg=self.panel_color, fg="#888888", font=self.font_text)
        self.image_label.pack(expand=True)

        # Jobb oldali panel (Eredmények)
        result_frame = tk.Frame(main_frame, bg=self.bg_color)
        result_frame.pack(side="left", fill="both", expand=True)

        title_label = tk.Label(result_frame, text="Elemzés Eredménye", bg=self.bg_color, fg=self.fg_color, font=self.font_title)
        title_label.pack(anchor="w", pady=(0, 20))

        # Eredmény kiíró labelek
        self.lbl_top = tk.Label(result_frame, text="Felső: -", bg=self.bg_color, fg=self.fg_color, font=self.font_text)
        self.lbl_top.pack(anchor="w", pady=2)
        
        self.lbl_right = tk.Label(result_frame, text="Jobb: -", bg=self.bg_color, fg=self.fg_color, font=self.font_text)
        self.lbl_right.pack(anchor="w", pady=2)
        
        self.lbl_bottom = tk.Label(result_frame, text="Alsó: -", bg=self.bg_color, fg=self.fg_color, font=self.font_text)
        self.lbl_bottom.pack(anchor="w", pady=2)
        
        self.lbl_left = tk.Label(result_frame, text="Bal: -", bg=self.bg_color, fg=self.fg_color, font=self.font_text)
        self.lbl_left.pack(anchor="w", pady=2)

        # Gomb a fájl betöltéséhez
        self.btn_load = tk.Button(result_frame, text="Kép betöltése...", bg=self.accent_color, fg=self.fg_color, 
                                  font=self.font_text, relief="flat", activebackground="#555555", 
                                  activeforeground="white", cursor="hand2", command=self.load_and_predict)
        self.btn_load.pack(anchor="w", pady=(30, 0), ipadx=10, ipady=5)

    def load_and_predict(self):
        # Fájl tallózó ablak
        file_path = filedialog.askopenfilename(
            title="Válassz egy képet",
            filetypes=[("Képfájlok", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        
        if not file_path:
            return

        self._load_from_path(file_path)

    def _load_from_path(self, file_path):
        # Kép megjelenítése a felületen
        try:
            display_img = Image.open(file_path)
            display_img.thumbnail((240, 240)) # Átméretezés a panelhez, képarány megtartásával
            self.photo = ImageTk.PhotoImage(display_img)
            
            self.image_label.config(image=self.photo, text="")
        except Exception as e:
            messagebox.showerror("Hiba", f"Nem sikerült megnyitni a képet:\n{e}")
            return

        # AI Előrejelzés
        try:
            image = Image.open(file_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                p_top, p_right, p_bottom, p_left = self.model(image_tensor)
                
                # Softmax alkalmazása a valószínűségek (százalékok) kinyeréséhez
                prob_top = F.softmax(p_top, dim=1)[0]
                prob_right = F.softmax(p_right, dim=1)[0]
                prob_bottom = F.softmax(p_bottom, dim=1)[0]
                prob_left = F.softmax(p_left, dim=1)[0]

                # Legnagyobb valószínűségű osztály kiválasztása
                top = torch.argmax(p_top, dim=1).item()
                right = torch.argmax(p_right, dim=1).item()
                bottom = torch.argmax(p_bottom, dim=1).item()
                left = torch.argmax(p_left, dim=1).item()
                
                # A kiválasztott osztály százalékos magabiztossága
                conf_top = prob_top[top].item() * 100
                conf_right = prob_right[right].item() * 100
                conf_bottom = prob_bottom[bottom].item() * 100
                conf_left = prob_left[left].item() * 100

            # Eredmények frissítése a felületen, a százalékos értékek megjelenítésével
            self.lbl_top.config(text=f"Felső: {self.classes[top]} ({conf_top:.1f}%)")
            self.lbl_right.config(text=f"Jobb: {self.classes[right]} ({conf_right:.1f}%)")
            self.lbl_bottom.config(text=f"Alsó: {self.classes[bottom]} ({conf_bottom:.1f}%)")
            self.lbl_left.config(text=f"Bal: {self.classes[left]} ({conf_left:.1f}%)")

        except Exception as e:
            messagebox.showerror("Hiba", f"Hiba az elemzés során:\n{e}")


if __name__ == '__main__':
    # GUI indítása
    root = tk.Tk()
    app = CarcassonneGUI(root)
    
    # Ha parancssorból adtak meg képet (pl. py predict.py image.png), automatikusan betöltjük
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
        if os.path.exists(target_image):
            root.after(500, lambda: app._load_from_path(target_image))
            
    root.mainloop()     