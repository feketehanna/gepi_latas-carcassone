import os
import glob
from PIL import Image

def rotate_images():
    folder = 'images'
    # Csak azokat a fájlokat olvassuk be, amikben nincs aláhúzás (tehát az alapképeket)
    images = [f for f in glob.glob(os.path.join(folder, '*.png')) if '_' not in os.path.basename(f)]
    
    count = 0
    for img_path in images:
        img = Image.open(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Elforgatjuk 90, 180 és 270 fokkal az óramutató járásával megegyezően
        for angle in [90, 180, 270]:
            # A mínusz jel miatt forog jobbra (óramutató szerint)
            rotated_img = img.rotate(-angle, expand=True)
            
            new_name = f"{base_name}_{angle}.png"
            rotated_img.save(os.path.join(folder, new_name))
            count += 1
            
    print(f"Kész! {count} darab új, elforgatott kép jött létre az images mappában.")

if __name__ == '__main__':
    rotate_images()