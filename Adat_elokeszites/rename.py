import os
import glob

def rename_images():
    folder = 'images'
    # Megkeressük az összes png és jpg fájlt
    images = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
    
    for index, img_path in enumerate(images, start=1):
        # Az új név garantáltan .png lesz
        new_path = os.path.join(folder, f"{index}.png")
        
        # Ha már eleve ez a neve, nem piszkáljuk
        if img_path != new_path:
            os.rename(img_path, new_path)
            
    print(f"Kész! {len(images)} darab kép átnevezve 1.png-től kezdődően.")

if __name__ == '__main__':
    rename_images()