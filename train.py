import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import CarcassonneClassifier

class CarcassonneDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Kiolvassuk a 4 címkét: [felső, jobb, alsó, bal]
        labels = self.data.iloc[idx][['top', 'right', 'bottom', 'left']].values.astype('int64')
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

def main():
    # 1. Adatok betöltése és felosztása (80% tanító, 20% tesztelő)
    df = pd.read_csv('labels_phone.csv')
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    print(f"Összes kép: {len(df)}")
    print(f"Tanító halmaz: {len(train_df)} kép")
    print(f"Tesztelő halmaz: {len(test_df)} kép")

    # 2. Kép átalakítások (A PyTorch szabványos 224x224-es méretre húzza az összes képet)
    transform = transforms.Compose([
        #kép méretezése a hivatalos mértteé, amit a reshnet elvár
        transforms.Resize((224, 224)),
        #0-255 RGB színek 0-1 közötti értékre 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CarcassonneDataset(train_df, img_dir='phone_images', transform=transform)
    test_dataset = CarcassonneDataset(test_df, img_dir='phone_images', transform=transform)

    # Batch size: egyszerre 16 képet adunk a hálónak (felgyorsítja a tanítást)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 3. Modell beállítása
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nHasznált eszköz: {device}")
    
    model = CarcassonneClassifier().to(device)
    #crossE.. többosztáloy klasszifikációs feladatokhoz, matematikailag ez a legjobb 
    #osztályozó hiba
    criterion = nn.CrossEntropyLoss()
    #Adam optimalizáció -- gyorsan és stabilan tanul 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Tanítási ciklus
    epochs = 10
    print("\nTanítás indul...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() #1.nullázás

            # a háló tippel
            p_top, p_right, p_bottom, p_left = model(images)
            
            # A 4 oldal hibájának összegzése, mind a 4 oldalra külön-külön
            loss = (criterion(p_top, labels[:, 0]) +
                    criterion(p_right, labels[:, 1]) +
                    criterion(p_bottom, labels[:, 2]) +
                    criterion(p_left, labels[:, 3]))

            
            loss.backward() #2.visszterjesztés -- ez csinálj amagát a tanulást 
            optimizer.step() #3.súlyok frissítése
            running_loss += loss.item() # összeadjuk ezeket a hibákat 
            
        # Validáció: teszteljük a modellt az ismeretlen (20%) képeken
        model.eval()
        correct_edges = 0
        total_edges = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                p_top, p_right, p_bottom, p_left = model(images)
                
                # Kiválasztjuk a legmagasabb pontszámú tippet (0, 1, 2 vagy 3)
                pred_top = torch.argmax(p_top, dim=1)
                pred_right = torch.argmax(p_right, dim=1)
                pred_bottom = torch.argmax(p_bottom, dim=1)
                pred_left = torch.argmax(p_left, dim=1)
                
                # Megszámoljuk, hány élt talált el pontosan
                correct_edges += (pred_top == labels[:, 0]).sum().item()
                correct_edges += (pred_right == labels[:, 1]).sum().item()
                correct_edges += (pred_bottom == labels[:, 2]).sum().item()
                correct_edges += (pred_left == labels[:, 3]).sum().item()
                
                total_edges += labels.size(0) * 4
        
        val_acc = (correct_edges / total_edges) * 100
        print(f"Korszak {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Teszt él-pontosság: {val_acc:.2f}%")

    # 5. A betanult tudás mentése
    torch.save(model.state_dict(), 'carcassonne_phone_model.pth')
    print("\nA modell sikeresen elmentve: carcassonne_phone_model.pth")

if __name__ == '__main__':
    main()