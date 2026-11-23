import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from model import CarcassonneClassifier

def evaluate_test_set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Betöltjük az elmentett modellt
    model = CarcassonneClassifier().to(device)
    model.load_state_dict(torch.load('simple_model_scanned.pth', map_location=device, weights_only=True))
    model.eval()

    # 2. Pontosan ugyanazt a 20%-ot választjuk le a random_state=42 segítségével
    df = pd.read_csv('labels.csv')
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index).reset_index(drop=True)

    print(f"Kiértékelés a független teszthalmazon ({len(test_df)} kép)...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_true_edges = []
    all_pred_edges = []
    full_card_correct = 0

    classes = ['Mezo', 'Ut', 'Varos', 'Folyo']

    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = f"images/{row['image_name']}"
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)

            p_top, p_right, p_bottom, p_left = model(tensor)

            pred = [
                torch.argmax(p_top, dim=1).item(),
                torch.argmax(p_right, dim=1).item(),
                torch.argmax(p_bottom, dim=1).item(),
                torch.argmax(p_left, dim=1).item()
            ]

            actual = [int(row['top']), int(row['right']), int(row['bottom']), int(row['left'])]

            all_pred_edges.extend(pred)
            all_true_edges.extend(actual)

            if pred == actual:
                full_card_correct += 1

    # 3. Számszerű statisztikák
    edge_acc = (np.array(all_pred_edges) == np.array(all_true_edges)).mean() * 100
    card_acc = (full_card_correct / len(test_df)) * 100

    print("\n--- EREDMÉNYEK (SZKENNELT TESZTHALMAZ) ---")
    print(f"Teljes kártya pontosság (mind a 4 oldal stimmel): {card_acc:.2f}% ({full_card_correct}/{len(test_df)})")
    print(f"Összesített él pontosság: {edge_acc:.2f}%")
    print("\nRészletes osztály-szintű kimutatás:")
    print(classification_report(all_true_edges, all_pred_edges, target_names=classes, zero_division=0))

    # 4. Konfúziós mátrix kirajzolása és mentése
    cm = confusion_matrix(all_true_edges, all_pred_edges)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.xlabel('Prediktált (Modell tippje)')
    plt.ylabel('Valós címke')

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    plt.title('Konfúziós Mátrix', pad=20)
    plt.tight_layout()
    plt.savefig('confusion_matrix_buta_scannelt.png', dpi=300)
    print("A hibamátrix elmentve 'confusion_matrix.png' néven a mappába!")

if __name__ == '__main__':
    evaluate_test_set()