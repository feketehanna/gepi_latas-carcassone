import pandas as pd

def szabvanyos_kartyatipus(row):
    # Kimentjük a 4 élt egy listába
    elek = [str(row['top']), str(row['right']), str(row['bottom']), str(row['left'])]
    
    # Létrehozzuk a kártya mind a 4 lehetséges forgatását
    forgatas_1 = "-".join(elek)                                      # Eredeti
    forgatas_2 = "-".join([elek[3], elek[0], elek[1], elek[2]])      # 90 fok
    forgatas_3 = "-".join([elek[2], elek[3], elek[0], elek[1]])      # 180 fok
    forgatas_4 = "-".join([elek[1], elek[2], elek[3], elek[0]])      # 270 fok
    
    # A 4 verzió közül mindig a "legkisebbet" (ABC sorrendben az elsőt) adjuk vissza.
    # Így a forgatott verziók is pontosan ugyanazt az azonosítót kapják!
    return min(forgatas_1, forgatas_2, forgatas_3, forgatas_4)

# 1. Beolvassuk az adatokat
df = pd.read_csv('labels_phone.csv')

# 2. Minden sorra alkalmazzuk a szabványosító logikánkat
df['kartyatipus'] = df.apply(szabvanyos_kartyatipus, axis=1)

# 3. Megszámoljuk a gyakoriságot
statisztika = df['kartyatipus'].value_counts()

print("A leggyakoribb kártyatípusok (elforgatástól függetlenül):")
print(statisztika.head(15))

import matplotlib.pyplot as plt

# 1. Számok dekódolása betűkre a szebb feliratokért (0:Mező, 1:Út, 2:Város, 3:Folyó)
mapping = {'0': 'M', '1': 'Ú', '2': 'V', '3': 'F'}
top_kombinaciok = statisztika

szep_cimek = []
for index in top_kombinaciok.index:
    betuk = [mapping[szam] for szam in index.split('-')]
    szep_cimek.append("-".join(betuk))

ertekek = top_kombinaciok.values

# 2. A diagram megrajzolása
plt.figure(figsize=(18, 6))

# Oszlopok létrehozása (elegáns kék színnel, fekete kerettel)
bars = plt.bar(szep_cimek, ertekek, color="#7f22d0", edgecolor='black')

# Értékek (számok) ráírása pontosan az oszlopok tetejére
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, int(yval), 
             ha='center', va='bottom', fontweight='bold')

# 3. Formázás és feliratok
plt.title('A leggyakoribb egyedi kártyatípusok eloszlása a tanító adathalmazban\n(M = Mező, Ú = Út, V = Város, F = Folyó)', 
          fontsize=14, pad=15)
plt.xlabel('Kártya él-kombináció (forgatástól független szabványos alak)', fontsize=12)
plt.ylabel('Előfordulás (darab)', fontsize=12)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Vízszintes segédvonalak
plt.tight_layout()

# 4. Mentés fájlba
plt.savefig('kartyatipus_eloszlas.png', dpi=300)
print("\nA diagram elkészült és elmentve 'kartyatipus_eloszlas.png' néven!")