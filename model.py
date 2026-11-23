import torch.nn as nn
from torchvision import models

class CarcassonneClassifier(nn.Module):
    def __init__(self):
        super(CarcassonneClassifier, self).__init__()
        #előtanított -- ne legyen súlyértéke 
        #mist -- egy ilyen hálózatot átalakíítani 
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # 4 kimeneti osztály: 0: Mező, 1: Út, 2: Város, 3: Folyó
        self.fc_top = nn.Linear(num_features, 4)
        self.fc_right = nn.Linear(num_features, 4)
        self.fc_bottom = nn.Linear(num_features, 4)
        self.fc_left = nn.Linear(num_features, 4)

    def forward(self, x):
        features = self.base_model(x)
        return (
            self.fc_top(features),
            self.fc_right(features),
            self.fc_bottom(features),
            self.fc_left(features)
        )

import torch.nn.functional as F

#bbutább csoda modell 
class SimpleCarcassonneCNN(nn.Module):
    def __init__(self):
        super(SimpleCarcassonneCNN, self).__init__()
        # 1. Konvolúciós blokk (Keresi az alap éleket és színeket)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)


        #minden convolució után képet megfelezzük(felbontásokat) 
        #--> felesleges zajokat kiszűri
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Konvolúciós blokk (Összetettebb textúrákat keres)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 3. Konvolúciós blokk
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # A 224x224-es kép 3 poololás után 28x28-as lesz (64 csatornával)
        # Kiszámoljuk, mennyi adat jön ki a 3. blokk végén (224x224-es bemenet esetén)
        #FONTOS HOGY 224*224-es legyen‼️‼️
        flattened_size = 64 * 28 * 28
        
        # A 4 döntéshozó fej (ugyanaz a logika, mint a ResNetnél)
        #ez már az egy dimenizós mátrixot kapta meg amit a .view csinált 
        #4 elemű kimenet --> melyikre mennyi esély van (mező, út, város, folyó)
        self.fc_top = nn.Linear(flattened_size, 4)
        self.fc_right = nn.Linear(flattened_size, 4)
        self.fc_bottom = nn.Linear(flattened_size, 4)
        self.fc_left = nn.Linear(flattened_size, 4)

    def forward(self, x):
        #conv1 --legegyszerűbb színátmeneteket, mintákat ismeri fel (16 különb. digitális szűrő segítségével )
        #F.relu --> negatív számokat 0-ra, többit békén hagyja --> ne csak lineárisan tudjon gondolkodni 
        x = self.pool(F.relu(self.conv1(x))) 
        #mélyebb rétegek --> conv1 vonaiaból rak össze bonyolultabb formákat, textúrákat 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        #conv2 -- Kiterítjük a térbeli adatokat egy egydimenziós vektorrá
        #(lényegében egy nagy egy dimenziós listává )
        #3. conv blokk után történik ez meg miután mindegyik kép meg van 
        features = x.view(x.size(0), -1)
        
        return (
            self.fc_top(features),
            self.fc_right(features),
            self.fc_bottom(features),
            self.fc_left(features)
        )