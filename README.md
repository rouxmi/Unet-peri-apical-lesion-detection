# Détection des lésions dentaires

La détection des lésions dentaires utilise Pytorch et [Monai](https://monai.io/). Les données utilisées proviennent de la base de données dentaires du CHR Metz-Thionville.

# Installation

1. Clonez le dépôt sur votre machine locale.
2. Installez les bibliothèques Python requises en utilisant la commande `pip install -r requirements.txt`.

# Répartition des données

Attention vous devez déjà mettre vos images au format dicom dans le dossier renseigner dans `config.ini` et de même pour le contourage au format XML.

Le jeu de données est composé de 220 images au total, qui peuvent être convertis depuis Dicom vers png et ont été divisées de manière aléatoire en ensembles d'entraînement, de validation et de test à raison de 70-20-10 %. La division est effectuée par :

```[Python]
    python3 convert_and_split_data.py
```

qui sauvegarde les listes de jeux de données dans le fichier `data.json`. Il stocke également les noms de classes et les poids de classes pour la segmentation.

# Modèle

Réseau UNet simple créé en utilisant `monai.netowrks`.

# Entraînement

```[Python]
    python3 train.py
```

Ce script lance l'entraînement. Il :

- sauvegarde le meilleur modèle dans le répertoire spécifié `model/model.pt`,
- utilise le dispositif `"mps"` (mais vous pouvez utiliser `"cuda"` ou `"cpu"`)
- Lit config.ini et configure l'entrainement de manière suivante:
- GPU à la position donnée,
- taille de lot (Batch size) donnée,
- taux d'apprentissage (Learning rate) donnée,
- le nombre maximal d'époques données.
- Sort un graphique de la progression du loss et du dice au cours de l'execution dans `outputs/history.png`

Exemple de config
```
[DEFAULT]

train_split = 0.8


[TRAIN]
device = cuda
gpu_id = 0
batch_size = 4
lr = 0.001
num_epochs = 100

[PATH]
img_folder = images
contourage_folder = contourages

[IMAGE]
width = 2048
height = 1024
uint = 8


```

# Évaluation

```[Python]
    python3 evaluation.py 
```

Ce script calcule l'évaluation sur l'ensemble d'entraînement, de validation et de test en sauvegardant le score dice pour chaque image dans le fichier `outputs/evaluation_results.csv`. et sauvegarde 5 exemples de prédiction dans `outputs/predictions`

# Exemple of outputs

Un dossier avec des exemples de ce que le modèle donne en sortie se trouve dans `Exemple of outputs`
