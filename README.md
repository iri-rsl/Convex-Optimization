# Convex Optimization Project: Student Depression Prediction

Implémentation mathématique complète d'algorithmes d'apprentissage supervisé linéaires et non-linéaires, appliqués au dataset Student Depression and Lifestyle (100k lignes) de Kaggle.

---

## Arborescence du Projet

```
Convex-Optimization/
├── data/
│   ├── raw/                          # Données brutes téléchargées depuis Kaggle
│   └── processed/                    # Données nettoyées et transformées (clean_student_data.csv)
├── notebooks/
│   └── 1.0_data_preprocessing.ipynb  # Pipeline complet d'EDA et preprocessing
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── pipeline.py               # Script d'automatisation du preprocessing
│   └── models/                       # Espace réservé pour SVM et réseau de neurones
├── README.md                         # Ce fichier
└── [requirements.txt]               # (Optionnel) Dépendances Python
```

### Description des rôles

| Répertoire | Rôle |
|---|---|
| `data/raw/` | Stockage temporaire des données brutes après téléchargement Kaggle |
| `data/processed/` | Données finales prétraitées, prêtes pour modélisation |
| `notebooks/` | Jupyter Notebooks pour exploration, validation et documentation |
| `src/preprocessing/` | Module Python pour automatiser le pipeline d'étapes (modularité) |
| `src/models/` | Implémentations mathématiques complètes du SVM convexe et réseau de neurones |

---

## Schema du Dataset

| Column Name | Description | Data Type | Range/Values | Traitement |
|---|---|---|---|---|
| **Student_ID** | Unique identifier | Integer | Unique IDs |  Supprimé |
| **Age** | Age du student | Integer | 18-24 |  Standardisé |
| **Gender** | Gender | String | Male, Female |  One-Hot Encoded |
| **Department** | Field of study | String | Engineering, Business, Arts | One-Hot Encoded |
| **CGPA** | Grade Point Average | Float | 0.0 - 4.0 | Standardisé |
| **Sleep_Duration** | Heures de sommeil/nuit | Float | Continuous | Standardisé |
| **Study_Hours** | Heures d'étude/jour | Float | Continuous | Standardisé |
| **Social_Media_Hours** | Heures sur réseaux/jour | Float | Continuous | Standardisé |
| **Physical_Activity** | Minutes d'activité/semaine | Integer | Continuous | Standardisé |
| **Stress_Level** | Niveau de stress (auto-rapporté) | Integer | 0-10 | Standardisé |
| **Depression** | Mental health status | Boolean | True / False |  {-1, 1} |

---

Le pipeline suit exactement **6 étapes rigoureuses** :

### **Étape 1: Chargement via Kagglehub**
- Téléchargement direct depuis Kaggle Dataset API
- Dataset: `aldinwhyudii/student-depression-and-lifestyle-100k-data`
- Format: CSV avec 100,000 observations

### **Étape 2: Contrôle Qualité**
- Vérification des valeurs manquantes (`isnull().sum()`) pour chaque colonne
- Suppression de l'identifiant non-informatif `Student_ID` (Integer: unique IDs)
  - Justification: Cet identifiant n'a pas de pouvoir prédictif et ne doit pas être inclus dans l'entraînement du modèle

### **Étape 3: Encodage One-Hot (Catégories)**
- Colonnes affectées: `Gender` (String: Male/Female), `Department` (String: Engineering, Business, Arts, etc.)
- Paramètres: `drop='first'` pour éviter la multicollinéarité
- Résultat: Variables binaires 0/1 indépendantes
- Variables créées : Gender_M, Department_B, Department_E/A (selon categories)

### **Étape 4: Standardisation Numérique (Scaling)**
- **Colonnes standardisées** (7 features):
  - `Age` (Integer, 18-24) → Scaling important (petite échelle)
  - `CGPA` (Float, 0.0-4.0) → Scaling important (petite échelle) 
  - `Sleep_Duration` (Float, heures) → Scaling
  - `Study_Hours` (Float, heures) → Scaling
  - `Social_Media_Hours` (Float, heures) → Scaling
  - `Physical_Activity` (Integer, minutes) → Scaling (grande échelle)
  - `Stress_Level` (Integer, 0-10) → Scaling
- Opération: $x_{\text{scaled}} = \frac{x - \mu}{\sigma}$
- Propriétés garanties: mean = 0, std = 1

### **Étape 5: Transformation de la Cible**
- Colonne source: `Depression` (Boolean)
  - `True` = Probable Depression (dépression probable)
  - `False` = Healthy (santé mentale normale)
- Conversion vers {**-1, 1**} (binaire strict)
  - `True` (Dépression) → `1`
  - `False` (Pas de dépression) → `-1`
- **Pourquoi?** Formulation mathématique du Hinge Loss du SVM

### **Étape 6: Export**
- Format final: CSV (`data/processed/clean_student_data.csv`)
- Index supprimés pour éviter redondance

---

## Justifications Mathématiques

### **Pourquoi Standardiser les Features?**

La standardisation (centering + scaling) est **critique** pour la convergence SGD :

1. **Évite la divergence des poids**: Sans scaling, les features à grande amplitude (ex: CGPA ∈ [0, 4]) entraîneraient des gradients explosifs comparés aux features petites (ex: Age ∈ [18, 25]).

2. **Accélère la convergence**: La standardisation rend la surface de la fonction de coût plus symétrique, permettant à SGD d'explorer l'espace des poids de manière plus efficace.

3. **Formulation mathématique**:
$$x_{\text{standardisé}} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}(x)}}$$

Après transformation: $\mathbb{E}[x'] = 0$ et $\text{Var}(x') = 1$ pour chaque feature.

---

### **Pourquoi des Labels en {-1, 1}?**

La fonction de perte Hinge Loss (SVM linéaire) repose sur la **formulation multiplicative** :

$$\text{Hinge Loss} = \max(0, 1 - y_i \cdot (w^T x_i + b))$$

où:
- $y_i \in \{-1, +1\}$ (label strictement binaire)
- $w^T x_i + b$ est le score de décision logit

**Propriété critique**: 
- Si $y_i \cdot (w^T x_i + b) > 1$ → pas de pénalité (bonne classification avec marge)
- Si $y_i \cdot (w^T x_i + b) \leq 1$ → perte positive

Avec labels 0/1 (standard), cette formulation ne fonctionne pas car le produit $y_i \cdot \text{score}$ ne traduit pas correctement la notion de **signe** de l'erreur.

---

## Commandes d'Exécution

### **Pré-requis**
Avant de lancer le projet, installez les dépendances Python requises :

```bash
pip install pandas numpy scikit-learn kagglehub jupyter
```

Si vous travaillez dans un environnement virtuel, activez-le avant d'exécuter ces commandes.

### **1. Lancer le Notebook complet**
Ouvrez le notebook pour exécuter pas à pas tout le prétraitement, y compris l'injection des variables socio-économiques :

```bash
jupyter notebook notebooks/1.0_data_preprocessing.ipynb
```
Puis exécutez les cellules dans l'ordre, de la cellule 1 jusqu'à l'export final. Le notebook charge les données Kaggle, injecte les variables socio-économiques, encode les variables catégorielles, standardise les variables numériques, transforme la cible et exporte le CSV final.

### **2. Lancer le pipeline automatisé**
Pour exécuter tout le flux de manière non interactive, lancez le script suivant depuis la racine du projet :

```bash
python src/preprocessing/pipeline.py
```
Ce script reproduit la même logique que le notebook, avec la même séquence de traitement : chargement Kaggle, injection des variables socio-économiques, contrôle qualité, encodage One-Hot, standardisation, transformation de la cible et export du CSV final.

### **3. Réutiliser le pipeline dans un script Python**
Si vous voulez intégrer ce prétraitement dans un autre script ou dans un futur entraînement de modèle, importez les fonctions du pipeline :

```python
from src.preprocessing.pipeline import (
    load_dataset,
    inject_socio_economic_features,
    quality_control,
    encode_categorical,
    standardize_numeric_features,
    transform_target_variable,
    save_processed_data
)

df = load_dataset()
df = inject_socio_economic_features(df)
df = quality_control(df)
df = encode_categorical(df)
df = standardize_numeric_features(df)
df = transform_target_variable(df)
save_processed_data(df)
```

### **Quel mode utiliser ?**
- Utilisez le **notebook** si vous voulez inspecter visuellement chaque étape et valider les transformations.
- Utilisez le **script `pipeline.py`** si vous voulez lancer tout le traitement d'un coup et produire `data/processed/clean_student_data.csv`.
- Utilisez l'**import des fonctions** si vous préparez une intégration dans un entraînement de SVM ou de réseau de neurones.

---

## Structure `src/models/` (À Venir)

Le répertoire `src/models/` est réservé pour les implémentations mathématiques suivantes :

### **Module 1: SVM Linéaire Convexe**
- **Optimiseur**: Descente de Gradient Stochastique (SGD)
- **Formulation mathématique**:
  - Minimisation du problème primal:
    $$\min_{w,b} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{n} \xi_i$$
    
  - Avec contrainte de marge douce: $y_i(w^T x_i + b) \geq 1 - \xi_i$

- **Conditions KKT**: Conditions de Karush-Kuhn-Tucker pour optimalité
- **Formulation duale Lagrangienne**: Pour analyse de convergence

### **Module 2: Réseau de Neurones Non-Convexe**
- **Architecture**: Couches feedforward avec activations ReLU
- **Fonction de coûts**: Cross-entropy pour classification binaire
- **Optimiseur**: SGD avec momentum / Adam
- **Pas de garantie de convergence globale** (surface non-convexe)

---

## Notes d'Intégration Futures

### **Features Complémentaires (Phase 2)**
Les features socio-économiques suivantes seront intégrées ultérieurement dans ce pipeline :
- Données de **dette** étudiant
- Données de **transport** (coûts de mobilité)
- Données de **logement** (type, coûts)
- Données d'**emploi** (job status, revenus)

**Stratégie**: Ces colonnes seront ajoutées dans le DataFrame après l'étape 2 (Quality Control) et subissent le même traitement (encoding, scaling) que les features actuelles. Les architectures SVM et réseau de neurones resteront inchangées.

### **Dépendances Python Requises**

```
pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.0.0
kagglehub>=0.1.0
jupyter>=1.0.0
```

(À installer via `pip install -r requirements.txt` si le fichier est disponible)

---

## Validation et Vérification

Après exécution du pipeline, vérifiez la présence de `data/processed/clean_student_data.csv` :

```bash
# Affiche les dimensions finales
python -c "import pandas as pd; df = pd.read_csv('data/processed/clean_student_data.csv'); print(f'Shape: {df.shape}'); print(f'Columns: {list(df.columns)}')"
```

---



---



## Dataset Source
https://www.kaggle.com/datasets/aldinwhyudii/student-depression-and-lifestyle-100k-data
