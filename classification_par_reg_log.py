import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder  # On convertit les labels en nombres entiers
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

anime_data=pd.read_csv("anime_data.csv")
anime_data=anime_data.drop(columns=["ranking", "rating", "title", "source", "media_type", "status", "nsfw"])

starting_time=time.time()
# Classement des différents types de popularités en 4 types distincts :

def categorize_popularity(popularity):
    if popularity < 50:
        return 3  # Très populaire
    elif 50 <= popularity < 200:
        return 2  # Populaire
    elif 200 <= popularity < 500:
        return 1  # Moyen
    elif 500 <= popularity < 1000:
        return 0  # Nul
    return -1  # Si aucune catégorie ne correspond (valeurs > 1000 ou autres cas)

# Création de la nouvelle colonne categorize_popularity 
anime_data['popularity_category'] = anime_data['popularity'].apply(categorize_popularity)

# Préparation des exemples et des étiquettes
X = anime_data.drop(columns=['popularity', 'popularity_category'])  # Autres caractéristiques, sans 'popularity'
y = anime_data['popularity_category']  # Variable dépendante (nouvelle catégorisation)

# Séparation des données en test/train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Assurer que toutes les classes sont représentées dans y_train et y_test
y_train_classes = y_train.unique()
y_test_classes = y_test.unique()

# Si des classes sont absentes, on peut en ajouter une ligne pour traiter cela
if len(set(y_train_classes) - set(y_test_classes)) > 0:
    y_test = pd.concat([y_test, pd.Series([y_train_classes[0]] * (len(y_train_classes) - len(y_test_classes)), index=y_test.index)])

y_test=y_test.reset_index(drop=True)

# Training time
training_time = time.time()-starting_time
print("Trainning time:", training_time)

# Modèle regression logistique et entrainement du modèle
model_lr = LogisticRegression(
    multi_class='multinomial',  #classification mutliclasses
    solver='lbfgs',
    max_iter=500
)

model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)
print(classification_report(y_test, y_pred))

# AUC Test
y_pred_proba = model_lr.predict_proba(X_test)  # Prédictions sous forme de probabilités
auc_test = roc_auc_score(y_true=y_test, y_score=y_pred_proba, multi_class='ovr')
print("auc_test_xgb", auc_test)


