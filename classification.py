import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder  # On convertit les labels en nombres entiers
import time
import pandas as pd

starting_time = time.time()
# Préparation des exemples et des étiquettes
df['popularity'] = LabelEncoder().fit_transform(df['popularity'])
X = df.drop(columns=['popularity'])  # autres caractéristiques que popularité ('variables indépendantes')
y = df['popularity']    # variable dépendante

# Séparation des données en test/train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Assurer que toutes les classes sont représentées dans y_train et y_test
y_train_classes = y_train.unique()
y_test_classes = y_test.unique()

# Si des classes sont absentes, on peut en ajouter une ligne pour traiter cela
if len(set(y_train_classes) - set(y_test_classes)) > 0:
    y_test = pd.concat([y_test, pd.Series([y_train_classes[0]] * (len(y_train_classes) - len(y_test_classes)), index=y_test.index)])

# Training time
training_time = time.time()-starting_time
print("Trainning time:", training_time)


# Modèle XGBoost et entrainement du modèle
model = xgb.XGBClassifier(
    objective='multi:softmax',  #classification mutliclasses
    num_class=len(y.unique()),
    use_label_encoder=False,
    eval_metric='mlogloss'  #perte log
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_train))

# AUC Test
y_pred_proba = model.predict_proba(X_test)  # Prédictions sous forme de probabilités
auc_test = roc_auc_score(y_true=y_test, y_score=y_pred_proba, multi_class='ovr')
print("auc_test_xgb", auc_test)

# Modèle de classification binaire - transformers et NLP
