import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder  # On convertit les labels en nombres entiers
import time

starting_time = time.time()
# Préparation des exemples et des étiquettes
df['popularity'] = LabelEncoder().fit_transform(df['popularity'])
X = df.drop(columns=['popularity'])  # autres caractéristiques que popularité ('variables indépendantes')
y = df['popularity']    # variable dépendante

# Training time
training_time = time.time()-starting_time
print("Trainning time:", training_time)

# Séparation des données en test/train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Modèle XGBoost et entrainement du modèle
model = xgb.XGBClassifier(
    objective='multi:softmax',  #classification mutliclasses
    num_class=len(y.unique()),
    use_label_encoder=False,
    eval_metric='mlogloss'  #perte log
)

model.fit(X_train, y_train)

y_pred = model.predit(X_test)
print(classification_report(y_test, y_train))

auc_test = roc_auc_score(y_true=y_train.values, y_score=y_pred)  #affichage de l'AUC

print("auc_test_xgb", auc_test)

# Modèle de classification binaire - transformers et NLP
