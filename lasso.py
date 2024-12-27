import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Fonction pour classifier la popularité
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

# préparer notre dataframe
df = pd.read_csv('anime_data.csv')

# création de la nouvelle target cible
df['popularity_category'] = df['popularity'].apply(categorize_popularity)

# Sélection des variables explicatives et de la nouvelle variable cible
X = df[['num_episodes', 'num_scoring_users', 'source', 'status', 'nsfw', 'rating', 'media_type', 'average_episode_duration', 'start_year']]
y = df['popularity_category']

# convertir les variables catégoriques en des variables numériques
categorical_features = ['source', 'status', 'nsfw', 'rating', 'media_type']
numeric_features = ['num_episodes', 'num_scoring_users', 'average_episode_duration', 'start_year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l1', solver='saga', max_iter=10000))
])

# diviser le jeux de données en une partie test et une partie d'entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit 
model.fit(X_train, y_train)

# Predict 
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Visualize Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['impopulaire', 'peu populaire', 'Moyen', 'Populaire', 'Très populaire'], yticklabels=['impopulaire', 'peu populaire', 'Moyen', 'Populaire', 'Très populaire'])
plt.xlabel('predicted class')
plt.ylabel('actual class')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_defaultlambda.png', dpi=300, bbox_inches='tight')
plt.show()



#Maintenant on exécute le même code mais cette fois on teste différentes valeurs de l'hyperparamètre lambda afin de trouver celui qui donne les meilleurs résultats

from sklearn.model_selection import GridSearchCV


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


df = pd.read_csv('anime_data.csv')
df['popularity_category'] = df['popularity'].apply(categorize_popularity)
X = df[['num_episodes', 'num_scoring_users', 'source', 'status', 'nsfw', 'rating', 'media_type', 'average_episode_duration', 'start_year']]
y = df['popularity_category']
for column in X.select_dtypes(include=['float64', 'int64']).columns:
    X[column].fillna(X[column].median(), inplace=True)
for column in X.select_dtypes(include=['object']).columns:
    X[column].fillna(X[column].mode()[0], inplace=True)


categorical_features = ['source', 'status', 'nsfw', 'rating', 'media_type']
numeric_features = ['num_episodes', 'num_scoring_users', 'average_episode_duration', 'start_year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(penalty='l1', solver='saga', max_iter=10000))
])

# on utilise une distribution log pour balayer un plus grand champs de valeur
param_grid = {
    'classifier__C': np.logspace(-4, 4, 10)
}

# on fixe cv=5 et l'accuracy comme critère de selection
grid_search = GridSearchCV(pipeline, param_grid, cv=4, scoring='accuracy', verbose=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# we run the different models
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best C: {grid_search.best_params_['classifier__C']}")
print(f"Best model accuracy: {accuracy}")
print("Classification Report:")
print(report)


# Visualize Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['impopulaire', 'peu populaire', 'Moyen', 'Populaire', 'Très populaire'], yticklabels=['impopulaire', 'peu populaire', 'Moyen', 'Populaire', 'Très populaire'])
plt.xlabel('predicted class')
plt.ylabel('actual class')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_grid_search.png', dpi=300, bbox_inches='tight')
plt.show()
