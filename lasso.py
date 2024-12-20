import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Function to categorize popularity
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

# Load data
df = pd.read_csv('anime_data.csv')

# creation of the new target variable
df['popularity_category'] = df['popularity'].apply(categorize_popularity)

# Selecting features and new target variable
X = df[['num_episodes', 'num_scoring_users', 'source', 'status', 'nsfw', 'rating', 'media_type', 'average_episode_duration', 'start_year']]
y = df['popularity_category']

# we change missing values if they exist with eiter the average or the median
for column in X.select_dtypes(include=['float64', 'int64']).columns:
    X[column].fillna(X[column].median(), inplace=True)
for column in X.select_dtypes(include=['object']).columns:
    X[column].fillna(X[column].mode()[0], inplace=True)

# changing categorical variables to numerical ones
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

# Split data into training and testing sets
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


#Now we execute the same code but this time we test different values of the hyperprarameter lambda and select the best one

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
    'classifier__C': np.logspace(-4, 4, 20)
}

# here we set up cv=5 and accuracy as the criteria for the selection
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

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
