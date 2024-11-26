import pandas as pd
import ast

# On charge le fichier csv créé dans le fichier de récupération des données sur les animes dans un DataFrame
anime_data = pd.read_csv(r'C:\Utilisateurs\fjenf\Téléchargements\myanimelist_dataset_non_nettoyé.csv')

print(anime_data.head())
print(anime_data.info())

#On voit que toutes les caractéristiques de chaque anime est contenu dans un "node", un dictionnaire, sauf le rang. 
#Il faut donc extraire chaque élément du dictionnaire node pour en faire des colonnes à part entière

# On extrait toutes les clés du dictionnaire 'node' et on les transforme en colonnes du dataframe
anime_data['node'] = anime_data['node'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)    
keys = set().union(*(d.keys() for d in anime_data['node'] if isinstance(d, dict)))
for y in keys:
    anime_data[f'{y}'] = anime_data['node'].apply(lambda x: x.get(y) if isinstance(x, dict) else None)

# On convertit les colonnes contenant des dictionnaires en chaînes
for column in anime_data.columns:
    if anime_data[column].map(type).eq(dict).any():
        anime_data[column] = anime_data[column].apply(lambda x: str(x) if isinstance(x, dict) else x)

print(anime_data.head())

#On supprime la colonne node qui n'apporte plus d'info
anime_data = anime_data.drop(columns=['node'])

#On vérifie s'il y a des doublons
nbr_doublons = anime_data.duplicated().sum()
print(f"Il y a {nbr_doublons} doublons")

#On regarde combien de valeurs NaN il y a dans chaque colonne
for i in anime_data.columns:
    k = anime_data[i].isna().sum()
    print(f"Le nombre de NaN dans la colonne '{i}' est : {k}")