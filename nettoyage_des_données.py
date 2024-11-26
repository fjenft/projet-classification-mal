import pandas as pd

# On charge le fichier csv créé dans le fichier de récupération des données sur les animes dans un DataFrame
anime_data = pd.read_csv(r'C:\Utilisateurs\fjenf\Téléchargements\myanimelist_dataset_non_nettoyé.csv')

print(anime_data.head())
print(anime_data.info())

import requests

#on voit que toutes les caractéristiques de chaque anime est contenu dans un "node", un dictionnaire, sauf le rang. 
#Il faut donc extraire chaque élément du dictionnaire node pour un faire une colonne à part entière

# Vérifier si 'node' est une colonne dans le DataFrame
if 'node' in anime_data.columns:
    # Fonction pour extraire toutes les clés d'un dictionnaire
    def extract_node_info(node):
        try:
            # Convertir la chaîne de caractères en dictionnaire
            node_dict = eval(node) if isinstance(node, str) else node
            # Retourner les éléments du dictionnaire sous forme de liste de tuples (clé, valeur)
            return node_dict
        except:
            return {}

    # Appliquer la fonction à chaque ligne de la colonne 'node'
    node_data = anime_data['node'].apply(extract_node_info)

    # Créer de nouvelles colonnes à partir des clés du dictionnaire 'node'
    for i, col in enumerate(node_data):
        for key in col.keys():
            anime_data[f'node_{key}'] = node_data.apply(lambda x: x.get(key) if isinstance(x, dict) else None)

# Vérifier les premières lignes pour voir les nouvelles colonnes
print(anime_data.head())

#On vérifie s'il y a des doublons
nbr_doublons = anime_data.duplicated().sum()
print(f"Il y a {nbr_doublons} doublons")

#On regarde combien de valeurs NaN il y a dans chaque colonne
for i in anime_data.columns:
    k = anime_data[i].isna().sum()
    print(f"Le nombre de NaN dans la colonne '{i}' est : {k}")