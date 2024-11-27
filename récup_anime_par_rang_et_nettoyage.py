import requests
import pandas as pd
import ast

#comme l'API ne nous permet pas de récupérer les animes en groupe à travers leur ID, on fait le faire par le rang décroissant (en prenant les 100 1ers rangs, ainsi de suite...)

all_anime = []  # List to store all anime data
nbr_needed = 27490  # Total number of anime on MAL as of 13/11/2024 (obtained by looking directly on the site of myanimelist)

ID = {'X-MAL-CLIENT-ID': 'c2db532c391bf31339ffd6afa650d528'}
url = 'https://api.myanimelist.net/v2/anime/ranking'
parameters = {
    'ranking_type': 'all',  # Retrieve anime across all rankings
    'limit': 100,  # Max limit per request, divides the total number of anime on mal
    'fields': 'id,title,mean,start_date,end_date,rank,popularity,num_list_users,num_scoring_users,nsfw,media_type,status,num_episodes,start_season,broadcast,source,average_episode_duration,rating'
}

k = 0  # offset but also the number of times the loop is used that is 27490/127 here

# Loop until we've collected the target number of anime
while k < nbr_needed:
    parameters['offset'] = k
    mal = requests.get(url, headers=ID, params=parameters)

    
    if mal.status_code == 200: # Check if the request is successful
        data = mal.json()
        # add as much new anime as number in limits
        all_anime.extend(data['data'])
        k += parameters['limit']

        # Print progress
        print(str(len(all_anime)) + " collected for the moment...")
    
        # When we reach the target number, stops
        if len(all_anime) >= nbr_needed:
            print("the total number of anime collected is " + str(len(all_anime)))
            break
    else :
        print("cannot retrieve more than " + str(len(all_anime))) 
        break

anime_data = pd.DataFrame(all_anime) #final data put in dataframe
print(anime_data.head(2))

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

#On supprime les colonnes qui ne serviront pas pour la recommendation
anime_data=anime_data.drop(columns=['main_picture','broadcast','start_season','end_date'],axis=1)
pd.set_option('display.max_columns', None)
print(anime_data.head())

#On regarde combien de valeurs NaN il y a dans chaque colonne
for i in anime_data.columns:
    k = anime_data[i].isna().sum()
    print(f"Le nombre de NaN dans la colonne '{i}' est : {k}")

anime_data['source'] = anime_data['source'].fillna('source_inconnue')
anime_data['rating'] = anime_data['source'].fillna('rating_inconnu')
anime_data['mean'] = anime_data['mean'].fillna(0)

#On veut uniquement garder l'année dans la colonne start_date
anime_data['start_date'] = pd.to_datetime(anime_data['start_date'], errors='coerce')  
anime_data['start_year'] = anime_data['start_date'].dt.year  
anime_data=anime_data.drop(columns=['start_date'],axis=1)

anime_data = anime_data.dropna(subset=['start_year'])

#On vérifie qu'il n'y a plus de NaN
nbr_nan = anime_data.isna().sum().sum()
print(f"Il reste {nbr_nan} NaN")


import s3fs
from io import StringIO
import os

print("AWS Access Key ID: ", os.environ.get('AWS_ACCESS_KEY_ID'))
print("AWS Secret Access Key: ", os.environ.get('AWS_SECRET_ACCESS_KEY'))

# Configuration du système de fichiers S3
fs = s3fs.S3FileSystem(anon=False)
# Définir le chemin cible dans le stockage S3
chemin_s3 = 's3://your-ssp-cloud-bucket/fjenft/anime_data_nettoyé.csv'
# Créer un buffer en mémoire pour stocker votre CSV
csv_buffer = StringIO()
# Convertir votre DataFrame en CSV et l'écrire dans le buffer en mémoire
anime_data.to_csv(csv_buffer, index=False)
# Rewind the buffer before uploading
csv_buffer.seek(0)
# Télécharger le CSV depuis le buffer vers S3
fs.put(csv_buffer, chemin_s3)
