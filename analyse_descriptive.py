import pandas as pd

anime_data=pd.read_csv("anime_data.csv")

print(anime_data['mean'].describe())

#Comme une moyenne du score des utilisateurs pour un anime donné était mise à 0 quand elle était inconnue (NaN), on va supprimer les animes dont la moyenne est 0.
#De manière intuitive, il est normal de supprimer ces animes car une moyenne de scores par les utilisateurs de 0 est quasi impossible

anime_data_for_score=anime_data[anime_data['mean'] !=0]
print(anime_data_for_score['mean'].describe())

print(anime_data.corr())