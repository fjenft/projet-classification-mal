# Analyse, modélisation et classification d'une base de données : MyAnimeList

client id = c2db532c391bf31339ffd6afa650d528

## Feature engineering
Pour récupérer les données qui vont nous servir au cours de ce projet, il faut avant tout lancer le ficher récup_anime_par_rang_et_nettoyage qui va à la fois récupérer un dataframe qui liste les animés et leur caractéristiques ainsi que faire une partie du nettoyage sur ces données (traiter les valeurs NaN, supprimer les colonnes inutiles, etc...). Le dataframe sera enregistré localement au format csv. La récupération du dataframe peut prendre un peu de temps.

## Analyse des données
## Modélisation et prédiction à l'aide des classifieurs
### Motivations 
Pour récupérer les données qui vont nous servir au cours de ce projet, il faut avant tout lancer le ficher récup_anime_par_rang_et_nettoyage qui va à la fois récupérer un dataframe qui liste les animés et leur caractéristiques ainsi que faire une partie du nettoyage sur ces données (traiter les valeurs NaN, supprimer les colonnes inutiles, etc...). Le dataframe sera enregistré localement au format csv. La récupération du dataframe peut prendre un peu de temps.

Après avoir récupérer les données, vous pouvez :
-lancer le fichier python analyse_descriptive. Cela vous fournira une analyse des données obtenues en première étape
-lancer le fichier classification_par_reg_log. Cela vous fournira une classification des animés en fonction de leur popularité avec une méthode par régression linéaire
-lancer le fichier classification_par_XGB, qui fera de même que classification_par_reg_log mais cette fois-ci par XGboost
-lancer le fichier lasso