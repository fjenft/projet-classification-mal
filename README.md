# Analyse, modélisation et classification d'une base de données : MyAnimeList

client id = c2db532c391bf31339ffd6afa650d528

## Feature engineering
Pour récupérer les données qui vont nous servir au cours de ce projet, il faut avant tout lancer le ficher récup_anime_par_rang_et_nettoyage qui va à la fois récupérer un dataframe qui liste les animés et leur caractéristiques ainsi que faire une partie du nettoyage sur ces données (traiter les valeurs NaN, supprimer les colonnes inutiles, etc...). Le dataframe sera enregistré localement au format csv. La récupération du dataframe peut prendre un peu de temps.

## Analyse des données
## Modélisation et prédiction à l'aide des classifieurs
### Motivations 
#### Dans cette partie, nous souhaite **prédire** si un *anime* sera ou non populaire en nous basant sur les données disponibles. A chaque utilisateur, on associe une réalisation du vecteur de variables aléatoires associées à chaque caractéristique mesurée (score donné par les utilisateurs, nombres de vues ...).

#### La variable **popularity** renseigne sur le rang de popularité d'une oeuvre basé sur le nombre total de membres l'ayant ajouté. Par exemple, un *anime* de popularité 1 signifie que l'oeuvre a été rajoutée par le plus grand nombre d'utilisateurs. C'est nombre variable dépendante.

#### Cependant, pour plus de visibilité nous allons créer différentes classes en fonction des valeurs de la variables **popularity** : {"nul", "moyen", "Très populaire", "populaire"},  L'intérêt de la classification devient immédiat : pour chaque nouvelles observations, être en mesure d'attribuer un ordre de grandeurs de popularité.



### Guide d'utilisation du code

Pour récupérer les données qui vont nous servir au cours de ce projet, il faut avant tout lancer la première partie du notebook (ou le fichier recup_anime_par_rang_et_nettoyage) qui va à la fois récupérer un dataframe qui liste les animés et leur caractéristiques ainsi que faire une partie du nettoyage sur ces données (traiter les valeurs NaN, supprimer les colonnes inutiles, etc...). Le dataframe sera enregistré localement au format csv. La récupération du dataframe peut prendre un peu de temps.

Après avoir récupérer les données, vous pouvez :
-lancer la seconde partie du notebook (ou le fichier python analyse_descriptive). Cela vous fournira une analyse des données obtenues en première étape
-lancer la troisième partie, première sous-section du notebook (ou le fichier classification_par_reg_log). Cela vous fournira une classification des animés en fonction de leur popularité avec une méthode par régression linéaire
-lancer la troisème partie, deuxième sous-section du notebook (ou le fichier lasso), qui vous fera une classification à l'aide d'une regression lasso
-lancer la troisième partie, troisième sous-section du notebook (ou le fichier classification_par_XGB), qui fera de même que classification à l'aide de XGboost
