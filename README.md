# Analyse, modélisation et classification d'une base de données : MyAnimeList

## Guide d'utilisation du code

Pour récupérer les données qui vont nous servir au cours de ce projet, il faut avant tout lancer la première partie du notebook (ou le fichier recup_anime_par_rang_et_nettoyage) qui va à la fois récupérer un dataframe qui liste les animés et leur caractéristiques ainsi que faire une partie du nettoyage sur ces données (traiter les valeurs NaN, supprimer les colonnes inutiles, etc...). Le dataframe sera enregistré localement au format csv. La récupération du dataframe peut prendre un peu de temps.

Après avoir récupérer les données, vous pouvez :
-lancer la seconde partie du notebook (ou le fichier python analyse_descriptive). Cela vous fournira une analyse des données obtenues en première étape
-lancer la troisième partie, première sous-section du notebook. Cela vous fournira une classification des animés en fonction de leur popularité avec une méthode par régression linéaire
-lancer la troisième partie, deuxième sous-section du notebook, qui vous fera une classification à l'aide d'une regression lasso
-lancer la troisième partie, troisième sous-section du notebook, qui fera de même que classification à l'aide de XGboost

## Justification des différentes parties

### Récupération des données
On récupère les données qu'on va utiliser avec l'API de My Anime List. Les données seront les animés identifiés par un numéro id propre, et leurs caractéristiques (telles que leur note moyenne donnée par les utilisateurs, leurs date de sortie, etc...)

### Feature engineering
Le feature engineering ou nettoyage des données va nous permettre de mettre correctement en forme nos données pour qu'elles soient ensuite réutilisable pour leur analyse et les modélisations faites.

### Analyse des données
Cette partie servira à étudier les données, leur structures, et tester quelques relations entre les variables et les intuitions qu'on pourrait avoir par rapport à celles-ci. Pour cela, on utilisera essentiellement des visualisations comme des histogrammes ou autres.

### Modélisation et prédiction à l'aide des classifieurs
- Dans cette partie, nous souhaite **prédire** si un *anime* sera ou non populaire en nous basant sur les données disponibles. A chaque utilisateur, on associe une réalisation du vecteur de variables aléatoires associées à chaque caractéristique mesurée (score donné par les utilisateurs, nombres de vues ...).

- La variable **popularity** renseigne sur le rang de popularité d'une oeuvre basé sur le nombre total de membres l'ayant ajouté. Par exemple, un *anime* de popularité 1 signifie que l'oeuvre a été rajoutée par le plus grand nombre d'utilisateurs. C'est nombre variable dépendante.

- Cependant, pour plus de visibilité nous allons créer différentes classes en fonction des valeurs de la variables **popularity** : {"nul", "moyen", "Très populaire", "populaire"},  L'intérêt de la classification devient immédiat : pour chaque nouvelles observations, être en mesure d'attribuer un ordre de grandeurs de popularité. A l'aide de la matrice de corrélation, nous pouvons d'ores et déjà exclure les variables très corrélées avec la variable dépendante comme 'ranking' par exemple.

- Nous proposons d'utiliser deux modèles de classification multi-classes : régression logistique (multinomiale) et une méthode de boosting (Extreme Gradient Boosted ou XGBoost). Le code est entièrement détaillé dans le notebook.

- Enfin, nous allons mesurer les performances de chacune des deux classifications puis les comparerons en se basant sur la matrice de confusion et l'AUC.



client id = c2db532c391bf31339ffd6afa650d528 (On utilisera cet id pour récupérer les animés et leur caractéristiques depuis l'API MAL)