import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = '/home/onyxia/work/anime_data.csv' #make sure the path is correct!
df = pd.read_csv(csv_file_path)

#-I- Exploration initiale des données:
#L'objectif est de se familiariser avec le contenu du DataFrame en examinant un échantillon des premières lignes, en générant des statistiques descriptives et en identifiant les types de variables.

#-1- Afficher les premières lignes pour avoir un aperçu des données
print(df.head())

#-2- Générer des statistiques descriptives pour les variables numériques
print(df.describe())

#-3- Identifier le type de chaque variable dans le DataFrame
print(df.dtypes)


#-II- Exploration approfondie des données:
#L'objectif ici est de visualiser la distribution des variables numériques et d'examiner les relations entre elles à l'aide d'une matrice de corrélation

# Distribution des variables
df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distributions of Numerical Variables")
plt.show()
plt.savefig('/home/onyxia/work/distributions.png')


# Matrice de corrélation
numerical_features = ['num_list_users', 'num_episodes', 'mean', 'rank', 'popularity', 'num_scoring_users', 'start_year']
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
plt.savefig('/home/onyxia/work/heatmap.png')



#-III- Analyse de la popularité (variable à prédire) en fonction de diverses variables


#1 Popularité en fonction des moyennes des notes
#Lors de l'analyse exploratoire (notamment avec la matrice de corrélation), nous avons constaté une corrélation significative entre la popularité et les moyennes des notes.
plt.figure(figsize=(10, 6))
plt.hexbin(x=df['mean'], y=df['popularity'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.title('Popularity as a function of Mean Ratings')
plt.xlabel('Mean Ratings')
plt.ylabel('Popularity Rank')
plt.show()
plt.savefig('/home/onyxia/work/Popularity&mean_scores.png')

#2 Popularité en fonction du nombre d'utilisateurs (qui ont regardé l'anime)
#Nous cherchons à déterminer la nature de la relation entre le nombre d'utilisateurs et la popularité des animes, car il est logique de supposer que ces deux variables sont liées.
plt.figure(figsize=(10, 6))
plt.scatter(df['num_list_users'], df['popularity'], alpha=0.6, s=10)
plt.xlim(0, df['num_list_users'].max())
plt.title('Popularity as a function of number of users')
plt.xlabel('num_list_users')
plt.ylabel('Popularity Rank')
plt.yscale('log') #échelle logarithmique
plt.show()
plt.savefig('/home/onyxia/work/Popularity&numbr_users.png')

# Popularité en fonction du type de média
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['media_type'], y=df['popularity'], order=df['media_type'].value_counts().index)
plt.title('Popularity vs. Media Type')
plt.xlabel('Media Type')
plt.ylabel('Popularity Rank')
plt.yscale('log')
plt.xticks(rotation=45)
plt.show()
plt.savefig('/home/onyxia/work/Popularity&type.png')
