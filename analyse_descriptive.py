import pandas as pd

anime_data=pd.read_csv("anime_data.csv")

print(anime_data['mean'].describe())

#Comme une moyenne du score des utilisateurs pour un anime donné était mise à 0 quand elle était inconnue (NaN), on va supprimer les animes dont la moyenne est 0.
#De manière intuitive, il est normal de supprimer ces animes car une moyenne de scores par les utilisateurs de 0 est quasi impossible

anime_data_for_score=anime_data[anime_data['mean'] !=0]
print(anime_data_for_score['mean'].describe())

anime_data['start_year'].value_counts().sort_index()

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Distribution des scores moyens des animes
plt.figure(figsize=(10, 6))
sns.histplot(anime_data_for_score['mean'], bins=30, kde=True, color='blue')
plt.title("Distribution des scores moyens des animes")
plt.xlabel("Score moyen")
plt.ylabel("Fréquence")
plt.savefig("distribution_scores.png")
plt.close()

# 2. Nombre d'animes produits par année
plt.figure(figsize=(12, 6))
sns.countplot(x='start_year', data=anime_data_for_score, palette='viridis', hue='start_year', legend=False)
plt.xticks(rotation=90)
plt.title("Nombre d'animes produits par année")
plt.xlabel("Année de sortie")
plt.ylabel("Nombre d'animes")
plt.savefig("animes_par_annee.png")
plt.close()

# 3. Distribution des sources des animes
plt.figure(figsize=(10, 6))
source_counts = anime_data_for_score['source'].value_counts()
sns.barplot(x=source_counts.index, y=source_counts.values, palette='coolwarm', hue=source_counts.index, legend=False)
plt.xticks(rotation=90)
plt.title("Distribution des sources des animes")
plt.xlabel("Source")
plt.ylabel("Nombre d'animes")
plt.savefig("distribution_sources.png")
plt.close()

# 4. Scores moyens par source
mean_scores_by_source = anime_data_for_score.groupby('source')['mean'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
mean_scores_by_source.plot(kind='bar', color='skyblue')
plt.title("Scores moyens par source")
plt.xlabel("Source")
plt.ylabel("Score moyen")
plt.savefig("scores_par_source.png")
plt.close()

# 5. Relation entre le nombre d'utilisateurs et le score moyen
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_list_users', y='mean', data=anime_data_for_score, alpha=0.6)
plt.title("Relation entre le nombre d'utilisateurs et le score moyen")
plt.xlabel("Nombre d'utilisateurs ayant noté")
plt.ylabel("Score moyen")
plt.xscale('log')  # Échelle logarithmique pour mieux visualiser
plt.savefig("relation_users_scores.png")
plt.close()

# On veut comparer le score moyen des animes regroupés par rang (par exemple : rang 1 à 100 puis rang 101 à 200, etc,...).
# Pour cela, on va créer 10 groupes de 100 animés

anime_data_for_score['Groupe'] = (anime_data_for_score['rank'] - 1) // 100 + 1 #on créé les groupes de 100 animés
anime_1000 = anime_data_for_score[anime_data_for_score['Groupe'] <= 10] #on ne garde que les 10 premiers groupes

grouped_means = anime_top_1000.groupby('Groupe')['mean'].mean().reset_index()
grouped_means['Groupe'] = grouped_means['Groupe'].astype(str)  # Conversion pour le graphique

plt.figure(figsize=(12, 6))
sns.barplot(x='Groupe', y='mean', data=grouped_means, palette='viridis')
plt.title("Scores moyens des animes par groupe de rangs (sans scores nuls)")
plt.xlabel("Groupe de rangs (par 100)")
plt.ylabel("Score moyen")
plt.savefig("scores_moyens_par_100_rangs.png")
plt.close()

# 7. Évolution des scores moyens au fil des années
mean_score_by_year = anime_data_for_score.groupby('start_year')['mean'].mean()
plt.figure(figsize=(12, 6))
mean_score_by_year.plot(kind='line', color='green', marker='o')
plt.title("Évolution des scores moyens des animes au fil des années")
plt.xlabel("Année")
plt.ylabel("Score moyen")
plt.savefig("evolution_scores_par_annee.png")
plt.close()

# 8. Distribution des durées moyennes des épisodes
plt.figure(figsize=(10, 6))
sns.histplot(anime_data_for_score['average_episode_duration'].dropna(), bins=30, kde=True, color='purple')
plt.title("Distribution des durées moyennes des épisodes")
plt.xlabel("Durée moyenne des épisodes (en secondes)")
plt.ylabel("Fréquence")
plt.savefig("distribution_duree_episodes.png")
plt.close()
