import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

anime_data=pd.read_csv("anime_data.csv")

print(anime_data['mean'].describe())

#Comme une note moyenne des utilisateurs pour un anime donné était mise à 0 quand elle était inconnue (NaN), on va supprimer les animes dont la note moyenne est 0.
#De manière intuitive, il est normal de supprimer ces animes car une note moyenne par les utilisateurs de 0 est quasi impossible

anime_data_for_score=anime_data[anime_data['mean'] !=0]
print(anime_data_for_score['mean'].describe())

anime_data['start_year'].value_counts().sort_index()

#1/On pense que la distribution du nombre d'animés en fonction de leur note moyenne est une courbe en cloche centré autour de 5. Pour vérifier cette hypothèse, on fait un histogramme qui représente le nombre d'animes par note moyenne.
plt.figure(figsize=(10, 6))
sns.histplot(anime_data_for_score['mean'], bins=30, kde=True, color='blue')
plt.title("Distribution des animes en fonction de leurs notes moyennes")
plt.xlabel("Note moyenne")
plt.ylabel("Nbr d'animes")
plt.savefig("Nbr_animés_par_note_moyenne.png")
plt.close()

#2/On pense que la production d'animés n'a cessé d'augmenter avec le temps. Pour vérifier notre intuition, on représente l'évolution du nombre d'animes produits par année à l'aide d'un histogramme
plt.figure(figsize=(12, 6))
sns.countplot(x='start_year', data=anime_data, palette='viridis', hue='start_year', legend=False)
plt.xticks(rotation=90)
plt.title("Nombre d'animes produits par année")
plt.xlabel("Année de sortie")
plt.ylabel("Nbr d'animés")
plt.savefig("Nbr_animés_par_annee.png")
plt.close()

#3/On veut savoir si une source est plus prolifique en animes que d'autres. Pour cela, on créé un histogramme qui représente le nombre d'animes en fonction de la source dont ils sont inspirés
plt.figure(figsize=(10, 6))
source_counts = anime_data['source'].value_counts()
sns.barplot(x=source_counts.index, y=source_counts.values, palette='coolwarm', hue=source_counts.index, legend=False)
plt.xticks(rotation=90)
plt.title("Nbr d'animes par sources'")
plt.xlabel("Source")
plt.ylabel("Nombre d'animés")
plt.savefig("Nbr_animes_par_source.png")
plt.close()

#4/On veut savoir si une source est plus susceptibles de produire des animés bien notés que d'autres. Pour cela on fait un histogramme qui représente la note moyenne des animés d'une même source en fonction de cette source.
mean_scores_by_source = anime_data_for_score.groupby('source')['mean'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
mean_scores_by_source.plot(kind='bar', color='skyblue')
plt.title("Notes moyennes par source")
plt.xlabel("Source")
plt.ylabel("Note moyenne")
plt.savefig("Notes_moyennes_par_source.png")
plt.close()

#5/On veut savoir si le nombre d'utilisateurs qui a noté un anime en particulier influence la note moyenne de cette animé. Pour voir cela, on fait un nuage de points dont les points sont la note moyenne d'un anime en fonction du nombre d'utilisateurs qui ont noté cette animé.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_list_users', y='mean', data=anime_data_for_score, alpha=0.6)
plt.title("Relation entre le nombre d'utilisateurs et la note moyenne")
plt.xlabel("Nombre d'utilisateurs ayant noté")
plt.ylabel("Note moyenne")
plt.xscale('log')  # On utilise une échelle logarithmique pour mieux visualiser
plt.savefig("relation_ entre_nbr_utilisateurs_et_notes_moyennes.png")
plt.close()

#6/On veut comparer le score moyen des animes regroupés par rang (par exemple : rang 1 à 100 puis rang 101 à 200, etc,...).
# Pour cela, on va créer 10 groupes de 100 animés

anime_data_for_score['Groupe'] = (anime_data_for_score['rank'] - 1) // 100 + 1 #on créé les groupes de 100 animés
anime_1000 = anime_data_for_score[anime_data_for_score['Groupe'] <= 10] #on ne garde que les 10 premiers groupes

grouped_means = anime_1000.groupby('Groupe')['mean'].mean().reset_index()
grouped_means['Groupe'] = grouped_means['Groupe'].astype(str)  # Conversion pour le graphique

plt.figure(figsize=(12, 6))
sns.barplot(x='Groupe', y='mean', data=grouped_means, palette='viridis')
plt.title("Scores moyens des animes par groupe de rangs (sans scores nuls)")
plt.xlabel("Groupe de rangs (par 100)")
plt.ylabel("Score moyen")
plt.savefig("scores_moyens_par_100_rangs.png")
plt.close()

#7/On veut voir si animés sont mieux notés d'une année sur l'autre. Pour cela, on fait un courbe dont chaque point est la moyenne des notes moyennes des animés d'une année en fonction de l'année
mean_score_by_year = anime_data_for_score.groupby('start_year')['mean'].mean()
plt.figure(figsize=(12, 6))
mean_score_by_year.plot(kind='line', color='green', marker='o')
plt.title("Évolution des notes moyennes des animes au fil des années")
plt.xlabel("Année")
plt.ylabel("Note moyenne")
plt.savefig("evolution_notes_moyennes_par_annee.png")
plt.close()

#8/La durée d'un épisode est généralement de 20 minutes (par expérience). On veut vérifier si cette intuition est vraie. Pour cela, on réalise un histogramme

anime_data['average_episode_duration']=anime_data['average_episode_duration']/60

plt.figure(figsize=(10, 6))
sns.histplot(anime_data['average_episode_duration'].dropna(), bins=30, kde=True, color='purple')
plt.title("Nbr d'animés en fonction de la durée moyenne de leurs épisodes")
plt.xlabel("Durée moyenne des épisodes (en minutes)")
plt.ylabel("Nbr d'animés")

x_ticks = np.arange(0, anime_data['average_episode_duration'].max() + 10, 5)  # de 0 à max+10 par pas de 5 minutes
plt.xticks(x_ticks)

plt.savefig("Nbr_animés_en_fct_durée_ep.png")
plt.close()

#9/On s'attend à ce que la plupart des animés ont 12 ou 24 épisodes. Pour vérifier cette intuition, on fait un histogramme qui représente le nbr d'animés en fonction de leur nombres d'épisodes (si le nbr d'épisodes est de plus de 100, il ne sera pas représenté, car peu d'animés valident cette condition et ce sont des anomalies)

anime_filtered = anime_data[anime_data['num_episodes'].between(1, 100)]

episode_counts = anime_filtered['num_episodes'].value_counts().reindex(range(1, 101), fill_value=0).sort_index()

plt.figure(figsize=(18, 8))
sns.barplot(x=episode_counts.index, y=episode_counts.values, color='blue')

plt.title("Nombre d'animés en fonction du nombre d'épisodes (1 à 100)", fontsize=14)
plt.xlabel("Nombre d'épisodes", fontsize=12)
plt.ylabel("Nombre d'animés", fontsize=12)

plt.xticks(ticks=range(1, 101, 5), labels=range(1, 101, 5), rotation=45)  # Ticks tous les 5 pour la lisibilité

plt.savefig("nbr_animes_par_nombre_episodes_1_to_100.png", dpi=300)