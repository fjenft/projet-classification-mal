import requests
import pandas as pd

#comme l'API ne nous permet pas de récupérer les animes en groupe à travers leur ID, on fait le faire par le rang décroissant (en prenant les 100 1ers rangs, ainsi de suite...)

all_anime = []  # List to store all anime data
nbr_needed = 27490  # Total number of anime on MAL as of 13/11/2024 (obtained by looking directly on the site of myanimelist)

ID = {'X-MAL-CLIENT-ID': 'c2db532c391bf31339ffd6afa650d528'}
url = 'https://api.myanimelist.net/v2/anime/ranking'
parameters = {
    'ranking_type': 'all',  # Retrieve anime across all rankings
    'limit': 100,  # Max limit per request, divides the total number of anime on mal
    'fields': 'id,title,mean,type,start_date,end_date,rank,popularity,num_list_users,num_scoring_users,nsfw,media_type,status,num_episodes,start_season,broadcast,source,average_episode_duration,rating'
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
        print("cannot retrive more than " + str(len(all_anime))) 
        break

dataf = pd.DataFrame(all_anime) #final data put in dataframe
print(dataf.head(2))

dataf.to_csv(r'C:\Utilisateurs\fjenf\Téléchargements\myanimelist_complete_dataset.csv', index=False)
