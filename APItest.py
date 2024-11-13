import requests 
import pandas as pd 

id ={'X-MAL-CLIENT-ID' : 'c2db532c391bf31339ffd6afa650d528'} 
url = f'https://api.myanimelist.net/v2/anime?q=1&limit=1000&fields=id,title,mean,type,start_date,end_date,rank,poularity,,num_list_users,num_scoring_users,nsfw,media_type,status,,num_episodes,start_season,broadcast,source,average_episode_duration,rating'

maldata = requests.get(url, id) 
data = maldata.json()
data1=pd.DataFrame(data)
data.head(2)