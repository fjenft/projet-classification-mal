import requests
import urllib.parse
import pandas as pd

CLIENT_ID = "c2db532c391bf31339ffd6afa650d528"
headers = {
    'X-MAL-CLIENT-ID' : CLIENT_ID
}

# Création d'un dictionnaire titres/id (pour utiliser field)


def dict_id(query, limit):
    dict_anime_id = {}
    new_query = urllib.parse.quote(query)
    parameters = {'q': new_query, 'limit': limit}
    url = "https://api.myanimelist.net/v2/anime?q={new_query}&limit={limit}"
    req = requests.get(url, headers=headers, params=parameters)
    data = req.json()
    for anime in data['data']:
        title = anime['node']['title']
        dict_anime_id[title] = anime['node']['id']
    return dict_anime_id


#print(dict_id('one', 5))

#Dictionnaire, qui pour chaque titre, renvoie les principales caractéristiques du manga


def features(title):
    features_anime = {}
    features_anime['title'] = title
    anime_id = dict_id(title, limit=1)
    if not anime_id:
        return f'Pas de manga trouvé au nom de : {title}'
    url = 'https://api.myanimelist.net/v2/anime/{anime_id}?fields=id,title,main_picture'
    req = requests.get(url, headers=headers)
    print("Code de statut", req.status_code)
    print("Réponse brute:", req.text)
    data = req.json()
    print(data)


print(features('One Punch Man Specials'))