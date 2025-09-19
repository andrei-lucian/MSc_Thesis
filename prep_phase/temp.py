import requests

api_key = '95dfd741'
imdb_id = 'tt0114709'

url = f'http://www.omdbapi.com/?i={imdb_id}&apikey={api_key}'
response = requests.get(url)
data = response.json()

data = response.json()
print(data)