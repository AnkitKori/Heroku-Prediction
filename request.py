import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age': 40, 'cp':2, 'chol':250, 'thalach':165})

print(r.json())
