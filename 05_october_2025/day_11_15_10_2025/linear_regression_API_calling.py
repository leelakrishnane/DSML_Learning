import requests
url = "http://127.0.0.1:5050/linear_model_predict"
payload = {"year": [2007, 2011, 2015]}
response = requests.post(url, json=payload)

print(response.text)
print(response.json())
print(response.status_code)