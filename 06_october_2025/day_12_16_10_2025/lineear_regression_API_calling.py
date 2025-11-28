import requests
url = "http://127.0.0.1:5060/linear_model_predict"
payload = {"carat": [0.5, 0.7, 1.0], "cut": ["Ideal", "Premium", "Good"], "depth": [60, 62, 65], "table": [55, 57, 58], "x": [3.5, 4.0, 4.5], "y": [4.0, 4.5, 5.0], "z": [2.5, 3.0, 3.5]}
response = requests.post(url, json=payload)

print(response.text)
print(response.json())
print(response.status_code)
