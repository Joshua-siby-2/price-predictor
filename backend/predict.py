import requests
import json

url = "http://127.0.0.1:8000/predict"

data = {
    "name": "Laptop",
    "description": "Laptop with RAM: 8GB, Storage: 256GB SSD, Processor: Intel i5, Battery: 6 hours"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())