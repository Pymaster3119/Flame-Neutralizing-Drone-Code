import base64
import requests

def CreateIMGBBURL(image_path, api_key):
    url = "https://api.imgbb.com/1/upload"
    with open(image_path, "rb") as file:
        image_data = base64.b64encode(file.read())
    payload = {
        "key": api_key,
        "image": image_data
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        json_data = response.json()
        direct_url = json_data["data"]["url"]
        return direct_url
    else:
        raise Exception(f"Upload failed: {response.text}")
