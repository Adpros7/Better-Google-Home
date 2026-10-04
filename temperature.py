import requests

def get_temperature(latitude, longitude):
    data = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature",
            "temperature_unit": "fahrenheit",
        },
    ).json()["current"]

    return round(data["temperature_2m"]), round(data["apparent_temperature"])