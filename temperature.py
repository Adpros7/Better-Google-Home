import requests
from num2words import num2words


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

    return num2words(round(data["temperature_2m"])), num2words(
        round(data["apparent_temperature"])
    )
