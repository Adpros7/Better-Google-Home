import os

from geopy import Nominatim
from num2words import num2words
from xsmeteo import ForecastResponse, XSMeteo


lat, long = eval(os.environ["MY_LOCATION"]) # in the form "(lat, long)""
with XSMeteo() as client:
    current: ForecastResponse = client.get_forecast(
    latitude=lat,
    longitude=long,
    current=["temperature_2m", "apparent_temperature"],
    temperature_unit="fahrenheit",
    )

print(current.current)
addon = f"SYSTEM PROMPT: Take this information and take what you need to ANSWER the question comprehensively yet concisely LOCATION: {Nominatim(user_agent='betterghome').reverse((lat, long))}, \
CURRENT TEMPERATURE: {num2words(round(float(current.current['temperature_2m'])))}, \
FEELS LIKE TEMPERATURE: {num2words(round(float(current.current['apparent_temperature'])))}\
FORMATTING RULES: NO NEWLINES OR ANYTHING. ONLY GIVE THE USER WHAT THEY ASKED FOR. NO EMOJIS. ALWAYS WRITE OUT THE SYMBOLS AND NUMBERS AS WORDS. for example 45.4°F becomes forty-five point four degrees fahrenheit. also, if referencing a location, you only need the town and place. so Grove street, watertown, MA, USA ebcomes watertown, Massachusetts\
Also, remember to only give what the user asked for, if they asked for temperatature, give them ONLY temperature, not also feels like temperature. An for god sakes, don't be too specfc with the location. Also, try to sound natural. For example, say things like It's twenty-four degrees fahrenheit in Watertown, Massachusetts. DON'T say its 24°F"

                