from urllib import response
from js2py import translate_js6
import requests, json  # For weather
from googletrans import Translator

def translate(key):
    translator = Translator()
    trans = (translator.translate(key, src ='en', dest = 'vi')).text
    return trans
def weather(city_name):
    api_key = "ec95cc9adb68ffeb4c165def9dfacad3"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        curret_temp = int(y["temp"] - 273.15) #K = C+ 273.15
        curret_pressure = y["pressure"]
        curret_humid = y["humidity"]
        z = x["weather"]
        weather_description = z[0]["description"]
        weather_description = translate(weather_description)
        robot_brain = "hiện đang có " + str(weather_description) + ", Nhiệt độ là " + str(curret_temp) + "độ C, Độ ẩm là " + str(curret_humid)
    else:
        robot_brain = "Không tìm thấy tên thành phố"
    return robot_brain

weather("Hà Nội")