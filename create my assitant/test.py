from urllib import response
from js2py import translate_js6
import requests, json  # For weather
from googletrans import Translator

trans = ""
def translate(key):
    translator = Translator()
    trans = translator.translate(key, src = 'en',  dest = 'vi')
    if trans is None:
        return None
    return trans.text

print(translate("flower"))