from ast import keyword
import speech_recognition
import pyttsx3
from datetime import date
from datetime import datetime
import wikipedia
import os

robot_brain = ""
robot_mouth = pyttsx3.init()
robot_ear = speech_recognition.Recognizer()

def wiki(keyword):
    #speak('searching on wikipedia')
    try:
        print('**Result from wiki**')
        print("Robot_brain: " + wikipedia.summary(keyword))      # search wiki
    except:
        print("robot_brain: Some error occurred! Try again.")
    run = False

while True:
    with speech_recognition.Microphone() as source:
        print("Robot: I'm listening")
        audio_data = robot_ear.listen(source)
    print("Robot:...")
    try:
        you = robot_ear.recognize_google(audio_data)
    except:
        you = ""
    print("You: " + you)

    if "hello" in you:
        robot_brain = "Hello Nga"
    elif you == "":
        robot_brain = "I can't hear you, try again"
    elif "today" in you:
        today = date.today()
        robot_brain = today.strftime("%B %d, %Y")
    elif "time" in you:
        now = datetime.now()
        robot_brain = now.strftime("%H hours %M minutes %S seconds")
    elif "boyfriend" in you:
        robot_brain = "Mai Duc Giang"
    elif "best friend" in you:
        robot_brain = "Doan Duc Thang"
    elif "bye" in you:
        robot_brain = "bye Nga"
        print("Robot_brain: " + robot_brain)
        robot_mouth.say(robot_brain)
        robot_mouth.runAndWait()
        break
    
    else:
        wiki(you)
        #robot_brain = "I'm fine thank you and you"

    if (robot_brain != ""):
        print("Robot_brain: " + robot_brain)
        robot_mouth.say(robot_brain)
        robot_mouth.runAndWait()
    
