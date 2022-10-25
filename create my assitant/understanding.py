import speech_recognition
robot_ear = speech_recognition.Recognizer()
with speech_recognition.Microphone() as source:
    print("Robot: I'm listening")
    audio_data = robot_ear.listen(source)

try:
    you = robot_ear.recognize_google(audio_data)
except:
    you = ""
print("You: " + you)
