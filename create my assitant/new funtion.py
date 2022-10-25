import wikipedia
import os
''''''
def startGoogle():
    global run
    run = False
    #speak('starting google')        #dòng này có cũng được không có cũng không sao :v

    os.startfile('C:\Program Files (x86)\Google\Chrome\Application\chrome.exe')  # mở google (và các file khác)
''''''
def playmusic():
    global run
    music_dir = 'D:\IELTS\audio Hacker Ielts\HACKERS IELTS LISTENING_MP3-20211123T142631Z-001\HACKERS IELTS LISTENING_MP3\Chapter 01'          # thư mục để nhạc
    songs = os.listdir(music_dir)
    os.startfile(os.path.join(music_dir,songs[0]))
    run = False
'''
def wiki(keyword):
    #speak('searching on wikipedia')
    try:
        print('**Result from wiki**')
        print(wikipedia.summary(keyword))       # search wiki
    except:
        print('Some error occurred! Try again.')
        print('')
    run = False

data = "Facebook"
wiki(data) #đưa vào tham số 'data' là từ khóa cần tìm
'''
#startGoogle()
playmusic()

