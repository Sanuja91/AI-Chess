import os, requests

def validate_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def notify(title, message = 'Just because I have to put something', priority = -1, sound = 'classical'):
    data = {
        'token' : 'avt8nb95eu37ct1nzq2f61maestuo2',
        'user' : 'u3y9och8os9zntmvdrzzbqemonu47q',
        'message' : str(message),
        'title' : str(title),
        'priority' : priority,
        'sound' : sound
    }

    response = requests.post('https://api.pushover.net/1/messages.json', data = data)
    print('Notifing', title, message, str(response))