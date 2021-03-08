'''
Simply plays a few second long audio clip for use in indicating when a 
program is done running.  Options are "Non-Stop" from Hamilton, "Hallelujah"
and a simple alert chime
'''

import subprocess


def nonstop():
    '''Plays a few seconds of Non-Stop from Hamilton'''
    print('\nDone')
    audio_file = "/users/bob/documents/python_library/donemusic/Non-Stop_SFX.wav"
    return_code = subprocess.call(["afplay", audio_file])

def hallelujah():
    '''Plays a few seconds of the Hallelujah chorus'''
    print('\nDone')
    audio_file = "/users/bob/documents/python_library/donemusic/Hallelujah_SFX.wav"
    return_code = subprocess.call(["afplay", audio_file])

def chime():
    '''Plays a chime that sounds like the airplane seatbelt alert'''
    print('\nDone')
    audio_file = "/Users/bob/Documents/python_library/donemusic/Store_Door_Chime-Mike_Koenig-570742973.wav"
    return_code = subprocess.call(["afplay", audio_file])