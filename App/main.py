from utils import *
import os
from silero_vad import get_speech_timestamps
import numpy as np
import sounddevice as sd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#MAIN -- Loop Vocale
#Configurazione microfono
SAMPLE_RATE = 16000  #Silero VAD funziona con 16 kHz
DURATION_VAD = 3         #Ascolta per 3 secondi
DURATION_VR = 5          #Ascolta per 5 secondi
DURATION_EN = 35         #Ascolta per 30 secondi
c = 0                    #Contatore SV
exit_flag = True         #Metodo per terminare il ciclo 


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    source='github'
    )
device = torch.device("cpu")
silero_vad = model.to(device)



def generate_beep(frequency=440, duration=0.3, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t)
    sd.play(beep, sample_rate)
    sd.wait()

def pipeline():
    print(device)
    while True:
        print("Microfono connesso")

        audio = listen_for_audio(DURATION_VAD, SAMPLE_RATE)
        if(VAD(audio, model, SAMPLE_RATE)):
            print("VAD: Utente sta parlando")
            generate_beep()
            if(speekerRecognition()):
                print("SpeakerRecognition: Utente riconosciuto")
                generate_beep()
                generate_beep()
                if(faceRecognition()):
                    print("FaceRecognition: Utente riconosciuto")
                    generate_beep()
                    generate_beep()
                    generate_beep(duration=1)
                    print("Accesso consentito! Benvenuto.")
                    break
                else:
                    print("FaceRecognition: Utente non riconosciuto")
            else:
                print("SpeakerRecognition: Utente non riconosciuto")
                generate_beep()
        else:
            print("VAD: Utente non sta parlando")

def VAD(audio,model,SAMPLE_RATE):
    """Esegue il Voice Activity Detection (VAD) sul segnale audio."""
    timestamps = get_speech_timestamps(audio, model, sampling_rate=SAMPLE_RATE)

    return len(timestamps) > 0  # True se c'è voce
	
def speekerRecognition():
    return identify_speaker(BASE_DIR,DURATION_VR, SAMPLE_RATE, threshold=0.6)

def faceRecognition():
    return recognize_face_live(threshold=0.65)


if __name__ == "__main__":
    #voice_enrollment(username="Leonardo")
    #speekerRecognition()
    pipeline()
    #face_enrollment("Leonardo")
    #faceRecognition()
    