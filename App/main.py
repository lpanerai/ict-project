from utils import *
import os
import glob
import argparse
from silero_vad import get_speech_timestamps

#Speaker Rec.
#Comandi per esecuzione:
#Eseguire il server in Bash: cmd --> node server.js

#MAIN -- Loop Vocale
#Configurazione microfono
SAMPLE_RATE = 16000  #Silero VAD funziona con 16 kHz
DURATION_VAD = 3         #Ascolta per 3 secondi
DURATION_VR = 5          #Ascolta per 5 secondi
DURATION_EN = 35         #Ascolta per 30 secondi
c = 0                    #Contatore SV
exit_flag = True         #Metodo per terminare il ciclo 

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    source='github'
    )

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


silero_vad = model.to(device)

def pipeline():
    print(device)
    while True:
            print("Microfono connesso")

            audio = listen_for_audio(DURATION_VAD, SAMPLE_RATE)
            if(VAD(audio,silero_vad,SAMPLE_RATE)):
                print("VAD: Utente sta parlando")
                if(speekerRecognition()):
                    print("SpeakerRecognition: Utente riconosciuto")
                    if(faceRecognition()):
                        print("FaceRecognition: Utente riconosciuto")
                        print("Accesso consentito! Benvenuto.")
                        break
                    else:
                        print("FaceRecognition: Utente non riconosciuto")
                else:
                    print("SpeakerRecognition: Utente non riconosciuto")
            else:
                print("VAD: Utente non sta parlando")

def VAD(audio,silero_vad,SAMPLE_RATE):
    return vad_detect(audio, silero_vad, SAMPLE_RATE)
	
def speekerRecognition():
    output_emb_dir = f"C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\"
    input_emb_dir = f"C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\People\\Embedding\\Voice\\"
    return identify_speaker(output_emb_dir,input_emb_dir, DURATION_VR, SAMPLE_RATE, threshold=0.6)

def faceRecognition():
    return recognize_face_live(threshold=0.65)


if __name__ == "__main__":
    #voice_enrollment(username="Leonardo")
    #speekerRecognition()
    #pipeline()
    #face_enrollment("Leonardo")
    faceRecognition()
    