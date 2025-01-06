from utils import *
import os
import glob
import argparse
from silero_vad import get_speech_timestamps

#Speaker Rec.
#Comandi per esecuzione:
#Eseguire il server in Bash: cmd --> node server.js
SERVER_URL = "http://localhost:3000/"

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

model_save_path = "C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Models\\silero_vad_model.pth"
torch.save(model.state_dict(), model_save_path)
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
                    else:
                        print("FaceRecognition: Utente non riconosciuto")
                else:
                    print("SpeakerRecognition: Utente non riconosciuto")
            else:
                print("VAD: Utente non sta parlando")

def VAD(audio,silero_vad,SAMPLE_RATE):
    vad_detect(audio, silero_vad, SAMPLE_RATE)
	
def speekerRecognition():
    audio_sv = listen_for_audio(DURATION_VR, SAMPLE_RATE)
    sd.wait()
    temp_filename = "temp_audio.wav"
    save_audio_to_file(audio_sv, temp_filename, SAMPLE_RATE)

    #Invio al server per identificazione
    print("\n------------------------------------------")
    print("Invio al server per identificazione...")
    print("------------------------------------------")
    response = send_audio_to_server(temp_filename, SERVER_URL + "/identify")

    print(f"\nRisposta del server: {response}")
    speaker_name, score, Idy = parse_server_response(response)
    print(f"Score: {score}, Idy: {Idy}")

    if Idy == "Allowed":
        if score >= 0.5:  # Soglia di riconoscimento
            print("\n------------------------------------------")
            print(f"Accesso consentito! Benvenuto, {speaker_name}.")
            print("------------------------------------------")
            return True
        else:
            print("\n------------------------------------------")
            print("Accesso negato! Utente non riconosciuto.")
            print("------------------------------------------")
            return False
    else:
           return False

def faceRecognition():
    recognize_face_live(threshold=0.6)

def voice_enrollment():
	return True

def face_enrollment():
    enroll_user("Leonardo", "Dataset/People/Leonardo/Photos/Leonardo.jpg")

if __name__ == "__main__":
    pipeline()