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
    input_embedding="C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\People\\Embedding\\Face\\"
    return recognize_face_live(input_embedding,threshold=0.90)

def voice_enrollment(username):
    path_txt = "C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\enroll.json"
    output_emb_dir = "C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\People\\Embedding\\Voice\\"
    output_voice_dir = f"C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\People\\{username}\\Voice\\"
    enroll_user_voice(path_txt,username,output_voice_dir, output_emb_dir ,DURATION_EN, SAMPLE_RATE, num_recording=1)
    enroll_user_voice(path_txt,username ,output_voice_dir, output_emb_dir, DURATION_EN, SAMPLE_RATE, num_recording=2)
    enroll_user_voice(path_txt,username,output_voice_dir, output_emb_dir, DURATION_EN, SAMPLE_RATE, num_recording=3)
    

def face_enrollment(username):
    image_path=f"C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\People\\{username}\\Photos\\{username}.jpg"
    output_emb_dir="C:\\Users\\4k\\Documents\\Università\\2°anno\\ict-project\\Database\\People\\Embedding\\Face\\"
    return enroll_user_face(username, image_path, output_emb_dir)

if __name__ == "__main__":
    #voice_enrollment(username="Leonardo")
    #speekerRecognition()
    #pipeline()
    #face_enrollment("Leonardo")
    faceRecognition()
    