import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchaudio.transforms as T
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition

#Face Enrollment
import dlib
from PIL import Image
from keras_facenet import FaceNet

#Face Recognition
import cv2
from scipy.spatial.distance import cosine

#Audio e Speech Recognition + Face Recognition
import numpy as np
import sounddevice as sd
import pyaudio
import soundfile as sf
import sys

from silero_vad import get_speech_timestamps

import requests
import wave
import time
import json
import os


#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Funzioni VAD
def vad_detect(audio, model, SAMPLE_RATE):
    """Esegue il Voice Activity Detection (VAD) sul segnale audio."""
    timestamps = get_speech_timestamps(audio, model, sampling_rate=SAMPLE_RATE)
    return len(timestamps) > 0  # True se c'è voce

def listen_for_audio(DURATION, SAMPLE_RATE):
    """Registra audio dal microfono per la durata specificata."""
    print("--------------------------------------")
    print("Ascoltando...")

    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  #Aspetta il termine della registrazione
    print("Recording Ended")
    sf.write("output_filename.wav", audio, SAMPLE_RATE)
    print("--------------------------------------")
    return (audio.flatten() * 32768).astype(np.int16)  # Conversione a int16
    

def check_microphone():
    """Controlla la disponibilità del microfono."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Microfono trovato: {device['name']}")
        return True
    print("Nessun microfono disponibile!")
    return False

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Funzioni SR
#Funzione per salvare l'audio registrato in un file temporaneo
def save_audio_to_file(audio_data, filename, samplerate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)

#Funzione per inviare il file al server
def send_audio_to_server(filename, SERVER_URL):
    with open(filename, 'rb') as f:
        files = {'file': f}
        response = requests.post(SERVER_URL, files=files)
        return response.text

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Enrollment Pt.1
#Funzione per inviare il file al server
def send_audio_name_to_server(filename, speaker_name, SERVER_URL):
    with open(filename, 'rb') as f:
        files = {'file': f}  # File da inviare
        data = {'speaker': speaker_name}  # Nome dello speaker da inviare come parte dei dati
        response = requests.post(SERVER_URL, files=files, data=data)  # Invia file e dati al server
        return response.text  # Restituisce la risposta del server

def parse_server_response(response):
    """
    Parsing della risposta del server in formato JSON.
    """
    try:
        response_data = json.loads(response)  # Converte la stringa JSON in un dizionario
        speaker_name = response_data.get("Speaker", "Unknown")
        score = response_data.get("Score", 0.0)
        Idy = response_data.get("Identification", "Unknown")
        
        return speaker_name, score, Idy
    
    except json.JSONDecodeError as e:
        print(f"Errore di parsing della risposta JSON: {e}")
        
        return "Unknown", 0.0, "Unknown"

def parse_server_response_enroll(response):
    """
    Parsing della risposta del server in formato JSON.
    """
    try:
        response_data = json.loads(response)  # Converte la stringa JSON in un dizionario
        status = response_data.get("Status", "Unknown")
        speaker_name = response_data.get("Speaker", "Unknown")
        
        return status, speaker_name
    
    except json.JSONDecodeError as e:
        print(f"Errore di parsing della risposta JSON: {e}")
        
        return "Unknown", "Unknown"

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Enrollment Voice
def enroll_user_voice(file_path, username, output_voice_dir, output_emb_dir, duration, sample_rate, num_recording):
    """
    Funzione di enrollment: salva file audio di riferimento e li invia al server.
    """

    print(f"\n--- Block Number: {num_recording} ---")
    file_json= json.load(open(file_path))
    print(str(file_json[f"chunk{num_recording}"]))

    #Countdown
    for i in range(3, 0, -1):
        print(f"{i}")
        time.sleep(2)

    print("\nStart the recording...")

    # Registra audio
    audio_data = listen_for_audio(duration, sample_rate)

    # Salva il file audio localmente
    
    audio_file_path= os.path.join(output_voice_dir, f"Ref_{username}_{num_recording}.wav")
    #audio_data_int16 = (audio_data.flatten() * 32767).astype(np.int16)  # Conversione a int16
    sf.write(audio_file_path, audio_data,sample_rate)
    print(f"File saved in: {audio_file_path}")

    calculate_embedding(audio_file_path,output_emb_dir,username,num_recording)

    print(f"Utente {username} registrato con successo!")

    print(f"\nPhase-{num_recording}: Voice Enrollment --> COMPLETED")

def calculate_embedding(file_path,output_dir, username, num_recording):
    """
    Funzione per calcolare l'embedding di un file audio e salvarlo.
    """
    # Carica il modello di riconoscimento speaker
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # Carica il file audio
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Calcola l'embedding del file audio
    embedding = classifier.encode_batch(waveform)

    #Salva l'embedding in un file npy
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{username}_{num_recording}.npy"), embedding)
    return True

def identify_speaker(file_path,input_emb_dir, DURATION_VR, SAMPLE_RATE, threshold):
    """
    Funzione per identificare lo speaker in un file audio.
    """
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    audio_sv = listen_for_audio( DURATION_VR, SAMPLE_RATE)
    sd.wait()
    temp_filepath = os.path.join(file_path,"temp_audio.wav")
    save_audio_to_file(audio_sv, temp_filepath, SAMPLE_RATE)

    calculate_embedding(temp_filepath,file_path,"temp",1)

    #Invio al server per identificazione
    print("\n------------------------------------------")
    print("Registrazione salvata, Inizio identificazione...")
    print("------------------------------------------")
    
    registered_voices= load_registered_voices(input_emb_dir)
    print("Embeddings caricati:", registered_voices.keys())
    
    max_score = float('-inf')  # Punteggio massimo iniziale
    best_match = None  # Username associato al punteggio massimo
    
    tmp_np_path= os.path.join(file_path,"temp_1.npy")
    tmp_np=np.load(tmp_np_path)
    tmp_torch= torch.tensor(tmp_np)

    # Identifica lo speaker confrontando con gli embedding registrati
    for username, registered_encoding in registered_voices.items():
        tmp_torch_enrolled_user= torch.tensor(registered_encoding)
        
        score = verification.similarity(tmp_torch,tmp_torch_enrolled_user).item()
        print(f"Utente: {username}, Score: {score}")
        
        # Aggiorna il punteggio massimo se necessario
        if score > max_score :
            max_score = score
            best_match = username

    # Stampa l'username con il punteggio più alto e maggiore della soglia
    print("\n------------------------------------------")
    if (best_match is not None) and (max_score > threshold): 
        print(f"Speaker identificato: {best_match} con score: {max_score}")
        return True
    else:
        print("Nessuno speaker identificato.")
        return False
    print("------------------------------------------")
    

def load_registered_voices(path):
    """Carica gli embedding registrati da file .npy."""
    embeddings = {}
    for file in os.listdir(path):
        if file.endswith(".npy"):
            username = os.path.splitext(file)[0]
            embeddings[username] = np.load(os.path.join(path, file))
    return embeddings

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Face Enrollment
def enroll_user_face(username, image_file, save_path):
    try:
        # Inizializza il modello FaceNet
        embedder = FaceNet()

        # Carica l'immagine con Pillow
        img = Image.open(image_file).convert("RGB")
        image = np.array(img)

        # Ottieni gli embedding con FaceNet
        face_embeddings = embedder.embeddings([image])
        if len(face_embeddings) == 0:
            print("Errore: nessun volto rilevato nell'immagine.")
            return False

        # Ottieni il primo embedding (poiché è garantito che ci sia solo un volto nell'immagine)
        face_encoding = face_embeddings[0]

        # Salva l'embedding in un file npy
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f"{username}.npy"), face_encoding)
        print(f"Utente {username} registrato con successo!")
        return True
    except Exception as e:
        print(f"Errore durante la registrazione dell'utente: {e}")
        return False

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Face Recognition
def load_registered_faces(path="ict-project\\Dataset\\People\\Embedding\\Face"):
    """Carica gli embedding registrati da file .npy."""
    embeddings = {}
    for file in os.listdir(path):
        if file.endswith(".npy"):
            username = os.path.splitext(file)[0]
            embeddings[username] = np.load(os.path.join(path, file))
    return embeddings

def recognize_face_live(input_embedding,threshold):
    """Riconoscimento facciale live utilizzando FaceNet e similarità coseno."""
    # Inizializza il modello FaceNet
    embedder = FaceNet()

    # Carica gli embedding registrati
    registered_faces = load_registered_faces(input_embedding)
    print("Embeddings caricati:", registered_faces.keys())

    # Avvia la webcam
    cap = cv2.VideoCapture(0)  # 0 indica la webcam predefinita
    print("Premi 'q' per uscire.")

    is_user_recognized = False

    while True:
        # Leggi un fotogramma dalla webcam
        ret, frame = cap.read()
        if not ret:
            print("Errore nell'acquisizione del video.")
            break

        # Converti il frame in RGB per FaceNet
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Mostra il video live
        cv2.imshow("Riconoscimento Facciale", frame)

        # Ottieni gli embedding per il frame attuale
        face_embeddings = embedder.embeddings([rgb_frame])

        for face_embedding in face_embeddings:
            # Confronta l'embedding con quelli registrati usando la similarità coseno
            for username, registered_encoding in registered_faces.items():
                similarity = 1 - cosine(registered_encoding, face_embedding)
                print(f"Similarità con {username}: {similarity}")
                if similarity > threshold:
                    # Disegna un rettangolo attorno al volto riconosciuto
                    left, top, right, bottom = 50, 50, 150, 150  # Aggiusta le coordinate
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{username}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    print(f"Utente riconosciuto: {username}")
                    is_user_recognized = True

        # Esci con il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if is_user_recognized:
            # Rilascia la webcam e chiudi le finestre
            cap.release()
            return True
    # Rilascia la webcam e chiudi le finestre
    cap.release()
    cv2.destroyAllWindows()

    