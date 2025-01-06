import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Face Enrollment
import dlib
from PIL import Image

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

#Generic Functions:
def print_menu():
    print("\n" + "="*40)
    print("   Welcome to the System: Figo-32cm")
    print("="*40)
    print("\nPlease, choose an option:")
    print(" 1) System Activation")
    print(" 2) New User Enrollment")
    print(" 3) End the Program")
    print("\n" + "="*40)

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
#Enrollment Pt.1
#Lettura File.txt e invio server
def read_text_file_in_chunks(file_path):
    """
    Legge un file di testo e restituisce un generatore che fornisce paragrafi.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph = []
        for line in file:
            if line.strip():  # Se la riga non è vuota, aggiungila al paragrafo corrente
                paragraph.append(line.strip())
            else:
                if paragraph:  # Se il paragrafo contiene testo, restituiscilo
                    yield " ".join(paragraph)
                    paragraph = []
        if paragraph:  # Restituisci l'ultimo paragrafo, se esiste
            yield " ".join(paragraph)

def enroll(file_path, name, output_dir, duration, sample_rate, server_url):
    """
    Funzione di enrollment: salva file audio di riferimento e li invia al server.
    """
    # Crea una cartella specifica per lo speaker
    speaker_dir = os.path.join(output_dir, name)
    os.makedirs(speaker_dir, exist_ok=True)

    # Leggi il file di testo in blocchi
    text_chunks = read_text_file_in_chunks(file_path)
    chunk_number = 1

    for chunk in text_chunks:
        print(f"\n--- Block Number: {chunk_number} ---")

        #Suddivide il testo in base ai punti e aggiunge a capo
        sentences = chunk.split('.')
        formatted_text = "\n".join(sentence.strip() + '.' for sentence in sentences if sentence.strip())

        print(formatted_text)
        
        #Countdown
        for i in range(3, 0, -1):
            print(f"{i}")
            time.sleep(2)

        print("\nStart the recording...")

        # Registra audio
        audio_data = listen_for_audio(duration, sample_rate)

        # Salva il file audio localmente
        audio_file_name = os.path.join(output_dir, name, f"Ref_{name}_{chunk_number}.wav")
        #audio_data_int16 = (audio_data.flatten() * 32767).astype(np.int16)  # Conversione a int16
        sf.write(audio_file_name, audio_data, sample_rate)
        print(f"File saved in: {audio_file_name}")

        # Invia il file audio al server
        #print(f"\nInvio del file {audio_file_name} al server per l'enrollment...")
        #with open(audio_file_name, 'rb') as audio_file:
        #    response = requests.post(
        #        url=f"{server_url}/enroll",
        #        files={"file": audio_file},
        #        data={"speaker": name}
        #    )

        response = send_audio_name_to_server(audio_file_name, name ,server_url + "/enroll")
        print(f"Server Response: {response}")

        status, speaker = parse_server_response_enroll(response)
        print(status)
        print(speaker)

        #if response.status_code == 200:
        #    print(f"Server Response: {response.text}")
        #else:
        #    print(f"Errore durante l'invio al server: {response.status_code} - {response.text}")

        # Avanza al prossimo blocco di testo
        chunk_number += 1

        # Pausa tra le registrazioni
        time.sleep(2)

    print("\nPhase-1: Voice Enrollment --> COMPLETED")

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#Face Enrollment
def enroll_user(username, image_file, save_path="ict-project\\Dataset\\People\\Embeddings"):
    try:
        # Percorso ai modelli pre-addestrati di dlib
        path_predictor = '.venv\\Lib\\site-packages\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat'
        path_recognition = '.venv\\Lib\\site-packages\\face_recognition_models\\models\\dlib_face_recognition_resnet_model_v1.dat'

        # Carica i modelli
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path_predictor)
        face_rec_model = dlib.face_recognition_model_v1(path_recognition)


        # Usa Pillow per caricare l'immagine e convertirla in un array NumPy
        img = Image.open(image_file).convert("RGB")
        image = np.array(img)
        
        # Rileva i volti nell'immagine
        dets = detector(image, 1)
        if len(dets) != 1:
            print("Errore: l'immagine deve contenere esattamente un volto.")
            return False

        # Calcola i punti di riferimento facciali
        shape = predictor(image, dets[0])
        
        # Calcola l'embedding facciale
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)

        # Converte l'embedding in un array numpy
        face_encoding = np.array(face_descriptor)
        
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
def load_registered_faces(path="ict-project\\Dataset\\People\\Embeddings"):
    """Carica gli embedding registrati da file .npy."""
    embeddings = {}
    for file in os.listdir(path):
        if file.endswith(".npy"):
            username = os.path.splitext(file)[0]
            embeddings[username] = np.load(os.path.join(path, file))
    return embeddings

def recognize_face_live(threshold=0.6):
    """Riconoscimento facciale live utilizzando solo `dlib` e similarità coseno."""
    # Percorso ai modelli pre-addestrati di dlib
    path_predictor = '.venv\\Lib\\site-packages\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat'
    path_recognition = '.venv\\Lib\\site-packages\\face_recognition_models\\models\\dlib_face_recognition_resnet_model_v1.dat'

    # Inizializza i modelli
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_predictor)
    face_rec_model = dlib.face_recognition_model_v1(path_recognition)

    # Carica gli embedding registrati
    registered_faces = load_registered_faces()
    print("Embeddings caricati:", registered_faces.keys())

    # Avvia la webcam
    cap = cv2.VideoCapture(0)  # 0 indica la webcam predefinita
    #print("Premi 'q' per uscire.")

    #Inizializzo la variabile user_found a False (serve per uscire dal ciclo while)
    user_found = False

    while True:
        
        # Leggi un fotogramma dalla webcam
        ret, frame = cap.read()
        if not ret:
            print("Errore nell'acquisizione del video.")
            break

        # Converti il frame da BGR (OpenCV) a RGB (dlib)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Rileva i volti nel frame
        detections = detector(rgb_frame)

        for det in detections:
            # Trova i punti di riferimento del viso
            shape = predictor(rgb_frame, det)
            # Calcola gli embedding
            face_embedding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))

            # Confronta l'embedding con quelli registrati usando la similarità coseno
            for username, registered_encoding in registered_faces.items():
                similarity = 1 - cosine(registered_encoding, face_embedding)
                print(f"Similarità con {username}: {similarity}")
                if similarity > threshold:
                    # Disegna un rettangolo attorno al volto riconosciuto
                    left, top, right, bottom = det.left(), det.top(), det.right(), det.bottom()
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{username}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    print(f"Utente riconosciuto: {username}")
                    user_found = True
                    return True
        
        if user_found:
            break
        
        # Mostra il video live
        cv2.imshow("Riconoscimento Facciale", frame)

        # Esci con il tasto 'q'
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break

    # Rilascia la webcam e chiudi le finestre
    cap.release()
    cv2.destroyAllWindows()