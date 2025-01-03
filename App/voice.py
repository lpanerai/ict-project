import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import sounddevice as sd
import sys

from silero_vad import get_speech_timestamps

import requests
import wave

#Device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#VAD:
#Scaricamento Modello
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    source='github'
)

#Silero Model
silero_vad = model.to(device)

#Utils
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

#Speaker Rec.
#Comandi per esecuzione:
#Eseguire il server in Bash: cmd --> node server.js
SERVER_URL = "http://localhost:3000/upload"

#MAIN -- Loop Vocale
#Configurazione microfono
SAMPLE_RATE = 16000  #Silero VAD funziona con 16 kHz
DURATION = 3         #Ascolta per 5 secondi
c = 0                #Contatore SV

if check_microphone():
    while True:
        c = 0

        # Ascolta l'audio
        audio = listen_for_audio(DURATION, SAMPLE_RATE)

        # Verifica presenza di voce
        if vad_detect(audio, silero_vad, SAMPLE_RATE):
            print("\nVoce rilevata!")
            
            #Sezione Speacker Identification:
            audio_sv = listen_for_audio(DURATION, SAMPLE_RATE)
            sd.wait()

            #Salva in File-Temporaneo
            temp_filename = "temp_audio.wav"
            save_audio_to_file(audio_sv, temp_filename, SAMPLE_RATE)

            #Inter.Server
            print("\nInvio al server...")
            response = send_audio_to_server(temp_filename, SERVER_URL)
            
            print(f"\nRisposta del server: {response}")

            if "Identity confermata" in response:
                print("Accesso consentito!")
                
                break  # Esci dal loop
            
            else:
                c += 1
                if c <= 3:
                    print("\nNessuna voce riconosciuta, riprovo...")
                    continue #Ricomincio Ciclo Speak.Verific.
                else:
                    print("\nTentativi esauriti per l'identificazione.")
                    break  #Esco dal Ciclo Speak.Verific.
        
        else:
            print("Nessuna voce rilevata. Ricomincio...")
