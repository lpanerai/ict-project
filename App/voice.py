import functions #File esterno con el funzioni VAD e SR

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
DURATION_VAD = 3         #Ascolta per 3 secondi
DURATION_VR = 5          #Ascolta per 5 secondi
c = 0                #Contatore SV

#MAIN: Loop vocale
if check_microphone():
    c = 0
    
    while True:

        # Ascolta l'audio
        audio = listen_for_audio(DURATION_VAD, SAMPLE_RATE)

        # Verifica presenza di voce
        if vad_detect(audio, silero_vad, SAMPLE_RATE):
            print("\n----------------------------------------")
            print("Voce rilevata!")
            print("----------------------------------------\n")
            
            #Sezione Speacker Identification:
            audio_sv = listen_for_audio(DURATION_VR, SAMPLE_RATE)
            sd.wait()

            #Salva in File-Temporaneo
            temp_filename = "temp_audio.wav"
            save_audio_to_file(audio_sv, temp_filename, SAMPLE_RATE)

            #Inter.Server
            print("\n------------------------------------------")
            print("Invio al server...")
            print("------------------------------------------")
            response = send_audio_to_server(temp_filename, SERVER_URL)
            
            print(f"\nRisposta del server: {response}")

            if "Identity confermata" in response:
                print("\n------------------------------------------")
                print("Accesso consentito!")
                print("------------------------------------------")
                
                #Sezione Face Verification
                #Azioni...
                print("\nBentornato...")
                break  # Esci dal loop
            
            else:
                c += 1
                if c <= 3:
                    print(f"\nNessuna voce riconosciuta. Tentativo {c}/3. Riprovando...")
                    continue  #Riprova

                else:
                    print("\n--------------------------------------------")
                    print("Tentativi esauriti per l'identificazione.")
                    print("--------------------------------------------")
                    print("\nNew User?")
                    
                    max_attempts = 2  # Numero massimo di tentativi

                    while max_attempts > 0:
                        ans = input("\nDo you want to enroll? (Y/N): ")

                        if (ans == "Y" or ans == "y"):
                            print("\n--------------------------------------------")
                            print("Inserire Key per Enrollment:")
                            print("--------------------------------------------")

                            key = input("Key: ")

                            if (key == "pussy" or key == "dick"):
                                #Azione in caso di successo
                                print("\n--------------------------------------------")
                                print("Chiave accettata. Procedo con l'enrollment...")
                                print("--------------------------------------------")
                                
                                print("\nPhase-1: Voice...")
                                #Azione Voice

                                print("\nPhase-2: Face...")
                                #Azione Face

                                break #Esce dal ciclo in caso di successo

                            else:
                                max_attempts -= 1
                                if max_attempts > 0:
                                    print("\n--------------------------------------------")
                                    print(f"\nChiave errata. Hai ancora {max_attempts} tentativi.")
                                    print("--------------------------------------------")

                                else:
                                    print("\n--------------------------------------------")
                                    print("\nHai esaurito tutti i tentativi. Accesso negato.")
                                    print("--------------------------------------------")

                        else:
                            print("\n--------------------------------------------")   
                            print("\nOperazione annullata.")
                            print("--------------------------------------------")
                            break

                    else:
                        break  #Esco dal Ciclo Principale

        else:
            print("\n--------------------------------------------")
            print("Nessuna voce rilevata. Ricomincio...")
            print("--------------------------------------------")

else:
    print("\n----------------------------------------------------")
    print("ERRORE:Controllare Dispositivi: Microfono o Sensore")
    print("----------------------------------------------------")