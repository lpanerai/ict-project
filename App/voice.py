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
import soundfile as sf
import sys

from silero_vad import get_speech_timestamps

import requests
import wave
import time
import json
import os

#Device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PATH vari:
path_txt = "C:\\Users\\giorg\\Downloads\\ict_work\\enroll.txt"
path_dir = "C:\\Users\\giorg\\Downloads\\ict_work\\Server\\uploads\\references\\"
enroll_path = path_dir

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
SERVER_URL = "http://localhost:3000/"

#MAIN -- Loop Vocale
#Configurazione microfono
SAMPLE_RATE = 16000  #Silero VAD funziona con 16 kHz
DURATION_VAD = 3         #Ascolta per 3 secondi
DURATION_VR = 5          #Ascolta per 5 secondi
DURATION_EN = 35         #Ascolta per 30 secondi
c = 0                    #Contatore SV
exit_flag = True         #Metodo per terminare il ciclo

#MAIN: Loop vocale
while exit_flag == True:
    
    print_menu()
    
    choice = input("Choice: ")
    if choice.isdigit() and choice in ["1", "2", "3"]:
        choice = int(choice)  # Converti in intero solo se la scelta è valida
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

    if choice == 1:

        #MAIN: Loop vocale
        if check_microphone():
            c = 0  # Contatore dei tentativi falliti
    
            while True:

                #Ascolta l'audio
                audio = listen_for_audio(DURATION_VAD, SAMPLE_RATE)

                #Verifica presenza di voce
                if vad_detect(audio, silero_vad, SAMPLE_RATE):
                    print("\n----------------------------------------")
                    print("Voce rilevata!")
                    print("----------------------------------------\n")
            
                    #Sezione Speaker Identification:
                    audio_sv = listen_for_audio(DURATION_VR, SAMPLE_RATE)
                    sd.wait()

                    #Salva in File-Temporaneo
                    temp_filename = "temp_audio.wav"
                    save_audio_to_file(audio_sv, temp_filename, SAMPLE_RATE)

                    #Invio al server per identificazione
                    print("\n------------------------------------------")
                    print("Invio al server per identificazione...")
                    print("------------------------------------------")
            
                    response = send_audio_to_server(temp_filename, SERVER_URL + "/identify")
            
                    #Parsing della risposta
                    print(f"\nRisposta del server: {response}")

                    speaker_name, score, Idy = parse_server_response(response)
                    print(score)
                    print(Idy)

                    if Idy == "Allowed":
                        if score >= 0.5:  # Soglia di riconoscimento
                            print("\n------------------------------------------")
                            print(f"Accesso consentito! Benvenuto, {speaker_name}.")
                            print("------------------------------------------")
                    
                            #Sezione Face Verification
                            print("\nBentornato...")
                            exit_flag = False  # Esci dal loop
                        
                        else:
                            print(f"\nSpeaker non riconosciuto con sufficiente accuratezza. (Score: {score})")
                            continue
                    else:
                        print("\nSpeaker non riconosciuto.")

                    #Incremento dei tentativi
                    c += 1
                    if c <= 2:
                        print(f"\nNessuna voce riconosciuta. Tentativo {c}/2. Riprovando...")
                        continue  # Riprova

                    else:
                        print("\n--------------------------------------------")
                        print("Tentativi esauriti per l'identificazione.")
                        print("--------------------------------------------")
                        print("\nNew User?")
                
                        #Fase Enrollment
                        max_attempts = 2  # Numero massimo di tentativi

                        while max_attempts > 0:
                            ans = input("\nDo you want to enroll? (Y/N): ")

                            if (ans == "Y" or ans == "y"):
                                print("\n--------------------------------------------")
                                print("Inserire Key per Enrollment:")
                                print("--------------------------------------------")

                                key = input("Key: ")

                                if (key == "pussy" or key == "dick"):
                                    print("\n--------------------------------------------")
                                    print("Chiave accettata. Procedo con l'enrollment...")
                                    print("--------------------------------------------")
                            
                                    print("\nPhase-1: Voice...\n")
                                    #Azione Voice
                                    name = input("\nInserisci il tuo nome: ").strip()

                                    #Funzione Enroll (Invia al server)
                                    print("\nRegistrazione e invio al server...")
                                    enroll(path_txt, name, enroll_path, DURATION_EN, SAMPLE_RATE, SERVER_URL)

                                    print("\n--------------------------------------------")
                                    print("Enrollment completato!")
                                    print("--------------------------------------------")

                                    break  # Esce dal ciclo enrollment

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
                            break  # Esco dal ciclo principale
                else:
                    print("\n--------------------------------------------")
                    print("Nessuna voce rilevata. Ricomincio...")
                    print("--------------------------------------------")

        else:
            print("\n----------------------------------------------------")
            print("ERRORE:Controllare Dispositivi: Microfono o Sensore")
            print("----------------------------------------------------")

    elif choice == 2:
        #EXT --> Enrolment Phase
        print("New User Enrollment Phase")

        #Fase Enrollment
        max_attempts = 2  # Numero massimo di tentativi

        while max_attempts > 0:
            
            ans = input("\nDo you want to enroll? (Y/N): ")

            if (ans == "Y" or ans == "y"):
                print("\n--------------------------------------------")
                print("Inserire Key per Enrollment:")
                print("--------------------------------------------")

                key = input("Key: ")

                if (key == "pussy" or key == "dick"):
                    print("\n--------------------------------------------")
                    print("Chiave accettata. Procedo con l'enrollment...")
                    print("--------------------------------------------")
                            
                    print("\nPhase-1: Voice...\n")
                    #Azione Voice
                    name = input("\nInserisci il tuo nome: ").strip()

                    #Funzione Enroll (Invia al server)
                    print("\nRegistrazione e invio al server...")
                    enroll(path_txt, name, enroll_path, DURATION_EN, SAMPLE_RATE, SERVER_URL)

                    print("\n--------------------------------------------")
                    print("Enrollment completato!")
                    print("--------------------------------------------")

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
                print("\n----------------------------------------")   
                print("\nOperazione annullata.")
                print("----------------------------------------")
                break
                
    elif choice == 3:
        #Exit System
        print("\n----------------------------------------") 
        print("Thank You for your time asshole")
        print("----------------------------------------") 
        break

    else:
        #Not Valid Option
        print("\n--------------------------------------------")
        print("Not Valid! Please select 1/2/3.")
        print("--------------------------------------------")