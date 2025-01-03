#Funzioni VAD
def vad_detect(audio, model, SAMPLE_RATE):
    """Esegue il Voice Activity Detection (VAD) sul segnale audio."""
    timestamps = get_speech_timestamps(audio, model, sampling_rate=SAMPLE_RATE)
    return len(timestamps) > 0  # True se c'è voce

def listen_for_audio(DURATION, SAMPLE_RATE):
    """Registra audio dal microfono per la durata specificata."""
    print("Ascoltando...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Aspetta il termine della registrazione
    return (audio.flatten() * 32768).astype(np.int16)  # Converti in formato int16 per Silero

def check_microphone():
    """Controlla la disponibilità del microfono."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Microfono trovato: {device['name']}")
            return True
    print("Nessun microfono disponibile!")
    return False

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
