import sys
import torch
import torchaudio
import os
from speechbrain.inference import SpeakerRecognition
import json
import torchaudio.transforms as T

def preprocess_audio(audio, sampling_rate=16000):
    # Normalizza il volume
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val

    # Rimuove il rumore di fondo
    vad = T.Vad(sample_rate=sampling_rate)
    audio = vad(audio.unsqueeze(0)).squeeze(0)
    return audio

#Funzione per la lettura dei file audio
def read_audio_torchaudio(file_path, sampling_rate=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != sampling_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        waveform = resample_transform(waveform)
    return waveform.squeeze(0)  # Rimuove la dimensione del batch

# Percorsi dei file audio
uploaded_file = sys.argv[1]
reference_files = sys.argv[2:]

# Carica il modello di Speaker Recognition
#print("\nCaricamento del modello di Speaker Verification...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model",
    use_auth_token=False,  # Se necessario per modelli privati
)

# Leggi il file caricato
uploaded_audio = read_audio_torchaudio(uploaded_file, sampling_rate=16000)
uploaded_audio = preprocess_audio(uploaded_audio)
uploaded_audio = torch.nn.functional.normalize(uploaded_audio, dim=0)

# Calcola lo score per ciascun file di riferimento
scores = []
THRESHOLD = 0.7  # Soglia di identificazione

for reference_file in reference_files:
    # Leggi il file di riferimento
    reference_audio = read_audio_torchaudio(reference_file, sampling_rate=16000)
    reference_audio = preprocess_audio(reference_audio)
    reference_audio = torch.nn.functional.normalize(reference_audio, dim=0)

    # Calcola la similarità tra il file caricato e il file di riferimento
    score, prediction = verification_model.verify_batch(reference_audio, uploaded_audio)
    #print(score[0].item())  # Stampa il risultato per ogni confronto
    scores.append(score[0].item())

# Calcola la media degli score
average_score = sum(scores) / len(scores)
max_score = max(scores)

# Stampa un JSON con i risultati
output = {
    "average_score": average_score,
    "max_score": max_score,
    "status": "confirmed" if max_score > THRESHOLD else "not_confirmed"
}

print(json.dumps(output))