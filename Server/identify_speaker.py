import sys
import json
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from speechbrain.pretrained import SpeakerRecognition

def remove_silence(waveform, threshold_db=-40):
    # Calcola l'energia (potenza) del segnale
    energy = torch.norm(waveform, dim=1)
    # Applica una soglia per eliminare i silenzi
    non_silent_indices = energy > torch.tensor(10**(threshold_db / 10), dtype=torch.float32)
    waveform_cleaned = waveform[:, non_silent_indices]
    return waveform_cleaned

def remove_noise(waveform, sample_rate):
    # Applicazione di un filtro bandpass (ad esempio, tra 300Hz e 3400Hz per la voce)
    transform = T.BandpassFilter(low_freq=300, high_freq=3400, sample_rate=sample_rate)
    return transform(waveform)


#Threshold Valore
THRESHOLD = 0.5

# Percorso al file caricato
uploaded_file = sys.argv[1]

# Carica il modello
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model"
)

# Leggi il file audio e converti in tensore
waveform, sample_rate = torchaudio.load(uploaded_file)
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

waveform = remove_noise(waveform, sample_rate)
waveform = remove_silence(waveform)

# Calcola l'embedding del file caricato
uploaded_embedding = verification_model.encode_batch(waveform).detach().cpu().numpy()

# Carica gli embedding salvati
embedding_file = "speaker_embeddings.json"
with open(embedding_file, "r") as file:
    embeddings = json.load(file)

# Calcola la similarità media con ogni speaker
similarities = {}
for speaker, ref_embeddings in embeddings.items():
    # Media degli embedding dello speaker
    ref_embeddings = np.array(ref_embeddings)
    ref_mean_embedding = np.mean(ref_embeddings, axis=0)

    # Calcola la similarità
    score = verification_model.similarity(torch.tensor(uploaded_embedding), torch.tensor(ref_mean_embedding)).item()
    similarities[speaker] = score

# Trova lo speaker con il punteggio più alto
identified_speaker = max(similarities, key=similarities.get)
max_score = similarities[identified_speaker]

if max_score > THRESHOLD:
    Idy = "Allowed"
else:
    Idy = "Not Allowed"    

# Restituisci il risultato come JSON
print(json.dumps({"Speaker": identified_speaker, "Score": max_score, "Identification": Idy}))