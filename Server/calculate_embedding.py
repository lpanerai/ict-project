import sys
import json
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

try:
    # Percorso del file e nome dello speaker
    uploaded_file = sys.argv[1]
    speaker_name = sys.argv[2]

    # Carica il modello
    verification_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_model"
    )

    # Leggi il file audio e converti in tensore
    try:
        waveform, sample_rate = torchaudio.load(uploaded_file)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)    
    except Exception as e:
        raise ValueError(f"Errore durante la lettura del file audio: {e}")

    
    waveform = remove_noise(waveform, sample_rate)
    waveform = remove_silence(waveform)

    # Calcola embedding
    try:
        embedding = verification_model.encode_batch(waveform).detach().cpu().numpy()
    except Exception as e:
        raise RuntimeError(f"Errore durante il calcolo dell'embedding: {e}")

    # Salva embedding (logica per aggiungere al JSON esistente)
    embedding_file = "speaker_embeddings.json"
    try:
        with open(embedding_file, "r") as file:
            embeddings = json.load(file)
    except FileNotFoundError:
        embeddings = {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Errore nel caricamento del file JSON: {e}")

    if speaker_name not in embeddings:
        embeddings[speaker_name] = []

    embeddings[speaker_name].append(embedding.tolist())

    try:
        with open(embedding_file, "w") as file:
            json.dump(embeddings, file, indent=4)
    except Exception as e:
        raise IOError(f"Errore durante il salvataggio del file JSON: {e}")

    # Restituisci risultato
    print(json.dumps({"Status": "success", "Speaker": speaker_name}))

except Exception as e:
    # Gestione degli errori generali
    print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
    sys.exit(1)
