from PIL import Image
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import cv2
import torch
import soundfile as sf
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.transforms as T
import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello di riconoscimento speaker
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# Inizializza il modello FaceNet
embedder = FaceNet()
# Inizializza il rilevatore di volti MTCNN
detector = MTCNN()

def extract_voice_embedding(audio_bytes):
    """
    Extract voice embedding from audio bytes using SpeechBrain's ECAPA-TDNN model.
    
    Args:
        audio_bytes (bytes): Raw audio file bytes
        
    Returns:
        numpy.ndarray: Voice embedding vector
        
    Raises:
        Exception: If audio processing or embedding extraction fails
    """
    
    try:
        
        # Convert bytes directly to tensor using torchaudio
        waveform, sample_rate = torchaudio.load(audio_bytes)
            
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        
        # Extract embedding
        with torch.no_grad():
            embedding = classifier.encode_batch(waveform)
            
        # Convert to numpy and return first embedding
        return embedding
        
    except Exception as e:
        print(f"Error extracting voice embedding: {str(e)}")
        raise Exception(f"Voice embedding extraction failed: {str(e)}")
    

def extract_face_embedding(image_file):
    try:
        

        # Carica l'immagine con Pillow
        img = Image.open(image_file).convert("RGB")
        image = np.array(img)

        # Rileva volti
        faces = detector.detect_faces(image)
        if not faces:
            print("Errore: nessun volto rilevato nell'immagine.")
            return False
        
        # Ordina i volti in base alla confidence (valore più alto prima)
        faces = sorted(faces, key=lambda f: f['confidence'], reverse=True)

        # Prende solo il volto con confidence più alta
        best_face = faces[0]
        x, y, w, h = best_face['box']

        print(f"Volto selezionato con confidence: {best_face['confidence']}")

        # Ritaglia il volto con più confidence
        face = image[y:y+h, x:x+w]

        # Ridimensiona l'immagine per FaceNet (160x160)
        face = cv2.resize(face, (160, 160))  

        # Salva il volto ritagliato in locale
        face_image = Image.fromarray(face)
        face_image.save("volto_ritagliato_160x160.jpg")
        print("Volto ritagliato salvato come 'volto_ritagliato_160x160.jpg'")

        # Ottieni gli embedding con FaceNet
        face_embeddings = embedder.embeddings([face])

        if len(face_embeddings) == 0:
            print("Errore: impossibile calcolare embedding.")
            return False

        face_encoding = face_embeddings[0]

        return np.array(face_encoding)

    except Exception as e:
        print(f"Errore durante la registrazione dell'utente: {e}")
        return False
