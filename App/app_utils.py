from PIL import Image
import numpy as np
from keras_facenet import FaceNet


def extract_voice_embedding():
    return True

def extract_face_embedding(image_file):
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
        embedding = np.array(face_encoding)
        return embedding
    except Exception as e:
        print(f"Errore durante la registrazione dell'utente: {e}")
        return False
