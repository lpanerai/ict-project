from PIL import Image
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import cv2

def extract_voice_embedding():
    return True

def extract_face_embedding(image_file):
    try:
        # Inizializza il modello FaceNet
        embedder = FaceNet()
        # Inizializza il rilevatore di volti MTCNN
        detector = MTCNN()

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
