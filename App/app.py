from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from typing import List
from pydantic import BaseModel
import os
import numpy as np
from app_utils import extract_voice_embedding, extract_face_embedding  # Funzioni da implementare
from pymongo import MongoClient
from bson.binary import Binary
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import io
from pydub import AudioSegment

# Connessione a MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Sostituire con il tuo URI di MongoDB
db = client["app"]  # Nome del database
user_collection = db["users"]  # Collezione per gli utenti
embedding_collection = db["embeddings"]  # Collezione per gli embedding

app = FastAPI()


# Setup templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
#app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Pydantic model for user registration
class UserRegister(BaseModel):
    username: str
    password: str


    
USER_DATABASE = {}

# Register Endpoint
@app.post("/auth/register")
def register(user: UserRegister):
    if user_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="User already exists.")

    # Salvataggio dell'utente
    user_collection.insert_one({
        "username": user.username,
        "password": user.password
    })

    print(f"User {user.username} registered.")

    return {"message": "Registration successful. Welcome, " + user.username + "!"}

# Login Page (Handled in Home)
@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...), response: Response = None):
    user = user_collection.find_one({"username": username})

    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    # Return a success message as JSON
    return {"message": "✅ Login effettuato con successo!\nBentornato " + username + "!"}

# Logout Endpoint
@app.post("/auth/logout")
def logout(response: Response):
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(key="username")
    return response
    
@app.post("/enroll/voice")
async def enroll_voice(
    request: Request,
    audio: UploadFile = File(...),  # Usa il nome del campo correttamente
    sample: int = Form(...)
):
    try:
        # Estrai username dalla richiesta
        username = request.cookies.get("username")
        print(f"Ricevuta registrazione numero: {sample}, dell'utente: {username}")
        print(f"Ricevuta registrazione numero: {sample}, dell'utente: {username}")
        if not username:
            raise HTTPException(status_code=404, detail="Username non trovato nei cookies.")
        
        # Verifica se l'utente esiste nel database
        if not user_collection.find_one({"username": username}):
            raise HTTPException(status_code=404, detail="User not found.")
        
        # Verifica il tipo del file audio
        if audio.content_type not in ["audio/webm"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are allowed.")

        # Leggi il contenuto del file audio
        # Read the uploaded file into a BytesIO buffer
        audio_bytes = await audio.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Load the WebM audio using pydub
        audio_segment = AudioSegment.from_file(audio_stream, format="webm")

        # Convert to WAV
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)  # Reset buffer position

        # Save to disk (Optional, for debugging)
        with open(f"{BASE_DIR}/converted_audio.wav", "wb") as f:
            f.write(wav_buffer.getbuffer())

        # Esegui l'estrazione dell'embedding (sostituire con la tua funzione di estrazione)
        embedding = extract_voice_embedding(f"{BASE_DIR}/converted_audio.wav")  # Funzione da implementare
        print(f"Embedding: {embedding}")

        # Salva l'embedding nel database
        embedding_collection.insert_one({
            "username": username, 
            "type": f"voice{sample}", 
            "embedding": Binary(np.array(embedding).tobytes())  # Memorizza come binario
        })

        return {"message": f"Voice recording successfully saved for {username}!"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/modify/voice")
async def modify_voice(
    request: Request,
    audio: UploadFile = File(...),  # Usa il nome del campo correttamente
    sample: int = Form(...)
):
    try:
        # Estrai username dalla richiesta
        username = request.cookies.get("username")
        print(f"Ricevuta registrazione numero: {sample}, dell'utente: {username}")

        if not username:
            raise HTTPException(status_code=404, detail="Username non trovato nei cookies.")
        
        # Verifica se l'utente esiste nel database
        if not user_collection.find_one({"username": username}):
            raise HTTPException(status_code=404, detail="User not found.")
        
        # Verifica il tipo del file audio
        if audio.content_type not in ["audio/webm"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are allowed.")

        # Leggi il contenuto del file audio
        # Read the uploaded file into a BytesIO buffer
        audio_bytes = await audio.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Load the WebM audio using pydub
        audio_segment = AudioSegment.from_file(audio_stream, format="webm")

        # Convert to WAV
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)  # Reset buffer position

        # Save to disk (Optional, for debugging)
        with open(f"{BASE_DIR}/converted_audio.wav", "wb") as f:
            f.write(wav_buffer.getbuffer())

        # Esegui l'estrazione dell'embedding (sostituire con la tua funzione di estrazione)
        embedding = extract_voice_embedding(f"{BASE_DIR}/converted_audio.wav")  # Funzione da implementare
        print(f"Embedding: {embedding}")

        # Sovrascrive il vecchio embedding se esiste
        type=f"voice{sample}"
        embedding_collection.update_one(
            {"username": username, "type": type},
            {"$set": {"embedding": Binary(np.array(embedding).tobytes())}}, 
            upsert=True
        )

        return {"message": f"Voice recording successfully saved for {username}!"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/delete/voice")
async def delete_face_embedding(request: Request, sample: int = Form(...)):
    username = request.cookies.get("username")
    
    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    result = embedding_collection.delete_one({"username": username, "type": f"voice{sample}"})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Face embedding not found.")

    return {"message": f"Il sample vocale di {username} è stato eliminato con successo."}


# Face Enrollment Endpoint
@app.post("/enroll/face")
async def enroll_face(request: Request, file: UploadFile = File(...)):
    username = request.cookies.get("username")
    
    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    if not user_collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="User not found.")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    file_content = await file.read()
    image_stream = BytesIO(file_content)

    # Estrai l'embedding
    embedding = extract_face_embedding(image_stream)
    if embedding is False:
        raise HTTPException(status_code=400, detail="Error extracting face embedding.")
    
    # Sovrascrive il vecchio embedding se esiste
    embedding_collection.insert_one(
        {"username": username, "type": "face", "embedding": Binary(np.array(embedding).tobytes())}
    )

    return {"message": f"Congratulazioni {username}! Il tuo volto è stato creato con successo."}

@app.get("/check/face")
def check_face_status(request: Request):
    username = request.cookies.get("username")

    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    has_embedding = embedding_collection.find_one({"username": username, "type": "face"}) is not None

    return {"has_embedding": has_embedding}

@app.get("/check/voice")
def check_voice_status(request: Request):
    username = request.cookies.get("username")

    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    has_voice_embedding_1 = embedding_collection.find_one({"username": username, "type": "voice1"}) is not None
    has_voice_embedding_2 = embedding_collection.find_one({"username": username, "type": "voice2"}) is not None
    has_voice_embedding_3 = embedding_collection.find_one({"username": username, "type": "voice3"}) is not None

    return {
        "has_voice_embedding_1": has_voice_embedding_1,
        "has_voice_embedding_2": has_voice_embedding_2,
        "has_voice_embedding_3": has_voice_embedding_3
    }


@app.post("/modify/face")
async def modify_face(request: Request, file: UploadFile = File(...)):
    username = request.cookies.get("username")
    
    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    if not user_collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="User not found.")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    file_content = await file.read()
    image_stream = BytesIO(file_content)

    # Estrai l'embedding
    embedding = extract_face_embedding(image_stream)
    if embedding is False:
        raise HTTPException(status_code=400, detail="Error extracting face embedding.")
    
    # Sovrascrive il vecchio embedding se esiste
    embedding_collection.update_one(
        {"username": username, "type": "face"},
        {"$set": {"embedding": Binary(np.array(embedding).tobytes())}}, 
        upsert=True
    )
    
    return {"message": f"Congratulazioni {username}! Il tuo volto è stato aggiornato con successo."}


@app.delete("/delete/face")
async def delete_face_embedding(request: Request):
    username = request.cookies.get("username")
    
    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    result = embedding_collection.delete_one({"username": username, "type": "face"})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Face embedding not found.")

    return {"message": f"Il volto di {username} è stato eliminato con successo."}

# first page
@app.get("/", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})

# Home Page
@app.get("/home", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    return templates.TemplateResponse("home.html", {"request": request})

#Register Page
@app.get("/register", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    return templates.TemplateResponse("register.html", {"request": request})

#Login Page
@app.get("/login", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})

#Enrollment Page
@app.get("/enrollment", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("enrollment.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})

#Face Enrollment Page
@app.get("/enroll/face", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("face_enrollment.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})

#Voice Enrollment Page
@app.get("/enroll/voice", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("voice_enrollment.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)