from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Depends, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from pydantic import BaseModel
import os
import numpy as np
import tempfile
from app_utils import extract_voice_embedding, extract_face_embedding  # Funzioni da implementare
from pymongo import MongoClient
from bson.binary import Binary
from io import BytesIO

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
    return {"message": "Login effettuato con successo!\nBentornato " + username + "!"}

# Logout Endpoint
@app.post("/auth/logout")
def logout(response: Response):
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(key="username")
    return response

# Voice Enrollment Endpoint
@app.post("/enroll/voice")
def enroll_voice(username: str = Form(...), file: UploadFile = File(...)):
    if user_collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="User not found.")

    if file.content_type not in ["audio/wav", "audio/mpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Save file temporarily and extract embedding
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        temp.write(file.file.read())
        embedding = extract_voice_embedding(temp.name)

    # Save voice embedding
    USER_DATABASE[username]["voice_embeddings"].append(np.array(embedding))
    return {"message": "Voice embedding enrolled."}

# Face Enrollment Endpoint
@app.post("/enroll/face")
def enroll_face(request: Request, file: UploadFile = File(...)):
    username = request.cookies.get("username")
    print(f"Username: {username}")

    if not username:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    if not user_collection.find_one({"username": username}):
        raise HTTPException(status_code=404, detail="User not found.")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    file_content = file.file.read()
    image_stream = BytesIO(file_content)

    # Estrai l'embedding
    embedding = extract_face_embedding(image_stream)
    
    # Save face embedding
    embedding_collection.insert_one({
        "username": username,
        "type": "face",
        "embedding": Binary(np.array(embedding).tobytes())  # Salva come binario
    })
    return {"message": "Face embedding enrolled."}

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

@app.get("/register", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("home.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/enrollment", response_class=HTMLResponse)
def home(request: Request, username: str = Cookie(default=None)):
    if username:
        return templates.TemplateResponse("enrollment.html", {"request": request, "username": username})
    return templates.TemplateResponse("login.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)