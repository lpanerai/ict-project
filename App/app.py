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

app = FastAPI()

# Setup templates and static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
#app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Placeholder for database (to replace with real DB integration)
USER_DATABASE = {}
EMBEDDING_DATABASE = {
    "voice": {},
    "face": {}
}

# Pydantic model for user registration
class UserRegister(BaseModel):
    username: str
    password: str

# Register Endpoint
@app.post("/auth/register")
def register(user: UserRegister):
    if user.username in USER_DATABASE:
        raise HTTPException(status_code=400, detail="User already exists.")

    # Salvataggio dell'utente
    USER_DATABASE[user.username] = {
        "password": user.password,
        "voice_embeddings": [],
        "face_embeddings": []
    }
    print(f"User {user.username} registered.")
    print(f"User database: {USER_DATABASE}")

    return {"message": "Registration successful. Welcome, " + user.username + "!"}

# Login Page (Handled in Home)
@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...), response: Response = None):
    if username not in USER_DATABASE or USER_DATABASE[username]["password"] != password:
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
    if username not in USER_DATABASE:
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
def enroll_face(username: str = Form(...), file: UploadFile = File(...)):
    if username not in USER_DATABASE:
        raise HTTPException(status_code=404, detail="User not found.")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # Save file temporarily and extract embedding
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        temp.write(file.file.read())
        embedding = extract_face_embedding(temp.name)

    # Save face embedding
    USER_DATABASE[username]["face_embeddings"].append(np.array(embedding))
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