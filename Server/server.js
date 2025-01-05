const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // Per eseguire Python dal server

const app = express();
const port = 3000;

// Percorso all'interprete Python
const pythonPath = "C:\\Users\\giorg\\Downloads\\ict_work\\work_ict\\Scripts\\python.exe"; 

// Configura multer per salvare i file caricati
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploads'); // Directory per salvare i file caricati
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage });

// Rotta principale
app.get('/', (req, res) => {
  res.send('Server attivo! Usa POST /enroll o POST /identify.');
});
 
// Endpoint per l'enrollment
app.post('/enroll', upload.single('file'), (req, res) => {
  const uploadedFilePath = req.file.path;
  const speaker = req.body.speaker;

  console.log(`File ricevuto per enrollment: ${uploadedFilePath} per lo speaker: ${speaker}`);

  // Esegui lo script Python per calcolare gli embedding
  const pythonProcess = spawn(pythonPath, ['calculate_embedding.py', uploadedFilePath, speaker]);

  let resultBuffer = ""; // Buffer per accumulare i dati stdout
  let errorBuffer = "";  // Buffer per accumulare i dati stderr

  pythonProcess.stdout.on('data', (data) => {
    resultBuffer += data.toString(); // Accumula i dati ricevuti
  });

  pythonProcess.stderr.on('data', (data) => {
    errorBuffer += data.toString(); // Accumula eventuali errori
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) { // Processo terminato correttamente
      try {
        const parsedData = JSON.parse(resultBuffer.trim()); // Parse del risultato JSON
        console.log(`Risultato enrollment: ${JSON.stringify(parsedData)}`);
        res.status(200).json(parsedData); // Risposta al client
      } catch (err) {
        console.error("Errore durante il parsing della risposta Python:", err);
        res.status(500).send("Errore durante l'elaborazione della risposta JSON.");
      }
    } else { // Processo terminato con errore
      console.error(`Errore durante l'enrollment: ${errorBuffer}`);
      res.status(500).send(`Errore durante l'enrollment: ${errorBuffer}`);
    }
  });
});

app.post('/identify', upload.single('file'), (req, res) => {
  const uploadedFilePath = req.file.path;

  console.log(`File ricevuto per identificazione: ${uploadedFilePath}`);

  // Esegui script Python per identificare lo speaker
  const pythonProcess = spawn(pythonPath, ['identify_speaker.py', uploadedFilePath]);

  let resultBuffer = ""; // Buffer per accumulare i dati stdout
  let errorBuffer = "";  // Buffer per accumulare i dati stderr

  pythonProcess.stdout.on('data', (data) => {
    resultBuffer += data.toString(); // Accumula i dati ricevuti
  });

  pythonProcess.stderr.on('data', (data) => {
    errorBuffer += data.toString(); // Accumula eventuali errori
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) { // Processo terminato correttamente
      try {
        const parsedData = JSON.parse(resultBuffer.trim()); // Parse del risultato JSON
        console.log(`Risultato identificazione: ${JSON.stringify(parsedData)}`);
        res.status(200).json(parsedData); // Risposta al client
      } catch (err) {
        console.error("Errore durante il parsing della risposta Python:", err);
        res.status(500).send("Errore durante l'elaborazione della risposta JSON.");
      }
    } else { // Processo terminato con errore
      console.error(`Errore durante l'identificazione: ${errorBuffer}`);
      res.status(500).send(`Errore durante l'identificazione: ${errorBuffer}`);
    }
  });
});

// Avvia il server
app.listen(port, () => {
  console.log(`Server in ascolto su http://localhost:${port}`);
});
