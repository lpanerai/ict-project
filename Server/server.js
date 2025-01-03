const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // Per eseguire Python dal server

const app = express();
const port = 3000;

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
  res.send('Server attivo! Usa POST /upload per caricare file audio e confrontarli.');
});

// Rotta per caricare file e fare Speaker Identification
app.post('/upload', upload.single('file'), (req, res) => {
  const uploadedFilePath = req.file.path;
  const referenceFilePath = path.join(__dirname, 'reference', 'reference.wav');

  console.log(`File ricevuto: ${uploadedFilePath}`);
  console.log(`Confronto con: ${referenceFilePath}`);

  // Esegui uno script Python per lo Speaker Identification
  const pythonProcess = spawn('python', ['speaker_identification.py', referenceFilePath, uploadedFilePath]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Risultato: ${data}`);
    res.status(200).send(`Risultato Speaker Identification: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Errore: ${data}`);
    res.status(500).send('Errore durante l\'elaborazione.');
  });

  pythonProcess.on('close', (code) => {
    console.log(`Processo Python terminato con codice ${code}`);
  });
});

// Avvia il server
app.listen(port, () => {
  console.log(`Server in ascolto su http://localhost:${port}`);
});
