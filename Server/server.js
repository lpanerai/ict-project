const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process'); // Per eseguire Python dal server

const app = express();
const port = 3000;

//----Togliere in versione finale ------
// Percorso all'interprete Python nella cartella venv
const pythonPath = "C:\\Users\\giorg\\Downloads\\ict_work\\work_ict\\Scripts\\python.exe"; // Windows
//----Togliere in versione finale ------

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
  const referenceFilesDirectory = path.join(__dirname, 'uploads', 'references'); // Directory dei file di riferimento

  console.log(`File ricevuto: ${uploadedFilePath}`);

  // Recupera tutti i file di riferimento
  const referenceFiles = fs.readdirSync(referenceFilesDirectory).filter(file => file.endsWith('.wav'));
  const referenceFilePaths = referenceFiles.map(file => path.join(referenceFilesDirectory, file));

  // Esegui uno script Python per lo Speaker Identification
  const pythonProcess = spawn(pythonPath, ['speaker_identification.py', uploadedFilePath, ...referenceFilePaths]);

  let results = [];
  let resultBuffer = '';

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Risultato: ${data}`);
    resultBuffer += data.toString(); // Accumula tutti i dati
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Errore: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code === 0 && resultBuffer) {
      try {
        const result = JSON.parse(resultBuffer.trim());
        console.log(`Media degli score: ${result.average_score}`);
        if (result.status === "confirmed") {
          res.status(200).send(`Identity confermata`);
        }  else {
          res.status(200).send(`Identity non riconosciuta. Media score: ${result.average_score}`);
        }
      } catch (error) {
        console.error("Errore durante la decodifica del JSON:", error);
        res.status(500).send('Errore durante l\'elaborazione del risultato.');
      }
    } else {
      res.status(500).send('Errore durante l\'elaborazione.');
    }
  });
}); 

// Avvia il server
app.listen(port, () => {
  console.log(`Server in ascolto su http://localhost:${port}`);

});
