from face_enrollment import enroll_user
from face_reco import recognize_face_live
import os
import glob
import argparse

def find_image_file(image_path, username):
    """Trova un file immagine per un utente specifico."""
    possible_files = glob.glob(os.path.join(image_path, f"{username}.*"))
    return possible_files[0] if possible_files else None

def main():
    # Configurazione degli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Sistema di riconoscimento facciale.")
    parser.add_argument('mode', choices=['1', '2'], help="Seleziona la modalità: 1 per registrazione, 2 per riconoscimento live.")
    parser.add_argument('--username', help="Nome utente da registrare (necessario in modalità 1).")
    parser.add_argument('--image_path', help="Percorso della cartella contenente l'immagine da registrare (necessario in modalità 1).")
    
    args = parser.parse_args()

    if args.mode == '1':
        if not args.username or not args.image_path:
            print("Errore: Per la modalità 1 è necessario specificare sia 'username' che 'image_path'.")
            return
        
        image_file = find_image_file(args.image_path, args.username)
        if image_file is None:
            print(f"Errore: immagine per '{args.username}' non trovata in '{args.image_path}'.")
            return
        else:
            print(f"Immagine trovata: {image_file}")

        # Passa il percorso completo del file immagine
        success = enroll_user(args.username, image_file)
        if success:
            print(f"Registrazione completata per {args.username}.")
        else:
            print(f"Errore nella registrazione per {args.username}.")

    elif args.mode == '2':
        print("Avvio del riconoscimento facciale live...")
        recognize_face_live()

if __name__ == "__main__":
    main()