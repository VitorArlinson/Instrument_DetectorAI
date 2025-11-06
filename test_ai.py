import sounddevice as sd
import numpy as np
import librosa
import os
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'models\intermediate_best.h5'
IS_RGB = True
SAMPLE_RATE = 22050
CHUNK_SECONDS = 4
PREDICTION_INTERVAL = 1
VOLUME_THRESHOLD = 0.01

# --- CONSTANTES DE PRÉ-PROCESSAMENTO ---
MAX_LEN = 175
N_MELS = 128

# --- Carregar o Modelo e o Label Encoder ---
print("Carregando o modelo...")
model = load_model(MODEL_PATH)
class_names = ['guitar', 'organ', 'flute', 'string', 'bass', 'reed', 'vocal', 'synth_lead', 'brass']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)
print("Modelo carregado com sucesso.")

def processar_audio(audio_chunk):
    """Processa um pedaço de áudio para o formato de entrada do modelo."""
    try:
        spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=SAMPLE_RATE, n_mels=N_MELS)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        if log_spectrogram.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - log_spectrogram.shape[1]
            padded_spec = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_spec = log_spectrogram[:, :MAX_LEN]
            
        return padded_spec
    except Exception as e:
        print(f"Erro no processamento: {e}")
        return None

# --- LOOP PRINCIPAL ---
print("\nIniciando a escuta em tempo real... Pressione Ctrl+C para parar.")
try:
    while True:
        # Grava um chunk de áudio do microfone
        chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        print(f"Gravando os próximos {CHUNK_SECONDS} segundos de áudio...") # <--- NOVO
        audio_chunk = sd.rec(samplerate=SAMPLE_RATE, channels=1, frames=chunk_samples, dtype='float32')
        sd.wait()
        
        audio_chunk = audio_chunk.flatten()
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        volume = np.abs(audio_chunk).mean()
        if volume < VOLUME_THRESHOLD:
            print("ouvindo... (Silêncio detectado)")
            time.sleep(PREDICTION_INTERVAL)
            continue
            
        print(f"ouvindo... (Volume: {volume:.4f}) - Processando...")
        
        processed_spec = processar_audio(audio_chunk)
        
        if processed_spec is not None:
            input_data = processed_spec[..., np.newaxis]
            if IS_RGB:
                input_data = np.repeat(input_data, 3, -1)
            input_data = np.expand_dims(input_data, axis=0)
            
            prediction = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction[0])
            predicted_class = label_encoder.classes_[predicted_index]
            confidence = np.max(prediction[0])
            
            print("\n=============================")
            print(f"   INSTRUMENTO: {predicted_class.upper()}")
            print(f"   CONFIANÇA: {confidence:.2%}")
            print("=============================")
            
            # --- REPRODUZIR O ÁUDIO CAPTURADO ---
            print("\nReproduzindo o áudio capturado...")
            sd.play(audio_chunk, samplerate=SAMPLE_RATE)
            sd.wait() 
            print("Reprodução concluída.") 
            
        time.sleep(PREDICTION_INTERVAL)

except KeyboardInterrupt:
    print("\nPrograma interrompido pelo usuário.")