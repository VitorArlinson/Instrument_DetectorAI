import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# --- CORREÇÃO APLICADA AQUI ---

# 1. Defina a lista de classes na mesma ordem que foi usada no treinamento.
#    Você pode verificar a ordem no seu notebook de treinamento com print(label_encoder.classes_)
class_names = ['guitar', 'organ', 'flute', 'string', 'bass', 'reed', 'vocal', 'synth_lead', 'brass']

# 2. Crie o LabelEncoder e "treine-o" com a lista de classes.
label_encoder = LabelEncoder()
label_encoder.fit(class_names) # <--- ESTA É A LINHA CRUCIAL QUE FALTAVA

# --- O RESTO DO SEU CÓDIGO CONTINUA IGUAL ---

# --- CONSTANTES DE PRÉ-PROCESSAMENTO (Devem ser as mesmas do treinamento) ---
MAX_LEN = 175
SAMPLE_RATE = 22050
N_MELS = 128

def processar_audio_para_predicao(file_path):
    """
    Carrega e processa um único arquivo de áudio, transformando-o em um 
    espectrograma padronizado, pronto para o modelo.
    """
    try:
        # (O conteúdo desta função continua o mesmo)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        if log_spectrogram.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - log_spectrogram.shape[1]
            padded_spec = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_spec = log_spectrogram[:, :MAX_LEN]
        return padded_spec[..., np.newaxis]
    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")
        return None

def prever_instrumento(audio_path, label_encoder_obj):
    """
    Função principal que recebe o caminho de um arquivo de áudio e
    imprime a predição dos três modelos treinados.
    """
    # (O conteúdo desta função continua o mesmo)
    processed_audio = processar_audio_para_predicao(audio_path)
    if processed_audio is None:
        return
    print(f"--- ANALISANDO O ARQUIVO: {os.path.basename(audio_path)} ---")
    processed_audio_rgb = np.repeat(processed_audio, 3, -1)
    input_baseline = np.expand_dims(processed_audio, axis=0)
    input_rgb = np.expand_dims(processed_audio_rgb, axis=0)
    
    # Modelo 1: Baseline
    model_b = load_model('models/baseline_best.h5')
    pred_b = model_b.predict(input_baseline)
    class_b = label_encoder_obj.classes_[np.argmax(pred_b[0])]
    conf_b = np.max(pred_b[0])
    
    # Modelo 2: ResNet50
    model_i = load_model('models/intermediate_best.h5')
    pred_i = model_i.predict(input_rgb)
    class_i = label_encoder_obj.classes_[np.argmax(pred_i[0])]
    conf_i = np.max(pred_i[0])
    
    # Imprime os resultados
    print("\n--- RESULTADOS DA PREDIÇÃO ---")
    print(f"Modelo Baseline (CNN Simples): {class_b.upper()} (Confiança: {conf_b:.2%})")
    print(f"Modelo Intermediário (ResNet50): {class_i.upper()} (Confiança: {conf_i:.2%})")

# --- COLOQUE O CAMINHO DO SEU ARQUIVO DE ÁUDIO AQUI ---
caminho_do_meu_audio = "data/nsynth-test/audio/vocal_synthetic_003-101-025.wav" 

# Chama a função para fazer a mágica
prever_instrumento(caminho_do_meu_audio, label_encoder)