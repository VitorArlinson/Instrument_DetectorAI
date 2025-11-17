import os
import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
import joblib
import sounddevice as sd

# Suprime logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- CONSTANTES DE PRÉ-PROCESSAMENTO ---
MAX_LEN = 175 
SAMPLE_RATE = 22050
N_MELS = 128

NUM_CLASSES = 9 
DURACAO_GRAVACAO_SEC = 4 # Duração da gravação (aprox. MAX_LEN=175)

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def processar_audio_para_predicao(sinal, sr=SAMPLE_RATE, n_mels=N_MELS, max_len=MAX_LEN):
    """
    Converte um sinal de áudio (array numpy) em um espectrograma
    pronto para o modelo (lógica do notebook).
    """
    try:
        # 1. Extrai o espectrograma em Mel
        spectrogram = librosa.feature.melspectrogram(y=sinal, sr=sr, n_mels=n_mels)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # 2. Padroniza o tamanho (padding ou truncating)
        padded_spec = None
        if log_spectrogram.shape[1] < max_len:
            pad_width = max_len - log_spectrogram.shape[1]
            padded_spec = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_spec = log_spectrogram[:, :max_len]
            
        return padded_spec[..., np.newaxis] 
            
    except Exception as e:
        print(f"Erro ao processar o áudio: {e}")
        return None

# --- FUNÇÕES DE CRIAÇÃO DE MODELO ---
#  Para recriar a arquitetura antes de carregar os pesos

def criar_modelo_baseline(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def criar_modelo_resnet50(input_shape_rgb, num_classes):
    base_model_resnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape_rgb)
    base_model_resnet.trainable = False
    model = Sequential([
        Input(shape=input_shape_rgb),
        base_model_resnet,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def criar_modelo_efficientnet(input_shape_rgb, num_classes):
    base_model_effnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape_rgb)
    base_model_effnet.trainable = False
    model = Sequential([
        Input(shape=input_shape_rgb),
        base_model_effnet,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def criar_modelo_vgg16(input_shape_rgb, num_classes):
    base_model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=input_shape_rgb)
    base_model_vgg.trainable = False
    model = Sequential([
        Input(shape=input_shape_rgb),
        base_model_vgg,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # --- DEFINIÇÃO DOS CAMINHOS ---
    CAMINHO_LABEL_ENCODER = "label_encoder.joblib"
    
    CAMINHO_MODELO_BASELINE = "meus_experimentos_cnn\\run_modelo_baseline_amostras_1500_batch_32_epochs_60\\modelo_melhor.h5"
    CAMINHO_MODELO_RESNET = "meus_experimentos_cnn\\run_modelo_resnet50_amostras_1000_batch_16_epochs_60\\modelo_melhor.h5"
    CAMINHO_MODELO_EFFICIENTNET = "meus_experimentos_cnn\\run_modelo_efficientnet_amostras_1000_batch_32_epochs_60\\modelo_melhor.h5"
    CAMINHO_MODELO_VGG16 = "meus_experimentos_cnn\\run_modelo_vgg16_amostras_1000_batch_32_epochs_60\\modelo_melhor.h5"

    # Dicionário de caminhos para os modelos
    modelos_paths = {
        "Baseline": CAMINHO_MODELO_BASELINE,
        "ResNet50": CAMINHO_MODELO_RESNET,
        "EfficientNet": CAMINHO_MODELO_EFFICIENTNET,
        "VGG16": CAMINHO_MODELO_VGG16
    }
    
    # --- VERIFICAÇÕES DE ARQUIVOS ---
    if not os.path.exists(CAMINHO_LABEL_ENCODER):
        print(f"Erro: O Label Encoder não foi encontrado em: {CAMINHO_LABEL_ENCODER}")
        sys.exit(1)

    for nome, path in modelos_paths.items():
        if not os.path.exists(path):
            print(f"Aviso: O modelo '{nome}' não foi encontrado em: {path}")
            print("Ele será pulado na predição.")
            modelos_paths[nome] = None 
            
    # --- CARREGAR LABEL ENCODER ---
    print("Carregando Label Encoder...")
    label_encoder = joblib.load(CAMINHO_LABEL_ENCODER)
    
    print("\n--- Modelos Prontos ---")
    print(f"Taxa de amostragem: {SAMPLE_RATE} Hz")
    print(f"Duração da gravação: {DURACAO_GRAVACAO_SEC} segundos")
    print("Pressione Ctrl+C para sair.")
    
    # --- LOOP DE GRAVAÇÃO E PREDIÇÃO ---
    try:
        while True:
            print("\n-------------------------------------------")
            print(f"Aguardando... Pressione Enter para começar a gravar {DURACAO_GRAVACAO_SEC}s de áudio...")
            input()
            
            print("Gravando...")
            
            # 1. Grava o áudio
            duracao_samples = int(DURACAO_GRAVACAO_SEC * SAMPLE_RATE)
            audio_gravado = sd.rec(duracao_samples, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait() 
            audio_gravado = audio_gravado.T[0] 
            
            print("Processando áudio...")
            
            # 2. Processa o áudio (mesmo pré-processamento do treino)
            espectrograma = processar_audio_para_predicao(audio_gravado)
            
            if espectrograma is None:
                print("Não foi possível processar o áudio.")
                continue
            
            # 3. Prepara os inputs (1 canal e 3 canais)
            input_baseline = np.expand_dims(espectrograma, axis=0)
            input_rgb = np.expand_dims(np.repeat(espectrograma, 3, -1), axis=0)
            
            input_shape_baseline = input_baseline.shape[1:]
            input_shape_rgb = input_rgb.shape[1:]

            print("\n--- RESULTADOS DA PREDIÇÃO ---")

            # --- 4. Loop pelos modelos para prever ---
            
            # Modelo 1: Baseline
            if modelos_paths["Baseline"]:
                try:
                    tf.keras.backend.clear_session()
                    model_b = criar_modelo_baseline(input_shape_baseline, NUM_CLASSES)
                    model_b.load_weights(modelos_paths["Baseline"])
                    pred_b = model_b.predict(input_baseline, verbose=0)
                    classe = label_encoder.classes_[np.argmax(pred_b[0])]
                    conf = np.max(pred_b[0])
                    print(f"Baseline: \t{classe.upper()} (Confiança: {conf:.2%})")
                except Exception as e:
                    print(f"Erro no Baseline: {e}")

            # Modelo 2: ResNet50
            if modelos_paths["ResNet50"]:
                try:
                    tf.keras.backend.clear_session()
                    model_r = criar_modelo_resnet50(input_shape_rgb, NUM_CLASSES)
                    model_r.load_weights(modelos_paths["ResNet50"])
                    pred_r = model_r.predict(input_rgb, verbose=0)
                    classe = label_encoder.classes_[np.argmax(pred_r[0])]
                    conf = np.max(pred_r[0])
                    print(f"ResNet50: \t{classe.upper()} (Confiança: {conf:.2%})")
                except Exception as e:
                    print(f"Erro no ResNet50: {e}")

            # Modelo 3: EfficientNet
            if modelos_paths["EfficientNet"]:
                try:
                    tf.keras.backend.clear_session()
                    model_e = criar_modelo_efficientnet(input_shape_rgb, NUM_CLASSES)
                    model_e.load_weights(modelos_paths["EfficientNet"])
                    pred_e = model_e.predict(input_rgb, verbose=0)
                    classe = label_encoder.classes_[np.argmax(pred_e[0])]
                    conf = np.max(pred_e[0])
                    print(f"EfficientNet: \t{classe.upper()} (Confiança: {conf:.2%})")
                except Exception as e:
                    print(f"Erro no EfficientNet: {e}")

            # Modelo 4: VGG16
            if modelos_paths["VGG16"]:
                try:
                    tf.keras.backend.clear_session()
                    model_v = criar_modelo_vgg16(input_shape_rgb, NUM_CLASSES)
                    model_v.load_weights(modelos_paths["VGG16"])
                    pred_v = model_v.predict(input_rgb, verbose=0)
                    classe = label_encoder.classes_[np.argmax(pred_v[0])]
                    conf = np.max(pred_v[0])
                    print(f"VGG16: \t\t{classe.upper()} (Confiança: {conf:.2%})")
                except Exception as e:
                    print(f"Erro no VGG16: {e}")

    except KeyboardInterrupt:
        print("\nEncerrando previsor...")
        sys.exit(0)

# Ponto de entrada do script
if __name__ == "__main__":
    main()