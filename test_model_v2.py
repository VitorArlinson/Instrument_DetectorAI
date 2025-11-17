import os
import sys
import argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16
import joblib

# Suprime logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- CONSTANTES DE PRÉ-PROCESSAMENTO ---
MAX_LEN = 175 
SAMPLE_RATE = 22050
N_MELS = 128
NUM_CLASSES = 9 # ('guitar', 'organ', 'flute', 'string', 'bass', 'reed', 'vocal', 'synth_lead', 'brass')

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def processar_audio_para_predicao(file_path):
    """
    Função para carregar um file de áudio, extrair o espectrograma em Mel 
    e padronizar seu tamanho para MAX_LEN (lógica do notebook).
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        padded_spec = None
        if log_spectrogram.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - log_spectrogram.shape[1]
            padded_spec = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_spec = log_spectrogram[:, :MAX_LEN]
            
        return padded_spec[..., np.newaxis]
            
    except Exception as e:
        print(f"Erro ao processar o áudio {file_path}: {e}")
        return None

# --- FUNÇÕES DE CRIAÇÃO DE MODELO ---
# para recriar a arquitetura antes de carregar os pesos

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


# --- FUNÇÃO DE PREDIÇÃO ---
def prever_instrumento(audio_path, label_encoder_obj, 
                       model_baseline_path, model_resnet_path,
                       model_vgg16_path, model_efficientnet_path):
    """
    Função principal que recria as arquiteturas e usa load_weights()
    para carregar e prever com os 4 modelos treinados.
    """
    processed_audio = processar_audio_para_predicao(audio_path)
    
    if processed_audio is None:
        return

    print(f"\n--- ANALISANDO O ARQUIVO: {os.path.basename(audio_path)} ---")
    
    # Input para Baseline (1 canal)
    input_baseline = np.expand_dims(processed_audio, axis=0)
    
    # Input para modelos de Transferência (3 canais)
    processed_audio_rgb = np.repeat(processed_audio, 3, -1)
    input_rgb = np.expand_dims(processed_audio_rgb, axis=0)

    # Pegar shapes das entradas
    input_shape_baseline = input_baseline.shape[1:]
    input_shape_rgb = input_rgb.shape[1:]

    resultados = {}

    # --- Modelo 1: Baseline ---
    tf.keras.backend.clear_session()
    try:
        print(f"Carregando Baseline: {model_baseline_path}")
        model_b = criar_modelo_baseline(input_shape_baseline, NUM_CLASSES) # 1. Recria
        model_b.load_weights(model_baseline_path)                        # 2. Carrega pesos
        pred_b = model_b.predict(input_baseline, verbose=0)
        class_b = label_encoder_obj.classes_[np.argmax(pred_b[0])]
        conf_b = np.max(pred_b[0])
        resultados["Baseline"] = f"{class_b.upper()} (Confiança: {conf_b:.2%})"
    except Exception as e:
        print(f"Erro ao carregar ou prever com o modelo Baseline: {e}")
        resultados["Baseline"] = "ERRO"

    # --- Modelo 2: ResNet50 ---
    tf.keras.backend.clear_session()
    try:
        print(f"Carregando ResNet50: {model_resnet_path}")
        model_i = criar_modelo_resnet50(input_shape_rgb, NUM_CLASSES) # 1. Recria
        model_i.load_weights(model_resnet_path)                      # 2. Carrega pesos
        pred_i = model_i.predict(input_rgb, verbose=0)
        class_i = label_encoder_obj.classes_[np.argmax(pred_i[0])]
        conf_i = np.max(pred_i[0])
        resultados["ResNet50"] = f"{class_i.upper()} (Confiança: {conf_i:.2%})"
    except Exception as e:
        print(f"Erro ao carregar ou prever com o modelo ResNet50: {e}")
        resultados["ResNet50"] = "ERRO"

    # --- Modelo 3: VGG16 ---
    tf.keras.backend.clear_session()
    try:
        print(f"Carregando VGG16: {model_vgg16_path}")
        model_v = criar_modelo_vgg16(input_shape_rgb, NUM_CLASSES) # 1. Recria
        model_v.load_weights(model_vgg16_path)                     # 2. Carrega pesos
        pred_v = model_v.predict(input_rgb, verbose=0)
        class_v = label_encoder_obj.classes_[np.argmax(pred_v[0])]
        conf_v = np.max(pred_v[0])
        resultados["VGG16"] = f"{class_v.upper()} (Confiança: {conf_v:.2%})"
    except Exception as e:
        print(f"Erro ao carregar ou prever com o modelo VGG16: {e}")
        resultados["VGG16"] = "ERRO"
        
    # --- Modelo 4: EfficientNet ---
    tf.keras.backend.clear_session()
    try:
        print(f"Carregando EfficientNet: {model_efficientnet_path}")
        model_e = criar_modelo_efficientnet(input_shape_rgb, NUM_CLASSES) # 1. Recria
        model_e.load_weights(model_efficientnet_path)                     # 2. Carrega pesos
        pred_e = model_e.predict(input_rgb, verbose=0)
        class_e = label_encoder_obj.classes_[np.argmax(pred_e[0])]
        conf_e = np.max(pred_e[0])
        resultados["EfficientNet"] = f"{class_e.upper()} (Confiança: {conf_e:.2%})"
    except Exception as e:
        print(f"Erro ao carregar ou prever com o modelo EfficientNet: {e}")
        resultados["EfficientNet"] = "ERRO"

    # --- Imprime os resultados ---
    print("\n--- RESULTADOS DA PREDIÇÃO ---")
    print(f"Modelo Baseline (CNN Simples):   {resultados.get('Baseline', 'N/A')}")
    print(f"Modelo Intermediário (ResNet50): {resultados.get('ResNet50', 'N/A')}")
    print(f"Modelo Avançado (VGG16):       {resultados.get('VGG16', 'N/A')}")
    print(f"Modelo Avançado (EfficientNet):  {resultados.get('EfficientNet', 'N/A')}")

def main():
    # --- DEFINAÇÃO DOS CAMINHOS ---
    CAMINHO_LABEL_ENCODER = "label_encoder.joblib"
    
    CAMINHO_MODELO_BASELINE = "meus_experimentos_cnn\\run_modelo_baseline_amostras_1500_batch_32_epochs_60\\modelo_melhor.h5"
    CAMINHO_MODELO_RESNET = "meus_experimentos_cnn\\run_modelo_resnet50_amostras_1000_batch_16_epochs_60\\modelo_melhor.h5"
    CAMINHO_MODELO_VGG16 = "meus_experimentos_cnn\\run_modelo_vgg16_amostras_1000_batch_32_epochs_60\\modelo_melhor.h5"
    CAMINHO_MODELO_EFFICIENTNET = "meus_experimentos_cnn\\run_modelo_efficientnet_amostras_1000_batch_32_epochs_60\\modelo_melhor.h5"

    # Configura o "parser" para ler argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Testador de Modelos de Instrumentos Musicais")
    parser.add_argument("arquivo_audio", help="Caminho para o arquivo de áudio (.wav) que você quer testar.")
    
    args = parser.parse_args()
    caminho_do_audio = args.arquivo_audio
    
    # --- VERIFICAÇÕES DE ARQUIVOS ---
    if not os.path.exists(caminho_do_audio):
        print(f"Erro: O file de áudio não foi encontrado em: {caminho_do_audio}")
        sys.exit(1)
        
    if not os.path.exists(CAMINHO_LABEL_ENCODER):
        print(f"Erro: O Label Encoder não foi encontrado em: {CAMINHO_LABEL_ENCODER}")
        print("Por favor, rode o script para salvar o 'label_encoder.joblib' no seu notebook.")
        sys.exit(1)

    modelos_paths = {
        "Baseline": CAMINHO_MODELO_BASELINE,
        "ResNet50": CAMINHO_MODELO_RESNET,
        "VGG16": CAMINHO_MODELO_VGG16,
        "EfficientNet": CAMINHO_MODELO_EFFICIENTNET
    }
    
    erro_path = False
    for nome, path in modelos_paths.items():
        if not os.path.exists(path):
            print(f"Erro: O modelo '{nome}' não foi encontrado no caminho:")
            print(f"  {path}")
            erro_path = True
            
    if erro_path:
        print("\nPor favor, verifique os caminhos dos modelos na seção 'main' do script prever.py")
        sys.exit(1)

    # --- EXECUÇÃO ---
    print("Carregando Label Encoder...")
    label_encoder = joblib.load(CAMINHO_LABEL_ENCODER)
    
    prever_instrumento(caminho_do_audio, 
                       label_encoder, 
                       CAMINHO_MODELO_BASELINE, 
                       CAMINHO_MODELO_RESNET,
                       CAMINHO_MODELO_VGG16,
                       CAMINHO_MODELO_EFFICIENTNET)

# Ponto de entrada do script
if __name__ == "__main__":
    main()