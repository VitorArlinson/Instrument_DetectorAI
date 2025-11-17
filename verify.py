import h5py
import sys
import os

def verificar_arquivo_h5(filepath):
    """
    Verifica se um arquivo .h5 é um modelo Keras completo ou
    contém apenas pesos.
    """
    if not os.path.exists(filepath):
        print(f"VEREDITO: ERRO! \nO arquivo não foi encontrado em: '{filepath}'")
        return False
        
    try:
        with h5py.File(filepath, 'r') as f:
            
            # Keras 2 (TF 2.10) salva a arquitetura aqui
            if 'model_config' in f.attrs:
                print(f"VEREDITO: SUCESSO! (Keras 2)")
                print(f"O arquivo '{os.path.basename(filepath)}' é um MODELO COMPLETO.")
                return True
            
            # Keras 3 (TF 2.16+) salva a arquitetura aqui
            if 'config' in f.attrs:
                print(f"VEREDITO: SUCESSO! (Keras 3)")
                print(f"O arquivo '{os.path.basename(filepath)}' é um MODELO COMPLETO.")
                return True
        
            if 'model_weights' in f or 'layer_names' in f:
                 print(f"VEREDITO: FALHA!")
                 print(f"O arquivo '{os.path.basename(filepath)}' contém APENAS PESOS.")
                 print("(Ele não tem a arquitetura 'model_config' ou 'config' nos atributos).")
                 return False
            
            print(f"VEREDITO: DESCONHECIDO.")
            print(f"O arquivo '{os.path.basename(filepath)}' não parece ser um modelo Keras padrão.")
            return False
            
    except Exception as e:
        print(f"Erro ao ler o arquivo '{filepath}': {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERRO: Forneça o caminho do arquivo .h5 ou .keras para verificar.")
        print("Exemplo: python verificar.py meus_modelos/modelo.h5")
        sys.exit(1)
    
    caminho = sys.argv[1]
    verificar_arquivo_h5(caminho)