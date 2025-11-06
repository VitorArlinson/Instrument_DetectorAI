import tensorflow as tf

# Verifica se a GPU está disponível
if tf.config.list_physical_devices('GPU'):
    print("GPU encontrada. Realizando teste de cálculo...")
    try:
        # Força a execução em um dispositivo específico (GPU:0)
        with tf.device('/GPU:0'):
            # Cria duas matrizes grandes
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

            # Realiza a multiplicação de matrizes
            c = tf.matmul(a, b)

        print("Cálculo na GPU realizado com sucesso!")
        print("Resultado da multiplicação:\n", c.numpy())
    except RuntimeError as e:
        print("Erro durante o cálculo na GPU:", e)
else:
    print("Nenhuma GPU encontrada pelo TensorFlow.")