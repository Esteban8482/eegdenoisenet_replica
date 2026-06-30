"""
Cambios realizados para GPU:
1. Verificación explícita de disponibilidad GPU
2. Configuración de crecimiento de memoria GPU
3. API moderna de TensorFlow 2.x
4. Conversión de datos numpy a tensores GPU explícita
5. Uso de tf.data.Dataset para pipeline de datos
"""

import tensorflow as tf
import numpy as np
import os
import sys

# ======================================================
# 1. CONFIGURACION GPU
# ======================================================
# Esto debe ejecutarse antes de importar cualquier módulo de TF
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configurar memoria GPU 
# Evita que TF reserve toda la memoria GPU de una vez
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"GPUs detectadas: {gpus}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] Crecimiento de memoria GPU habilitado")
        print(f"[OK] GPU activa: {gpus[0]}")
    except RuntimeError as e:
        print(f"[ERROR] Al configurar GPU: {e}")
else:
    print("[ADVERTENCIA] No se detectaron GPUs. El entrenamiento usará CPU.")
    print("              Verifica la instalación de CUDA/cuDNN.")

# ======================================================
# 2. IMPORTAR MODULOS LOCALES
# ======================================================
sys.path.append('models')
sys.path.append('utils')

from network_structure import fcNN, simple_CNN, Complex_CNN, RNN_lstm
from data_prepare import prepare_data
from train_method_gpu import train
from save_method_gpu import save_eeg

# ======================================================
# 3. PARAMETROS CONFIGURABLES
# ======================================================
DATA_DIR = './data'      # Directorio del dataset
RESULT_DIR = './results' # Directorio de resultados

# Seleccion de modelo: 'fcNN' | 'Simple_CNN' | 'Complex_CNN' | 'RNN_lstm'
MODEL_NAME = 'Simple_CNN'

# Tipo de artefacto: 'EOG' | 'EMG'
NOISE_TYPE = 'EOG'

# Hiperparametros
EPOCHS = 40       # 60(fcNN), 40(CNNs), 100(RNN) para EOG
BATCH_SIZE = 40
COMBIN_NUM = 10   # Factor de expansion del dataset

# Optimizador - API moderna TF 2.x
# Adam: learning_rate=5e-5, beta_1=0.5, beta_2=0.9
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9)

# ======================================================
# 4. CONFIGURACION AUTOMATICA
# ======================================================
if NOISE_TYPE == 'EOG':
    DATANUM = 512
    EEG_FILE = 'EEG_all_epochs.npy'
    NOISE_FILE = 'EOG_all_epochs.npy'
elif NOISE_TYPE == 'EMG':
    DATANUM = 1024
    EEG_FILE = 'EEG_all_epochs.npy'
    NOISE_FILE = 'EMG_all_epochs.npy'

FOLDER_NAME = f'{NOISE_TYPE}_{MODEL_NAME}_run'

# ======================================================
# 5. CARGA DE DATOS (CPU - numpy)
# ======================================================
EEG_all = np.load(os.path.join(DATA_DIR, EEG_FILE))
noise_all = np.load(os.path.join(DATA_DIR, NOISE_FILE))
print(f'EEG: {EEG_all.shape}, Ruido: {noise_all.shape}')

# ======================================================
# 6. PREPARACION DE DATOS
# ======================================================
(noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
 noiseEEG_test, EEG_test, test_std) = prepare_data(
    EEG_all=EEG_all, noise_all=noise_all,
    combin_num=COMBIN_NUM, train_per=0.8, noise_type=NOISE_TYPE)

print(f'Train: {noiseEEG_train.shape}, Val: {noiseEEG_val.shape}, '
      f'Test: {noiseEEG_test.shape}')

# ======================================================
# 7. CONVERTIR DATOS A TENSORES GPU
# ======================================================
# Convertir numpy arrays a tensores de TensorFlow
print("\n[GPU] Transfiriendo datos a GPU...")

noiseEEG_train_tf = tf.convert_to_tensor(noiseEEG_train, dtype=tf.float32)
EEG_train_tf = tf.convert_to_tensor(EEG_train, dtype=tf.float32)
noiseEEG_val_tf = tf.convert_to_tensor(noiseEEG_val, dtype=tf.float32)
EEG_val_tf = tf.convert_to_tensor(EEG_val, dtype=tf.float32)
noiseEEG_test_tf = tf.convert_to_tensor(noiseEEG_test, dtype=tf.float32)
EEG_test_tf = tf.convert_to_tensor(EEG_test, dtype=tf.float32)

print(f"[OK] Datos transferidos a dispositivo: {noiseEEG_train_tf.device}")

# Calcular memoria GPU usada por datos
mem_train = noiseEEG_train_tf.nbytes + EEG_train_tf.nbytes
mem_val = noiseEEG_val_tf.nbytes + EEG_val_tf.nbytes
mem_test = noiseEEG_test_tf.nbytes + EEG_test_tf.nbytes
mem_total_mb = (mem_train + mem_val + mem_test) / (1024**2)
print(f"[INFO] Memoria GPU usada por datos: {mem_total_mb:.2f} MB")

# ======================================================
# 8. CREACION DEL MODELO
# ======================================================
if MODEL_NAME == 'fcNN':
    model = fcNN(DATANUM)
elif MODEL_NAME == 'Simple_CNN':
    model = simple_CNN(DATANUM)
elif MODEL_NAME == 'Complex_CNN':
    model = Complex_CNN(DATANUM)
elif MODEL_NAME == 'RNN_lstm':
    model = RNN_lstm(DATANUM)

# Construir modelo con un batch de ejemplo para inicializar pesos
print("\n[INFO] Construyendo modelo...")
if MODEL_NAME == 'fcNN':
    model.build(input_shape=(None, DATANUM))
else:
    model.build(input_shape=(None, DATANUM, 1))

# Mostrar información del modelo
model.summary()

# ======================================================
# 9. ENTRENAMIENTO
# ======================================================
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO EN GPU")
print(f"Modelo: {MODEL_NAME} | Artefacto: {NOISE_TYPE} | Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE} | Dataset expandido: {COMBIN_NUM}x")
print("="*60 + "\n")

saved_model, history = train(
    model=model,
    noiseEEG=noiseEEG_train_tf,
    EEG=EEG_train_tf,
    noiseEEG_val=noiseEEG_val_tf,
    EEG_val=EEG_val_tf,
    noiseEEG_test=noiseEEG_test_tf,
    EEG_test=EEG_test_tf,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    optimizer=optimizer,
    denoise_network=MODEL_NAME,
    result_location=RESULT_DIR,
    foldername=FOLDER_NAME,
    train_num='1'
)

# ======================================================
# 10. GUARDAR RESULTADOS
# ======================================================
# Guardar historial de perdida
np.save(f'{RESULT_DIR}/{FOLDER_NAME}/1/loss_history.npy', history)

# Guardar señales de test
save_eeg(saved_model, RESULT_DIR, FOLDER_NAME,
         False, False, True,
         noiseEEG_train_tf, EEG_train_tf,
         noiseEEG_val_tf, EEG_val_tf,
         noiseEEG_test_tf, EEG_test_tf,
         train_num='1', denoise_network=MODEL_NAME, datanum=DATANUM)

print('\n=== ENTRENAMIENTO COMPLETADO ===')
print(f'Resultados guardados en: {RESULT_DIR}/{FOLDER_NAME}/1/')
