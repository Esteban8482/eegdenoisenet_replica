import tensorflow as tf
import numpy as np
import os
import sys

# ======================================================
# 1. CONFIGURACION GPU (preservada de version GPU)
# ======================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    print("[ADVERTENCIA] No se detectaron GPUs. Usando CPU.")

# ======================================================
# 2. IMPORTAR MODULOS LOCALES
# ======================================================
sys.path.append('models')
sys.path.append('utils')

from network_structure import fcNN, simple_CNN, Complex_CNN, RNN_lstm
from data_prepare import prepare_data
from train_method import train
from save_method import save_eeg

# ======================================================
# 3. PARAMETROS CONFIGURABLES
# ======================================================
# NOTA: Estos valores deben coincidir EXACTAMENTE con el paper:
# - FCNN: 60 epochs (EOG), 60 epochs (EMG)
# - Simple/Complex CNN: 40 epochs (EOG), 10 epochs (EMG)
# - RNN: 100 epochs (EOG), 60 epochs (EMG)
# - Batch size: 40 (todos los modelos)
# - Optimizador: Adam(lr=0.00005, beta_1=0.5, beta_2=0.9)

DATA_DIR = './data'
RESULT_DIR = './results'

# --- CONFIGURACION DEL MODELO ---
MODEL_NAME = 'fcNN'       # 'fcNN' | 'Simple_CNN' | 'Complex_CNN' | 'RNN_lstm'
NOISE_TYPE = 'EOG'        # 'EOG' | 'EMG'

# --- HIPERPARAMETROS (valores del paper) ---
if MODEL_NAME == 'fcNN':
    EPOCHS = 60 if NOISE_TYPE == 'EOG' else 60
elif MODEL_NAME in ['Simple_CNN', 'Complex_CNN']:
    EPOCHS = 40 if NOISE_TYPE == 'EOG' else 10
elif MODEL_NAME == 'RNN_lstm':
    EPOCHS = 100 if NOISE_TYPE == 'EOG' else 60
else:
    raise ValueError(f"Modelo desconocido: {MODEL_NAME}")

BATCH_SIZE = 40
COMBIN_NUM = 10  # Factor de expansion del dataset (10 niveles SNR)

# --- OPTIMIZADOR---
# tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
# API moderna:
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.00005,
    beta_1=0.5,
    beta_2=0.9,
    epsilon=1e-08
)

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

print(f"\n{'='*60}")
print(f"CONFIGURACION DEL EXPERIMENTO")
print(f"{'='*60}")
print(f"Modelo: {MODEL_NAME}")
print(f"Tipo de artefacto: {NOISE_TYPE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Data points: {DATANUM}")
print(f"Dataset expansion: {COMBIN_NUM}x")
print(f"{'='*60}\n")

# ======================================================
# 5. CARGA DE DATOS
# ======================================================
EEG_all = np.load(os.path.join(DATA_DIR, EEG_FILE))
noise_all = np.load(os.path.join(DATA_DIR, NOISE_FILE))
print(f'[OK] EEG: {EEG_all.shape}, Ruido: {noise_all.shape}')

# ======================================================
# 6. PREPARACION DE DATOS
# ======================================================
(noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
 noiseEEG_test, EEG_test, test_std) = prepare_data(
    EEG_all=EEG_all, noise_all=noise_all,
    combin_num=COMBIN_NUM, train_per=0.8, noise_type=NOISE_TYPE)

print(f'[OK] Train: {noiseEEG_train.shape}, Val: {noiseEEG_val.shape}, Test: {noiseEEG_test.shape}')

noiseEEG_train = noiseEEG_train.astype(np.float32)
EEG_train = EEG_train.astype(np.float32)
noiseEEG_val = noiseEEG_val.astype(np.float32)
EEG_val = EEG_val.astype(np.float32)
noiseEEG_test = noiseEEG_test.astype(np.float32)
EEG_test = EEG_test.astype(np.float32)

# ======================================================
# 7. CREACION DEL MODELO
# ======================================================
if MODEL_NAME == 'fcNN':
    model = fcNN(DATANUM)
elif MODEL_NAME == 'Simple_CNN':
    model = simple_CNN(DATANUM)
elif MODEL_NAME == 'Complex_CNN':
    model = Complex_CNN(DATANUM)
elif MODEL_NAME == 'RNN_lstm':
    model = RNN_lstm(DATANUM)
else:
    raise ValueError(f'Modelo no reconocido: {MODEL_NAME}')

# Construir modelo
if MODEL_NAME == 'fcNN':
    model.build(input_shape=(None, DATANUM))
else:
    model.build(input_shape=(None, DATANUM, 1))

model.summary()

# ======================================================
# 8. VERIFICACION DE MEMORIA GPU
# ======================================================
# Estimar uso de memoria para validacion completa (sin batching)
val_size_mb = (noiseEEG_val.nbytes + EEG_val.nbytes) / (1024**2)
print(f"\n[INFO] Memoria estimada para validacion completa: {val_size_mb:.2f} MB")
print(f"[INFO] VRAM GPU disponible: ~16 GB (NVIDIA T4)")
if val_size_mb > 1024:
    print(f"[ADVERTENCIA] Validacion completa usa >1GB. Considerar batching.")
else:
    print(f"[OK] Validacion completa cabe comodamente en VRAM.")

# ======================================================
# 9. ENTRENAMIENTO
# ======================================================
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO (Version Hibrida)")
print("="*60 + "\n")

# Pasar numpy arrays
# train() internamente crea tf.data.Dataset para eficiencia GPU
saved_model, history = train(
    model=model,
    noiseEEG=noiseEEG_train,
    EEG=EEG_train,
    noiseEEG_val=noiseEEG_val,
    EEG_val=EEG_val,
    noiseEEG_test=noiseEEG_test,
    EEG_test=EEG_test,
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
os.makedirs(f'{RESULT_DIR}/{FOLDER_NAME}/1/nn_output', exist_ok=True)
np.save(f'{RESULT_DIR}/{FOLDER_NAME}/1/nn_output/loss_history.npy', history)

save_eeg(saved_model, RESULT_DIR, FOLDER_NAME,
         False, False, True,
         noiseEEG_train, EEG_train,
         noiseEEG_val, EEG_val,
         noiseEEG_test, EEG_test,
         train_num='1', denoise_network=MODEL_NAME, datanum=DATANUM)

print('\n' + '='*60)
print('ENTRENAMIENTO COMPLETADO')
print(f'Resultados guardados en: {RESULT_DIR}/{FOLDER_NAME}/1/')
print('='*60)
