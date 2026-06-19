import tensorflow as tf
import numpy as np
import os
import sys

# Importar modulos locales
sys.path.append('models')
sys.path.append('utils')

from network_structure import fcNN, simple_CNN, Complex_CNN, RNN_lstm
from data_prepare import prepare_data
from train_method import train, test_step
from save_method import save_eeg

# ======================================================
# PARAMETROS CONFIGURABLES
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

# Optimizador
# Adam: lr=5e-5, beta_1=0.5, beta_2=0.9
# RMSprop: lr=5e-5, rho=0.9
optimizer = tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9)

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ======================================================
# CONFIGURACION AUTOMATICA
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
# CARGA DE DATOS
# ======================================================
EEG_all = np.load(os.path.join(DATA_DIR, EEG_FILE))
noise_all = np.load(os.path.join(DATA_DIR, NOISE_FILE))
print(f'EEG: {EEG_all.shape}, Ruido: {noise_all.shape}')

# ======================================================
# PREPARACION DE DATOS
# ======================================================
(noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
 noiseEEG_test, EEG_test, test_std) = prepare_data(
    EEG_all=EEG_all, noise_all=noise_all,
    combin_num=COMBIN_NUM, train_per=0.8, noise_type=NOISE_TYPE)

print(f'Train: {noiseEEG_train.shape}, Val: {noiseEEG_val.shape}, '
      f'Test: {noiseEEG_test.shape}')

# ======================================================
# CREACION DEL MODELO
# ======================================================
if MODEL_NAME == 'fcNN':
    model = fcNN(DATANUM)
elif MODEL_NAME == 'Simple_CNN':
    model = simple_CNN(DATANUM)
elif MODEL_NAME == 'Complex_CNN':
    model = Complex_CNN(DATANUM)
elif MODEL_NAME == 'RNN_lstm':
    model = RNN_lstm(DATANUM)

# ======================================================
# ENTRENAMIENTO
# ======================================================
saved_model, history = train(
    model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
    EPOCHS, BATCH_SIZE, optimizer, MODEL_NAME,
    RESULT_DIR, FOLDER_NAME, train_num='1')

# Guardar historial de perdida
np.save(f'{RESULT_DIR}/{FOLDER_NAME}/1/loss_history.npy', history)

# Guardar senales de test
save_eeg(saved_model, RESULT_DIR, FOLDER_NAME,
         False, False, True,
         noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
         noiseEEG_test, EEG_test, train_num='1')

print('=== ENTRENAMIENTO COMPLETADO ===')