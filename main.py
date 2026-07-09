import numpy as np
import os
import sys
import json
import time
import csv
from datetime import datetime

# ======================================================
# 1. CONFIGURACION GPU
# ======================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

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
from train_method import train, test_step
from save_method import save_eeg

# ======================================================
# 3. CONFIGURACION EXPERIMENTAL DEL ARTICULO
# ======================================================
# Paper: Zhang et al., 2021. EEGdenoiseNet.
# 4 modelos x 2 tipos de ruido x 10 repeticiones = 80 ejecuciones.

MODELS = ['fcNN', 'Simple_CNN', 'Complex_CNN', 'RNN_lstm']
NOISE_TYPES = ['EOG', 'EMG']
REPETITIONS = 10

# Epocas segun modelo y tipo de ruido (de hiperparametros.md)
EPOCHS_CONFIG = {
    'EOG': {'fcNN': 60, 'Simple_CNN': 40, 'Complex_CNN': 40, 'RNN_lstm': 100},
    'EMG': {'fcNN': 60, 'Simple_CNN': 10, 'Complex_CNN': 10, 'RNN_lstm': 60}
}

BATCH_SIZE = 40
COMBIN_NUM = 10  # Factor de expansion del dataset (10 niveles SNR)
DATA_DIR = './data'
RESULT_DIR = './results'

# ======================================================
# 4. FUNCIONES AUXILIARES
# ======================================================

def get_data_files(noise_type):
    """
    Devuelve (datanum, eeg_file, noise_file) segun el tipo de ruido.

    EOG usa senales de 512 puntos.
    EMG usa senales de 1024 puntos (archivos _512hz).
    """
    if noise_type == 'EOG':
        return 512, 'EEG_all_epochs.npy', 'EOG_all_epochs.npy'
    elif noise_type == 'EMG':
        return 1024, 'EEG_all_epochs_512hz.npy', 'EMG_all_epochs_512hz.npy'
    else:
        raise ValueError(f"Tipo de ruido no soportado: {noise_type}")


def build_model(model_name, datanum):
    """
    Crea una nueva instancia del modelo solicitado.

    fcNN espera entrada 2D (batch, datanum).
    CNN/RNN esperan entrada 3D (batch, datanum, 1).
    """
    if model_name == 'fcNN':
        model = fcNN(datanum)
        model.build(input_shape=(None, datanum))
    elif model_name == 'Simple_CNN':
        model = simple_CNN(datanum)
        model.build(input_shape=(None, datanum, 1))
    elif model_name == 'Complex_CNN':
        model = Complex_CNN(datanum)
        model.build(input_shape=(None, datanum, 1))
    elif model_name == 'RNN_lstm':
        model = RNN_lstm(datanum)
        model.build(input_shape=(None, datanum, 1))
    else:
        raise ValueError(f'Modelo no reconocido: {model_name}')
    return model


def build_optimizer():
    """Construye el optimizador Adam con los parametros del paper."""
    return tf.keras.optimizers.Adam(
        learning_rate=0.00005,
        beta_1=0.5,
        beta_2=0.9,
        epsilon=1e-08
    )


def compute_test_metrics(saved_model, noiseEEG_test, EEG_test, model_name, datanum):
    """Calcula las metricas finales sobre el conjunto de test."""
    if saved_model is None:
        return {
            'mse': None,
            'rrmse_t': None,
            'rrmse_s': None,
            'cc': None
        }

    EEG_test_r = tf.cast(EEG_test, tf.float32)
    if model_name == 'fcNN':
        noiseEEG_test_r = tf.cast(noiseEEG_test, tf.float32)
    else:
        noiseEEG_test_r = tf.cast(tf.reshape(noiseEEG_test, [-1, datanum, 1]), tf.float32)

    _, metrics = test_step(saved_model, noiseEEG_test_r, EEG_test_r)
    return {k: float(v) for k, v in metrics.items()}


def save_audit(run_dir, audit_info):
    """Guarda el fichero de auditoria JSON para una ejecucion."""
    audit_path = os.path.join(run_dir, 'audit.json')
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit_info, f, indent=2, ensure_ascii=False)
    print(f"[OK] Auditoria guardada en: {audit_path}")


def append_summary_csv(summary_csv_path, audit_info):
    """Anade una fila al resumen CSV global."""
    fieldnames = [
        'timestamp', 'model_name', 'noise_type', 'repetition',
        'datanum', 'epochs', 'batch_size', 'combin_num',
        'best_model_saved', 'training_time_seconds',
        'final_val_mse', 'final_val_rrmse_t', 'final_val_rrmse_s', 'final_val_cc',
        'test_mse', 'test_rrmse_t', 'test_rrmse_s', 'test_cc',
        'error'
    ]
    file_exists = os.path.isfile(summary_csv_path)
    with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        final_val = audit_info.get('final_val_metrics', {})
        test_metrics = audit_info.get('test_metrics', {})
        writer.writerow({
            'timestamp': audit_info.get('timestamp'),
            'model_name': audit_info.get('model_name'),
            'noise_type': audit_info.get('noise_type'),
            'repetition': audit_info.get('repetition'),
            'datanum': audit_info.get('datanum'),
            'epochs': audit_info.get('epochs'),
            'batch_size': audit_info.get('batch_size'),
            'combin_num': audit_info.get('combin_num'),
            'best_model_saved': audit_info.get('best_model_saved'),
            'training_time_seconds': audit_info.get('training_time_seconds'),
            'final_val_mse': final_val.get('mse'),
            'final_val_rrmse_t': final_val.get('rrmse_t'),
            'final_val_rrmse_s': final_val.get('rrmse_s'),
            'final_val_cc': final_val.get('cc'),
            'test_mse': test_metrics.get('mse'),
            'test_rrmse_t': test_metrics.get('rrmse_t'),
            'test_rrmse_s': test_metrics.get('rrmse_s'),
            'test_cc': test_metrics.get('cc'),
            'error': audit_info.get('error')
        })


def run_experiment(model_name, noise_type, repetition):
    """
    Ejecuta un unico experimento: carga datos, entrena y guarda resultados.

    Args:
        model_name: Nombre del modelo (fcNN, Simple_CNN, Complex_CNN, RNN_lstm).
        noise_type: Tipo de ruido ('EOG' o 'EMG').
        repetition: Numero de repeticion (0-indexado).

    Returns:
        dict: Informacion de auditoria del experimento.
    """
    # Liberar memoria de ejecuciones anteriores
    tf.keras.backend.clear_session()

    datanum, eeg_file, noise_file = get_data_files(noise_type)
    epochs = EPOCHS_CONFIG[noise_type][model_name]

    folder_name = f'{noise_type}_{model_name}_run'
    train_num = str(repetition + 1)
    run_dir = os.path.join(RESULT_DIR, folder_name, train_num)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONFIGURACION DEL EXPERIMENTO")
    print(f"{'='*60}")
    print(f"Modelo: {model_name}")
    print(f"Tipo de artefacto: {noise_type}")
    print(f"Repeticion: {train_num}/{REPETITIONS}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Data points: {datanum}")
    print(f"Dataset expansion: {COMBIN_NUM}x")
    print(f"Directorio de salida: {run_dir}")
    print(f"{'='*60}\n")

    # ======================================================
    # CARGA DE DATOS
    # ======================================================
    EEG_all = np.load(os.path.join(DATA_DIR, eeg_file))
    noise_all = np.load(os.path.join(DATA_DIR, noise_file))
    print(f'[OK] EEG: {EEG_all.shape}, Ruido: {noise_all.shape}')

    # ======================================================
    # PREPARACION DE DATOS
    # ======================================================
    (noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
     noiseEEG_test, EEG_test, test_std) = prepare_data(
        EEG_all=EEG_all, noise_all=noise_all,
        combin_num=COMBIN_NUM, train_per=0.8, noise_type=noise_type)

    print(f'[OK] Train: {noiseEEG_train.shape}, Val: {noiseEEG_val.shape}, Test: {noiseEEG_test.shape}')

    noiseEEG_train = noiseEEG_train.astype(np.float32)
    EEG_train = EEG_train.astype(np.float32)
    noiseEEG_val = noiseEEG_val.astype(np.float32)
    EEG_val = EEG_val.astype(np.float32)
    noiseEEG_test = noiseEEG_test.astype(np.float32)
    EEG_test = EEG_test.astype(np.float32)

    # ======================================================
    # CREACION DEL MODELO Y OPTIMIZADOR
    # ======================================================
    model = build_model(model_name, datanum)
    optimizer = build_optimizer()
    model.summary()

    # ======================================================
    # ENTRENAMIENTO
    # ======================================================
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60 + "\n")

    start_time = time.time()
    saved_model, history = train(
        model=model,
        noiseEEG=noiseEEG_train,
        EEG=EEG_train,
        noiseEEG_val=noiseEEG_val,
        EEG_val=EEG_val,
        noiseEEG_test=noiseEEG_test,
        EEG_test=EEG_test,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        optimizer=optimizer,
        denoise_network=model_name,
        result_location=RESULT_DIR,
        foldername=folder_name,
        train_num=train_num
    )
    training_time = time.time() - start_time

    # ======================================================
    # GUARDAR HISTORIAL DE ENTRENAMIENTO
    # ======================================================
    nn_output_dir = os.path.join(run_dir, 'nn_output')
    os.makedirs(nn_output_dir, exist_ok=True)
    np.save(os.path.join(nn_output_dir, 'loss_history.npy'), history)

    # ======================================================
    # GUARDAR RESULTADOS DE TEST (mejor modelo)
    # ======================================================
    test_metrics = {'mse': None, 'rrmse_t': None, 'rrmse_s': None, 'cc': None}
    if saved_model is not None:
        save_eeg(
            saved_model, RESULT_DIR, folder_name,
            False, False, True,
            noiseEEG_train, EEG_train,
            noiseEEG_val, EEG_val,
            noiseEEG_test, EEG_test,
            train_num=train_num, denoise_network=model_name, datanum=datanum
        )
        test_metrics = compute_test_metrics(
            saved_model, noiseEEG_test, EEG_test, model_name, datanum
        )
    else:
        print("[ADVERTENCIA] No se guardo ningun modelo (val_mse no mejoro). "
              "No se generaran archivos de test ni se calcularan metricas de test.")

    # ======================================================
    # CONSTRUIR AUDITORIA
    # ======================================================
    final_val_metrics = {
        'mse': float(history['loss']['val_mse'][-1]) if history['loss']['val_mse'] else None,
        'rrmse_t': float(history['loss']['val_rrmse_t'][-1]) if history['loss']['val_rrmse_t'] else None,
        'rrmse_s': float(history['loss']['val_rrmse_s'][-1]) if history['loss']['val_rrmse_s'] else None,
        'cc': float(history['loss']['val_cc'][-1]) if history['loss']['val_cc'] else None,
    }

    audit_info = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'noise_type': noise_type,
        'repetition': repetition + 1,
        'datanum': int(datanum),
        'epochs': int(epochs),
        'batch_size': int(BATCH_SIZE),
        'combin_num': int(COMBIN_NUM),
        'optimizer': {
            'name': 'Adam',
            'learning_rate': 0.00005,
            'beta_1': 0.5,
            'beta_2': 0.9,
            'epsilon': 1e-08
        },
        'data_files': {
            'EEG': eeg_file,
            'noise': noise_file
        },
        'data_shapes': {
            'EEG_all': list(EEG_all.shape),
            'noise_all': list(noise_all.shape),
            'train': list(noiseEEG_train.shape),
            'val': list(noiseEEG_val.shape),
            'test': list(noiseEEG_test.shape)
        },
        'training_time_seconds': float(training_time),
        'best_model_saved': saved_model is not None,
        'final_val_metrics': final_val_metrics,
        'test_metrics': test_metrics
    }

    save_audit(run_dir, audit_info)

    print('\n' + '='*60)
    print(f'EXPERIMENTO COMPLETADO: {model_name} / {noise_type} / rep {train_num}')
    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')
    print(f'Resultados guardados en: {run_dir}')
    print('='*60)

    return audit_info


# ======================================================
# 5. ORQUESTADOR PRINCIPAL
# ======================================================
def main():
    """Orquesta las 80 ejecuciones del articulo original."""
    os.makedirs(RESULT_DIR, exist_ok=True)

    summary_json_path = os.path.join(RESULT_DIR, 'summary.json')
    summary_csv_path = os.path.join(RESULT_DIR, 'summary.csv')

    summary = []
    total_runs = len(MODELS) * len(NOISE_TYPES) * REPETITIONS
    run_count = 0

    print("\n" + "="*60)
    print("ORQUESTADOR EEGdenoiseNet")
    print(f"Total de ejecuciones planificadas: {total_runs}")
    print(f"Modelos: {MODELS}")
    print(f"Tipos de ruido: {NOISE_TYPES}")
    print(f"Repeticiones: {REPETITIONS}")
    print("="*60 + "\n")

    orchestration_start = time.time()

    for noise_type in NOISE_TYPES:
        for model_name in MODELS:
            for rep in range(REPETITIONS):
                run_count += 1
                print("\n" + "="*60)
                print(f"EJECUCION {run_count}/{total_runs}")
                print(f"Modelo: {model_name} | Ruido: {noise_type} | Repeticion: {rep + 1}")
                print("="*60)

                try:
                    audit_info = run_experiment(model_name, noise_type, rep)
                    summary.append(audit_info)
                except Exception as e:
                    error_msg = str(e)
                    print(f"[ERROR] La ejecucion {run_count} fallo: {error_msg}")
                    summary.append({
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'noise_type': noise_type,
                        'repetition': rep + 1,
                        'error': error_msg
                    })

                # Guardar resumen parcial despues de cada ejecucion
                with open(summary_json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                append_summary_csv(summary_csv_path, summary[-1])

    orchestration_time = time.time() - orchestration_start

    # Guardar resumen final
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    successful = sum(1 for item in summary if 'error' not in item)

    print("\n" + "="*60)
    print("ORQUESTACION FINALIZADA")
    print(f"Ejecuciones exitosas: {successful}/{total_runs}")
    print(f"Ejecuciones con error: {total_runs - successful}/{total_runs}")
    print(f"Tiempo total: {orchestration_time:.2f} segundos "
          f"({orchestration_time/3600:.2f} horas)")
    print(f"Resumen JSON: {summary_json_path}")
    print(f"Resumen CSV: {summary_csv_path}")
    print("="*60)


if __name__ == '__main__':
    main()
