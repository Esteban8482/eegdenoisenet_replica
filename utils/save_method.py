"""
Metodos de guardado para EEGdenoiseNet - Version GPU
=====================================================
Cambios realizados:
1. Convierte tensores GPU a numpy arrays antes de guardar en disco
   (np.save no puede escribir tensores TF directamente)
2. Usa test_step_vectorizado para predicciones en GPU
3. Deteccion automatica de tipo (tensor vs numpy)
"""

import numpy as np
import tensorflow as tf
import os

from train_method_gpu import test_step_vectorized


def to_numpy(data):
    """
    Convierte un tensor de TensorFlow a numpy array si es necesario.
    Si ya es numpy, lo retorna sin cambios.

    Args:
        data: TensorFlow tensor o numpy array

    Returns:
        numpy array
    """
    if isinstance(data, tf.Tensor):
        return data.numpy()
    return data


def save_eeg(saved_model, result_location, foldername,
             save_train, save_vali, save_test,
             noiseEEG_train, EEG_train,
             noiseEEG_val, EEG_val,
             noiseEEG_test, EEG_test,
             train_num, denoise_network='Simple_CNN', datanum=512):
    """
    Guarda las senales denoised y las originales en archivos .npy.

    CAMBIO PRINCIPAL: Convierte tensores GPU a numpy antes de np.save().
    La version original trabajaba con numpy arrays directamente, pero ahora
    los datos son tensores TF residentes en GPU. np.save() no acepta tensores,
    asi que debemos convertirlos primero con .numpy() (trae datos de GPU a CPU).

    Args:
        saved_model: Modelo entrenado (Keras model)
        result_location: Directorio base para resultados
        foldername: Nombre de la carpeta del experimento
        save_train: Boolean - guardar datos de entrenamiento
        save_vali: Boolean - guardar datos de validacion
        save_test: Boolean - guardar datos de test
        noiseEEG_train: Tensor/numpy - EEG con ruido (train)
        EEG_train: Tensor/numpy - EEG limpio (train)
        noiseEEG_val: Tensor/numpy - EEG con ruido (val)
        EEG_val: Tensor/numpy - EEG limpio (val)
        noiseEEG_test: Tensor/numpy - EEG con ruido (test)
        EEG_test: Tensor/numpy - EEG limpio (test)
        train_num: String identificador del entrenamiento
        denoise_network: Nombre del modelo usado
        datanum: Numero de puntos de muestra (512 o 1024)
    """
    output_dir = os.path.join(result_location, foldername, train_num, "nn_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Guardar datos de entrenamiento ---
    if save_train:
        print("[GPU→CPU] Procesando y guardando datos de entrenamiento...")

        # Prediccion en GPU (mas rapido)
        Denoiseoutput_train, train_mse = test_step_vectorized(
            saved_model, noiseEEG_train, EEG_train, denoise_network, datanum
        )

        # Convertir GPU→CPU antes de guardar
        np.save(os.path.join(output_dir, "noiseinput_train.npy"),
                to_numpy(noiseEEG_train))
        np.save(os.path.join(output_dir, "Denoiseoutput_train.npy"),
                to_numpy(Denoiseoutput_train))
        np.save(os.path.join(output_dir, "EEG_train.npy"),
                to_numpy(EEG_train))

        print(f"  Train MSE: {float(train_mse):.6f}")

    # --- Guardar datos de validacion ---
    if save_vali:
        print("[GPU→CPU] Procesando y guardando datos de validacion...")

        Denoiseoutput_val, val_mse = test_step_vectorized(
            saved_model, noiseEEG_val, EEG_val, denoise_network, datanum
        )

        np.save(os.path.join(output_dir, "noiseinput_val.npy"),
                to_numpy(noiseEEG_val))
        np.save(os.path.join(output_dir, "Denoiseoutput_val.npy"),
                to_numpy(Denoiseoutput_val))
        np.save(os.path.join(output_dir, "EEG_val.npy"),
                to_numpy(EEG_val))

        print(f"  Val MSE: {float(val_mse):.6f}")

    # --- Guardar datos de test ---
    if save_test:
        print("[GPU→CPU] Procesando y guardando datos de test...")

        Denoiseoutput_test, test_mse = test_step_vectorized(
            saved_model, noiseEEG_test, EEG_test, denoise_network, datanum
        )

        np.save(os.path.join(output_dir, "noiseinput_test.npy"),
                to_numpy(noiseEEG_test))
        np.save(os.path.join(output_dir, "Denoiseoutput_test.npy"),
                to_numpy(Denoiseoutput_test))
        np.save(os.path.join(output_dir, "EEG_test.npy"),
                to_numpy(EEG_test))

        print(f"  Test MSE: {float(test_mse):.6f}")

    print(f"[OK] Archivos guardados en: {output_dir}")
