import numpy as np
import tensorflow as tf
import os

from train_method import test_step


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


def _reshape_for_model(noiseEEG, EEG, denoise_network, datanum):
    """
    Reshapea los datos segun el tipo de red neuronal.

    fcNN espera entrada: [batch, datanum] (2D)
    CNN/RNN esperan entrada: [batch, datanum, 1] (3D)

    Args:
        noiseEEG: Array/tensor de EEG con ruido [N, datanum]
        EEG: Array/tensor de EEG limpio [N, datanum]
        denoise_network: String con nombre del modelo ('fcNN', etc.)
        datanum: Numero de puntos de muestra (512 o 1024)

    Returns:
        tuple: (noiseEEG_reshaped, EEG_reshaped)
    """
    noiseEEG = tf.cast(noiseEEG, tf.float32)
    EEG = tf.cast(EEG, tf.float32)
    
    if denoise_network == 'fcNN':
        noiseEEG_r = noiseEEG
    else:
        noiseEEG_r = tf.reshape(noiseEEG, [-1, datanum, 1])
    EEG_r = EEG

    return noiseEEG_r, EEG_r


def save_eeg(saved_model, result_location, foldername,
             save_train, save_vali, save_test,
             noiseEEG_train, EEG_train,
             noiseEEG_val, EEG_val,
             noiseEEG_test, EEG_test,
             train_num, denoise_network='Simple_CNN', datanum=512):
    """
    Guarda las senales denoised y las originales en archivos .npy.

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
        denoise_network: Nombre del modelo usado (para reshape)
        datanum: Numero de puntos de muestra (para reshape)
    """
    output_dir = os.path.join(result_location, foldername, train_num, "nn_output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Guardar datos de entrenamiento ---
    if save_train:
        print("[SAVE] Procesando y guardando datos de entrenamiento...")

        # Reshapear segun tipo de red, luego llamar test_step
        noiseEEG_train_r, EEG_train_r = _reshape_for_model(
            noiseEEG_train, EEG_train, denoise_network, datanum
        )
        Denoiseoutput_train, train_metrics = test_step(
            saved_model, noiseEEG_train_r, EEG_train_r
        )

        np.save(os.path.join(output_dir, "noiseinput_train.npy"),
                to_numpy(noiseEEG_train))
        np.save(os.path.join(output_dir, "Denoiseoutput_train.npy"),
                to_numpy(Denoiseoutput_train))
        np.save(os.path.join(output_dir, "EEG_train.npy"),
                to_numpy(EEG_train))

        print(f"  [OK] Train MSE: {float(train_metrics['mse']):.6f}, "
              f"RRMSE_t: {float(train_metrics['rrmse_t']):.6f}, "
              f"RRMSE_s: {float(train_metrics['rrmse_s']):.6f}, "
              f"CC: {float(train_metrics['cc']):.6f}")

    # --- Guardar datos de validacion ---
    if save_vali:
        print("[SAVE] Procesando y guardando datos de validacion...")

        noiseEEG_val_r, EEG_val_r = _reshape_for_model(
            noiseEEG_val, EEG_val, denoise_network, datanum
        )
        Denoiseoutput_val, val_metrics = test_step(
            saved_model, noiseEEG_val_r, EEG_val_r
        )

        np.save(os.path.join(output_dir, "noiseinput_val.npy"),
                to_numpy(noiseEEG_val))
        np.save(os.path.join(output_dir, "Denoiseoutput_val.npy"),
                to_numpy(Denoiseoutput_val))
        np.save(os.path.join(output_dir, "EEG_val.npy"),
                to_numpy(EEG_val))

        print(f"  [OK] Val MSE: {float(val_metrics['mse']):.6f}, "
              f"RRMSE_t: {float(val_metrics['rrmse_t']):.6f}, "
              f"RRMSE_s: {float(val_metrics['rrmse_s']):.6f}, "
              f"CC: {float(val_metrics['cc']):.6f}")

    # --- Guardar datos de test ---
    if save_test:
        print("[SAVE] Procesando y guardando datos de test...")

        noiseEEG_test_r, EEG_test_r = _reshape_for_model(
            noiseEEG_test, EEG_test, denoise_network, datanum
        )
        Denoiseoutput_test, test_metrics = test_step(
            saved_model, noiseEEG_test_r, EEG_test_r
        )

        np.save(os.path.join(output_dir, "noiseinput_test.npy"),
                to_numpy(noiseEEG_test))
        np.save(os.path.join(output_dir, "Denoiseoutput_test.npy"),
                to_numpy(Denoiseoutput_test))
        np.save(os.path.join(output_dir, "EEG_test.npy"),
                to_numpy(EEG_test))

        print(f"  [OK] Test MSE: {float(test_metrics['mse']):.6f}, "
              f"RRMSE_t: {float(test_metrics['rrmse_t']):.6f}, "
              f"RRMSE_s: {float(test_metrics['rrmse_s']):.6f}, "
              f"CC: {float(test_metrics['cc']):.6f}")

    print(f"[OK] Archivos guardados en: {output_dir}")
