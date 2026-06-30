"""
Funciones de perdida para EEGdenoiseNet - Version GPU
======================================================
Cambios realizados:
1. API moderna de TensorFlow 2.x
2. Uso de tf.keras.losses en lugar de tf.losses (obsoleto)
3. Tipos de datos explícitos para compatibilidad GPU
"""

import tensorflow as tf

# Instancia reutilizable de MSE (más eficiente)
mse_loss_fn = tf.keras.losses.MeanSquaredError()


def denoise_loss_mse(denoise, clean):
    """
    Calcula el error cuadratico medio (MSE) entre la senal denoised y la limpia.

    Args:
        denoise: Tensor con la senal denoised (prediccion del modelo)
        clean: Tensor con la senal EEG limpia (ground truth)

    Returns:
        Scalar tensor con el valor promedio de MSE
    """
    return mse_loss_fn(clean, denoise)


def denoise_loss_rmse(denoise, clean):
    """
    Calcula la raiz del error cuadratico medio (RMSE).

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia

    Returns:
        Scalar tensor con el valor de RMSE
    """
    mse = mse_loss_fn(clean, denoise)
    return tf.math.sqrt(mse)


def denoise_loss_rrmset(denoise, clean):
    """
    Calcula el RMSE relativo (RRMSE) entre la senal denoised y la limpia.
    Metrica comun en papers de denoising de senales fisiologicas.

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia

    Returns:
        Scalar tensor con el valor de RRMSE
    """
    rmse_signal = denoise_loss_rmse(denoise, clean)
    # RMSE de la senal limpia contra cero (energia de referencia)
    rmse_reference = denoise_loss_rmse(clean, tf.zeros_like(clean))
    return rmse_signal / (rmse_reference + 1e-8)  # epsilon para evitar div/0
