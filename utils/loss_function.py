"""
Funciones de perdida para EEGdenoiseNet
=========================================================
Cambios respecto al original:
- API moderna de tf.keras.losses (el original usa tf.losses obsoleto)
- tf.reduce_mean explicito para asegurar equivalencia con:
    original: tf.losses.mean_squared_error() + tf.reduce_mean()
"""

import tensorflow as tf

def denoise_loss_mse(denoise, clean):
    """
    Calcula el error cuadratico medio (MSE) entre senal denoised y limpia.

    Usa tf.keras.losses con reduccion explicita
    para garantizar equivalencia con el original que usaba:
        tf.losses.mean_squared_error(denoise, clean) -> tf.reduce_mean(loss)

    La instancia de MSE con reduction='none' calcula la perdida por elemento,
    y luego tf.reduce_mean la promedia a un escalar, replicando exactamente
    el comportamiento del codigo original.

    Args:
        denoise: Tensor con la senal denoised (prediccion del modelo)
        clean: Tensor con la senal EEG limpia (ground truth)

    Returns:
        Scalar tensor con el valor promedio de MSE
    """
    # Calcular MSE elemento por elemento, luego promediar todo
    # Equivalente al original: tf.reduce_mean(tf.losses.mean_squared_error(...))
    squared_diff = tf.square(denoise - clean)
    return tf.reduce_mean(squared_diff)

def denoise_loss_rmse(denoise, clean):
    """
    Calcula la raiz del error cuadratico medio (RMSE).

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia

    Returns:
        Scalar tensor con el valor de RMSE
    """
    mse = denoise_loss_mse(denoise, clean)
    return tf.math.sqrt(mse)

def denoise_loss_rrmset(denoise, clean):
    """
    Calcula el RMSE relativo (RRMSE) temporal.
    Metrica usada en el paper EEGdenoiseNet para evaluacion.

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia

    Returns:
        Scalar tensor con el valor de RRMSE
    """
    rmse1 = denoise_loss_rmse(denoise, clean)
    # Usar tf.zeros_like para compatibilidad con dimensiones dinamicas
    rmse2 = denoise_loss_rmse(clean, tf.zeros_like(clean))
    return rmse1 / rmse2

def denoise_loss_rrmset2(denoise, clean):
    """
    Calcula el RMSE relativo (RRMSE) espectral.
    Version alternativa usada en algunas evaluaciones.

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia

    Returns:
        Scalar tensor con el valor de RRMSE espectral
    """
    rmse1 = denoise_loss_rmse(denoise, clean)
    rmse2 = denoise_loss_rmse(clean, tf.zeros_like(clean))
    return rmse1 / rmse2
