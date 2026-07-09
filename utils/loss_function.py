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

    Args:
        denoise: Tensor con la senal denoised (prediccion del modelo)
        clean: Tensor con la senal EEG limpia (ground truth)

    Returns:
        Scalar tensor con el valor promedio de MSE
    """
    # Aplanar para tolerar diferencias de forma (p. ej. [N, samples] vs
    # [N, samples, 1]) sin cambiar el valor del MSE.
    denoise = tf.reshape(denoise, [-1])
    clean = tf.reshape(clean, [-1])

    # Calcular MSE elemento por elemento, luego promediar todo
    # tf.reduce_mean(tf.losses.mean_squared_error(...))
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


def denoise_loss_rrmses(denoise, clean):
    """
    Calcula el RMSE relativo (RRMSE) espectral.

    Compara los espectros de magnitud (valor absoluto de la FFT) de la
    senal denoised y la senal limpia, normalizando por la energia del
    espectro limpio.

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia

    Returns:
        Scalar tensor con el valor de RRMSE espectral
    """
    # Asegurar forma [batch, samples] para la FFT
    denoise = tf.squeeze(denoise)
    clean = tf.squeeze(clean)

    # Espectros de magnitud
    denoise_fft = tf.abs(tf.signal.rfft(denoise))
    clean_fft = tf.abs(tf.signal.rfft(clean))

    rmse1 = denoise_loss_rmse(denoise_fft, clean_fft)
    rmse2 = denoise_loss_rmse(clean_fft, tf.zeros_like(clean_fft))

    # Evitar division por cero
    return rmse1 / (rmse2 + 1e-12)


def denoise_loss_pearson(denoise, clean, eps=1e-8):
    """
    Calcula el coeficiente de correlacion de Pearson promedio.

    Se calcula la correlacion entre cada par de epocas (denoised, clean)
    y se promedia sobre todo el batch. El valor varia entre -1 y 1;
    valores cercanos a 1 indican mayor similitud.

    Args:
        denoise: Tensor con la senal denoised
        clean: Tensor con la senal EEG limpia
        eps: Valor pequeno para evitar division por cero

    Returns:
        Scalar tensor con el CC de Pearson promedio
    """
    # Aplanar cada epoca a [batch, samples]
    denoise = tf.reshape(denoise, [tf.shape(denoise)[0], -1])
    clean = tf.reshape(clean, [tf.shape(clean)[0], -1])

    # Medias por epoca
    denoise_mean = tf.reduce_mean(denoise, axis=1, keepdims=True)
    clean_mean = tf.reduce_mean(clean, axis=1, keepdims=True)

    # Senales centradas
    denoise_centered = denoise - denoise_mean
    clean_centered = clean - clean_mean

    # Covarianza y varianzas
    cov = tf.reduce_sum(denoise_centered * clean_centered, axis=1)
    denoise_var = tf.reduce_sum(tf.square(denoise_centered), axis=1)
    clean_var = tf.reduce_sum(tf.square(clean_centered), axis=1)

    pearson = cov / (tf.sqrt(denoise_var * clean_var) + eps)

    # Promedio sobre el batch
    return tf.reduce_mean(pearson)

