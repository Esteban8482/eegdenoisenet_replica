import tensorflow as tf
import time
from tqdm import tqdm
import os
import math

from loss_function import denoise_loss_mse


# ======================================================
# 1. TRAIN_STEP VECTORIZADO MATEMATICAMENTE EQUIVALENTE
# ======================================================
@tf.function
def train_step(model, noiseEEG_batch, EEG_batch, optimizer, denoise_network, datanum):
    """
    Paso de entrenamiento vectorizado.

    Args:
        model: Instancia del modelo Keras
        noiseEEG_batch: Batch de EEG con ruido [batch_size, datanum]
        EEG_batch: Batch de EEG limpio [batch_size, datanum]
        optimizer: Optimizador de TensorFlow
        denoise_network: String con el nombre del modelo ('fcNN', etc.)
        datanum: Numero de puntos de muestra (512 o 1024)

    Returns:
        M_loss: Perdida promedio del batch (scalar tensor)
        mse_grads[0]: Primer gradiente
    """
    # Reshape segun el tipo de red neuronal
    if denoise_network == 'fcNN':
        noiseEEG_batch_r = tf.reshape(noiseEEG_batch, [-1, datanum])
    else:
        noiseEEG_batch_r = tf.reshape(noiseEEG_batch, [-1, datanum, 1])

    EEG_batch_r = tf.cast(tf.reshape(EEG_batch, [-1, datanum, 1]), tf.float32)

    with tf.GradientTape() as loss_tape:
        # Forward pass: batch completo en GPU
        denoiseoutput = model(noiseEEG_batch_r, training=True)
        denoiseoutput = tf.reshape(denoiseoutput, [-1, datanum, 1])

        # Perdida MSE sobre el batch completo
        # MSE(batch) == mean(MSE(muestra_i)) por linealidad del promedio
        M_loss = denoise_loss_mse(denoiseoutput, EEG_batch_r)

    # Backpropagation
    mse_grads = loss_tape.gradient(M_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(mse_grads, model.trainable_variables))

    # Retornar solo el primer gradiente
    return M_loss, mse_grads[0]


# ======================================================
# 2. TEST_STEP SIN @tf.function 
# ======================================================
def test_step(model, noiseEEG_test, EEG_test):
    """
    Paso de validacion/prueba

    Args:
        model: Instancia del modelo Keras
        noiseEEG_test: Datos de test/validacion con ruido
        EEG_test: Datos de test/validacion limpios

    Returns:
        denoiseoutput_test: Senal denoised
        loss: Valor de perdida MSE
    """
    denoiseoutput_test = model(noiseEEG_test, training=False)
    loss = denoise_loss_mse(EEG_test, denoiseoutput_test)

    return denoiseoutput_test, loss


# ======================================================
# 3. FUNCION DE ENTRENAMIENTO
# ======================================================
def train(model, noiseEEG, EEG, noiseEEG_val, EEG_val,
          noiseEEG_test, EEG_test,
          epochs, batch_size, optimizer, denoise_network,
          result_location, foldername, train_num):
    """
    Entrena el modelo de denoising EEG

    Combina eficiencia GPU con fidelidad metodologica:
    - tf.data.Dataset para pipeline eficiente (con drop_remainder=False)
    - Validacion sobre dataset completo
    - test_step sin @tf.function
    - train_step vectorizado pero matematicamente equivalente

    Args:
        model: Modelo Keras a entrenar
        noiseEEG: Tensor/array de entrenamiento con ruido [N, datanum]
        EEG: Tensor/array de entrenamiento limpio [N, datanum]
        noiseEEG_val: Tensor/array de validacion con ruido
        EEG_val: Tensor/array de validacion limpio
        noiseEEG_test: Tensor/array de test con ruido
        EEG_test: Tensor/array de test limpio
        epochs: Numero de epocas
        batch_size: Tamano de batch
        optimizer: Optimizador de TensorFlow
        denoise_network: Nombre del modelo
        result_location: Directorio para guardar resultados
        foldername: Nombre de la carpeta de resultados
        train_num: Identificador del entrenamiento

    Returns:
        saved_model: Mejor modelo guardado
        history: Diccionario con historial de entrenamiento
    """

    # --- 3.1 Inicializar historial  ---
    history = {}
    history['grads'], history['loss'] = {}, {}
    train_mse_history, val_mse_history = [], []
    mse_grads_history = []
    val_mse_min = 100.0  # cualquier numero mayor que 1 
    saved_model = None
    datanum = noiseEEG.shape[1]

    # --- 3.2 Crear directorios para TensorBoard ---
    train_log_dir = result_location + '/' + foldername + '/' + train_num + '/train'
    val_log_dir = result_location + '/' + foldername + '/' + train_num + '/test'
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # --- 3.3 Calcular numero de batches  ---
    # El original usa math.ceil para incluir el ultimo batch parcial
    batch_num = math.ceil(noiseEEG.shape[0] / batch_size)

    # --- 3.4 Crear tf.data.Dataset (EFICIENTE pero con drop_remainder=False) ---
    # drop_remainder=False preserva el ultimo batch parcial
    print(f"[GPU] Pipeline tf.data.Dataset (batch_size={batch_size}, drop_remainder=False)...")

    dataset = tf.data.Dataset.from_tensor_slices((noiseEEG, EEG))
    dataset = dataset.shuffle(buffer_size=noiseEEG.shape[0], reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # --- 3.5 Bucle de entrenamiento ---
    for epoch in range(epochs):
        start = time.time()

        # Inicializar metricas de la epoca
        mse_grads_epoch, train_mse = 0, 0

        with tqdm(total=batch_num, position=0, leave=True) as pbar:
            for noiseEEG_batch, EEG_batch in dataset:
                # Train step (vectorizado pero equivalente)
                mse_loss_batch, mse_grads_batch = train_step(
                    model, noiseEEG_batch, EEG_batch,
                    optimizer, denoise_network, datanum
                )

                # Convertir a formato usable
                mse_grads_batch = tf.reduce_mean(
                    tf.sqrt(tf.reduce_sum(tf.square(mse_grads_batch)))
                ).numpy()
                mse_loss_batch = tf.reduce_mean(mse_loss_batch).numpy()

                # Acumular metricas (promedio sobre batch_num)
                train_mse += mse_loss_batch / float(batch_num)
                mse_grads_epoch += mse_grads_batch / float(batch_num)

                pbar.update()
            pbar.close()

        # --- 3.6 Guardar historial de entrenamiento ---
        mse_grads_history.append(mse_grads_epoch)
        train_mse_history.append(train_mse)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_mse, step=epoch)

        # --- 3.7 Validacion sobre dataset completo ---

        # Preparar datos de validacion (reshape segun tipo de red)
        if denoise_network == 'fcNN':
            noiseEEG_val_r = tf.cast(noiseEEG_val, tf.float32)
            EEG_val_r = tf.cast(tf.reshape(EEG_val, [-1, datanum, 1]), tf.float32)
        else:
            noiseEEG_val_r = tf.cast(tf.reshape(noiseEEG_val, [-1, datanum, 1]), tf.float32)
            EEG_val_r = tf.cast(tf.reshape(EEG_val, [-1, datanum, 1]), tf.float32)

        denoiseoutput, val_mse = test_step(model, noiseEEG_val_r, EEG_val_r)

        # Guardar historial de validacion
        val_mse_history.append(val_mse)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_mse, step=epoch)

        # --- 3.8 Guardar mejor modelo ---
        if epoch > epochs * 0.8 and float(val_mse) < val_mse_min:
            print('yes,smaller ', float(val_mse), val_mse_min)
            val_mse_min = float(val_mse)
            saved_model = model

            path = os.path.join(result_location, foldername, train_num, "denoise_model")
            tf.keras.models.save_model(model, path)
            print('Best model has been saved')

        # --- 3.9 Reporte de epoca ---
        print('Epoch #: {}/{}, Time taken: {} secs,\\n Grads: mse= {},\\n Losses: train_mse= {}, val_mse={}'
              .format(epoch + 1, epochs, time.time() - start, mse_grads_epoch, train_mse, val_mse))

    # --- 3.10 Finalizar ---
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except ImportError:
        pass  # No es Jupyter, ignorar

    # Guardar historial en diccionario
    history['grads']['mse'] = mse_grads_history
    history['loss']['train_mse'] = train_mse_history
    history['loss']['val_mse'] = val_mse_history

    return saved_model, history
