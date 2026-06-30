"""
Metodos de entrenamiento para EEGdenoiseNet - Version GPU Optimizada
=====================================================================
1. TRAIN_STEP VECTORIZADO: Elimina el loop Python muestra-por-muestra
   - Procesa todo el batch en paralelo en GPU
2. Pipeline tf.data.Dataset para transferencia asincrona CPU→GPU
3. @tf.function con input_signature para grafo optimizado
4. test_step vectorizado con @tf.function
5. Monitoreo de memoria GPU en cada epoch
"""

import tensorflow as tf
import time
from tqdm import tqdm
import os
import math

from loss_function_gpu import denoise_loss_mse


# ======================================================
# 1. TRAIN_STEP VECTORIZADO (PROCESA BATCH COMPLETO)
# ======================================================
@tf.function
def train_step_vectorized(model, noiseEEG_batch, EEG_batch, optimizer, denoise_network, datanum):
    """
    Paso de entrenamiento vectorizado - procesa todo el batch en paralelo.

    CAMBIO CLAVE: La version original iteraba muestra por muestra con un loop
    Python (for x in range(batch_size)), lo cual hacia que la GPU procesara
    una muestra a la vez. Esta version procesa todo el batch simultaneamente,
    logrando paralelismo completo en los 896 CUDA cores de la GTX 1650.

    Args:
        model: Instancia del modelo Keras
        noiseEEG_batch: Batch de EEG con ruido [batch_size, datanum]
        EEG_batch: Batch de EEG limpio [batch_size, datanum]
        optimizer: Optimizador de TensorFlow
        denoise_network: String con el nombre del modelo ('fcNN', etc.)
        datanum: Numero de puntos de muestra (512 o 1024)

    Returns:
        M_loss: Perdida promedio del batch (scalar tensor)
        mse_grads_norm: Norma L2 del gradiente (scalar tensor)
    """
    # Reshape segun el tipo de red neuronal
    if denoise_network == 'fcNN':
        noiseEEG_batch_r = tf.reshape(noiseEEG_batch, [-1, datanum])
    else:
        noiseEEG_batch_r = tf.reshape(noiseEEG_batch, [-1, datanum, 1])

    EEG_batch_r = tf.reshape(EEG_batch, [-1, datanum, 1])

    with tf.GradientTape() as tape:
        # Forward pass: procesa TODO el batch en una sola llamada
        denoiseoutput = model(noiseEEG_batch_r, training=True)
        denoiseoutput = tf.reshape(denoiseoutput, [-1, datanum, 1])

        # Perdida: calcula MSE para todo el batch simultaneamente
        M_loss = denoise_loss_mse(denoiseoutput, EEG_batch_r)

    # Backpropagation
    mse_grads = tape.gradient(M_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(mse_grads, model.trainable_variables))

    # Norma del gradiente para monitoreo
    mse_grads_norm = tf.reduce_mean(tf.sqrt(
        tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in mse_grads])
    ))

    return M_loss, mse_grads_norm


# ======================================================
# 2. TEST_STEP VECTORIZADO CON @tf.function
# ======================================================
@tf.function
def test_step_vectorized(model, noiseEEG_test, EEG_test, denoise_network, datanum):
    """
    Paso de validacion/prueba vectorizado con @tf.function para GPU.

    CAMBIO: La version original no usaba @tf.function, por lo que cada
    llamada a test_step se ejecutaba en modo eager (lento). Ahora se
    compila a grafo para maxima velocidad en GPU.

    Args:
        model: Instancia del modelo Keras
        noiseEEG_test: Datos de test con ruido
        EEG_test: Datos de test limpios
        denoise_network: String con el nombre del modelo
        datanum: Numero de puntos de muestra

    Returns:
        denoiseoutput: Senal denoised
        loss: Valor de perdida MSE
    """
    # Reshape segun el tipo de red
    if denoise_network == 'fcNN':
        noiseEEG_test_r = tf.reshape(noiseEEG_test, [-1, datanum])
    else:
        noiseEEG_test_r = tf.reshape(noiseEEG_test, [-1, datanum, 1])

    EEG_test_r = tf.reshape(EEG_test, [-1, datanum, 1])

    # Forward pass
    denoiseoutput = model(noiseEEG_test_r, training=False)
    denoiseoutput = tf.reshape(denoiseoutput, [-1, datanum, 1])

    # Perdida
    loss = denoise_loss_mse(denoiseoutput, EEG_test_r)

    return denoiseoutput, loss


# ======================================================
# 3. FUNCION DE ENTRENAMIENTO PRINCIPAL
# ======================================================
def train(model, noiseEEG, EEG, noiseEEG_val, EEG_val,
          noiseEEG_test, EEG_test,
          epochs, batch_size, optimizer, denoise_network,
          result_location, foldername, train_num):
    """
    Entrena el modelo de denoising EEG usando GPU.

    CAMBIOS PRINCIPALES:
    - Usa tf.data.Dataset para pipeline eficiente CPU→GPU
    - train_step vectorizado procesa batches completos en paralelo
    - Monitoreo de memoria GPU cada epoch (util para GTX 1650 4GB)
    - test_step con @tf.function para validacion rapida

    Args:
        model: Modelo Keras a entrenar
        noiseEEG: Tensor de entrenamiento con ruido [N, datanum]
        EEG: Tensor de entrenamiento limpio [N, datanum]
        noiseEEG_val: Tensor de validacion con ruido
        EEG_val: Tensor de validacion limpio
        noiseEEG_test: Tensor de test con ruido
        EEG_test: Tensor de test limpio
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

    # --- 3.1 Inicializar historial ---
    history = {
        'grads': {'mse': []},
        'loss': {
            'train_mse': [],
            'val_mse': [],
            'test_mse': []
        }
    }
    val_mse_min = 100.0
    saved_model = None
    datanum = noiseEEG.shape[1]

    # --- 3.2 Crear directorios para logs ---
    train_log_dir = os.path.join(result_location, foldername, train_num, 'train')
    val_log_dir = os.path.join(result_location, foldername, train_num, 'test')
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # --- 3.3 Crear tf.data.Dataset (pipeline eficiente) ---
    # tf.data.Dataset maneja automaticamente:
    # - Prefetching de batches (carga el siguiente batch mientras GPU entrena)
    # - Transferencia asincrona CPU→GPU
    # - Mejor utilizacion de la GTX 1650
    print(f"[GPU] Creando pipeline tf.data.Dataset (batch_size={batch_size})...")

    dataset = tf.data.Dataset.from_tensor_slices((noiseEEG, EEG))
    dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Dataset de validacion (sin shuffle)
    val_dataset = tf.data.Dataset.from_tensor_slices((noiseEEG_val, EEG_val))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    # --- 3.4 Bucle de entrenamiento ---
    for epoch in range(epochs):
        start_time = time.time()

        # Inicializar metricas de la epoca
        epoch_loss = 0.0
        epoch_grads = 0.0
        num_batches = 0

        # Barra de progreso
        total_batches = tf.math.ceil(tf.cast(noiseEEG.shape[0], tf.float32) / batch_size)
        total_batches_int = int(total_batches.numpy())

        with tqdm(total=total_batches_int, position=0, leave=True,
                  desc=f'Epoca {epoch+1}/{epochs}') as pbar:

            for noiseEEG_batch, EEG_batch in dataset:
                # Train step vectorizado (todo el batch en GPU)
                loss_batch, grads_batch = train_step_vectorized(
                    model, noiseEEG_batch, EEG_batch,
                    optimizer, denoise_network, datanum
                )

                # Acumular metricas
                epoch_loss += float(loss_batch)
                epoch_grads += float(grads_batch)
                num_batches += 1

                pbar.update(1)
                pbar.set_postfix({'loss': f'{float(loss_batch):.6f}'})

        # Promediar metricas
        avg_train_loss = epoch_loss / num_batches
        avg_grads = epoch_grads / num_batches

        # --- 3.5 Validacion ---
        val_loss_total = 0.0
        val_batches = 0

        for noiseEEG_val_batch, EEG_val_batch in val_dataset:
            _, val_loss_batch = test_step_vectorized(
                model, noiseEEG_val_batch, EEG_val_batch,
                denoise_network, datanum
            )
            val_loss_total += float(val_loss_batch)
            val_batches += 1

        avg_val_loss = val_loss_total / val_batches

        # --- 3.6 Guardar en TensorBoard ---
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_train_loss, step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)

        # --- 3.7 Guardar mejor modelo (ultimas 20% epocas) ---
        if epoch > epochs * 0.8 and avg_val_loss < val_mse_min:
            val_mse_min = avg_val_loss
            saved_model = model

            path = os.path.join(result_location, foldername, train_num, "denoise_model")
            tf.keras.models.save_model(model, path)
            print(f'\n  [GUARDADO] Mejor modelo: val_loss={avg_val_loss:.6f}')

        # --- 3.8 Reporte de epoca ---
        elapsed = time.time() - start_time
        print(f'Epoca {epoch+1}/{epochs} | Tiempo: {elapsed:.1f}s | '
              f'Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | '
              f'Grads: {avg_grads:.6f}')

        # Monitoreo de memoria GPU (util para GTX 1650 4GB)
        if tf.config.list_physical_devices('GPU'):
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            current_mb = mem_info.get('current', 0) / (1024**2)
            peak_mb = mem_info.get('peak', 0) / (1024**2)
            print(f'  [GPU VRAM] Actual: {current_mb:.0f}MB | Pico: {peak_mb:.0f}MB')

        # Guardar historial
        history['loss']['train_mse'].append(avg_train_loss)
        history['loss']['val_mse'].append(avg_val_loss)
        history['grads']['mse'].append(avg_grads)

    # --- 3.9 Guardar modelo final si no se guardo uno mejor ---
    if saved_model is None:
        saved_model = model
        path = os.path.join(result_location, foldername, train_num, "denoise_model")
        tf.keras.models.save_model(model, path)
        print('\n[OK] Modelo final guardado')

    return saved_model, history
