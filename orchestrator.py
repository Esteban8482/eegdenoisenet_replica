"""
orchestrator.py - Framework experimental EEGdenoiseNet
Basado EXCLUSIVAMENTE en el codigo original del repositorio:
https://github.com/ncclabsustech/EEGdenoiseNet

Importa SOLO funciones del repositorio original (fuente de verdad):
  - Network_structure.py: fcNN, simple_CNN, Complex_CNN, RNN_lstm
  - data_prepare.py:      prepare_data
  - loss_function.py:     denoise_loss_mse
  - train_method.py:      train, test_step
  - save_method.py:       save_eeg

Funcionalidad adicional (metricas por SNR, graficas, JSON) se implementa
localmente en este archivo, sin modificar los modulos originales.

USO:
    python orchestrator.py
"""
import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'models'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'utils'))

def _safe_import(module_name, attrs, source_file):
    """Importa con verificacion de existencia."""
    try:
        mod = __import__(module_name, fromlist=attrs)
        result = []
        for attr in attrs:
            if not hasattr(mod, attr):
                raise ImportError(
                    f"'{attr}' no existe en {module_name}.py\n"
                    f"  Verifique que {source_file} contiene '{attr}'")
            result.append(getattr(mod, attr))
        return result[0] if len(result) == 1 else result
    except ImportError as e:
        print(f"\n[FATAL] Error importando {module_name}: {e}")
        sys.exit(1)


# --- models/Network_structure.py ---
(fcNN, simple_CNN, Complex_CNN, RNN_lstm) = _safe_import(
    'network_structure', ['fcNN', 'simple_CNN', 'Complex_CNN', 'RNN_lstm'],
    os.path.join(SCRIPT_DIR, 'models', 'network_structure.py'))

# --- utils/data_prepare.py ---
prepare_data = _safe_import(
    'data_prepare', ['prepare_data'],
    os.path.join(SCRIPT_DIR, 'utils', 'data_prepare.py'))

# --- utils/loss_function.py ---
denoise_loss_mse = _safe_import(
    'loss_function', ['denoise_loss_mse'],
    os.path.join(SCRIPT_DIR, 'utils', 'loss_function.py'))

# --- utils/train_method.py ---
(train, test_step) = _safe_import(
    'train_method', ['train', 'test_step'],
    os.path.join(SCRIPT_DIR, 'utils', 'train_method.py'))

# --- utils/save_method.py ---
save_eeg = _safe_import(
    'save_method', ['save_eeg'],
    os.path.join(SCRIPT_DIR, 'utils', 'save_method.py'))

print("[OK] Todos los modulos originales importados correctamente")

# ============================================================
# CONFIGURACION GLOBAL
# ============================================================

CONFIG = {
    'data_dir': os.path.join(SCRIPT_DIR, 'data'),
    'results_dir': os.path.join(SCRIPT_DIR, 'results'),
    'n_runs': 10,
    'batch_size': 40,
    'combin_num': 10,
    'train_split': 0.8,
    'optimizer': {
        'learning_rate': 5e-5,
        'beta_1': 0.5,
        'beta_2': 0.9,
    },
    'epochs': {
        'fcNN':        {'EOG': 60, 'EMG': 60},
        'Simple_CNN':  {'EOG': 40, 'EMG': 10},
        'Complex_CNN': {'EOG': 40, 'EMG': 10},
        'RNN_lstm':    {'EOG': 100, 'EMG': 60},
    },
    'snr_levels': {
        'EOG': [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
        'EMG': [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
    },
    'fs': {'EOG': 256, 'EMG': 512},
    'datanum': {'EOG': 512, 'EMG': 1024},
    'files': {'EEG': 'EEG_all_epochs.npy', 'EOG': 'EOG_all_epochs.npy', 'EMG': 'EMG_all_epochs.npy'},
}

MODEL_BUILDERS = {'fcNN': fcNN, 'Simple_CNN': simple_CNN, 'Complex_CNN': Complex_CNN, 'RNN_lstm': RNN_lstm}
MODEL_NAMES = ['fcNN', 'Simple_CNN', 'Complex_CNN', 'RNN_lstm']
NOISE_TYPES = ['EOG', 'EMG']

# ============================================================
# METRICAS DE EVALUACION (implementadas localmente,
# no existen en el repositorio original)
# ============================================================

def _rrmse_temporal(denoised, clean):
    rmse = np.sqrt(np.mean((denoised - clean) ** 2))
    rms_clean = np.sqrt(np.mean(clean ** 2))
    return rmse / rms_clean if rms_clean > 0 else 0.0

def _rrmse_spectral(denoised, clean, fs=256):
    _, psd_den = scipy_signal.welch(denoised, fs, nperseg=len(denoised))
    _, psd_clean = scipy_signal.welch(clean, fs, nperseg=len(clean))
    rmse = np.sqrt(np.mean((psd_den - psd_clean) ** 2))
    rms_clean = np.sqrt(np.mean(psd_clean ** 2))
    return rmse / rms_clean if rms_clean > 0 else 0.0

def _correlation_coeff(denoised, clean):
    cov = np.mean((denoised - np.mean(denoised)) * (clean - np.mean(clean)))
    v_d, v_c = np.var(denoised), np.var(clean)
    return cov / np.sqrt(v_d * v_c) if v_d > 0 and v_c > 0 else 0.0

def _compute_metrics(denoised, clean, fs=256):
    d, c = denoised.flatten(), clean.flatten()
    return {
        'rrmse_t': float(_rrmse_temporal(d, c)),
        'rrmse_s': float(_rrmse_spectral(d, c, fs)),
        'cc': float(_correlation_coeff(d, c))
    }

# ============================================================
# GRAFICAS (implementadas localmente)
# ============================================================

_COLORS = {'fcNN': '#1f77b4', 'Simple_CNN': '#ff7f0e',
           'Complex_CNN': '#2ca02c', 'RNN_lstm': '#d62728'}
_LABELS = {'fcNN': 'FCNN', 'Simple_CNN': 'Simple CNN',
           'Complex_CNN': 'Complex CNN', 'RNN_lstm': 'RNN'}

def _plot_convergence(all_histories, noise_type, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Convergencia ({noise_type})', fontsize=14)
    for idx, model in enumerate(MODEL_NAMES):
        ax = axes[idx // 2, idx % 2]
        c = _COLORS[model]
        max_ep = max(len(h['loss']['train_mse']) for h in all_histories[model])
        tr = np.full((10, max_ep), np.nan)
        vl = np.full((10, max_ep), np.nan)
        for r, h in enumerate(all_histories[model]):
            n = len(h['loss']['train_mse'])
            tr[r, :n] = h['loss']['train_mse']
            vl[r, :n] = h['loss']['val_mse']
        ep = np.arange(1, max_ep + 1)
        ax.plot(ep, np.nanmean(tr, axis=0), c, label='Train', lw=1.5)
        ax.plot(ep, np.nanmean(vl, axis=0), '--', c=c, label='Val', lw=1.5)
        ax.set_title(_LABELS[model], fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f'  [Plot] {out_path}')

def _plot_vs_snr(consolidated, metric_key, ylabel, noise_type, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_NAMES:
        snr_lvls = sorted(consolidated[model].keys())
        means = [np.mean(consolidated[model][s][metric_key]) for s in snr_lvls]
        stds = [np.std(consolidated[model][s][metric_key]) for s in snr_lvls]
        ax.plot(snr_lvls, means, 'o-', color=_COLORS[model], label=_LABELS[model], lw=2)
        ax.fill_between(snr_lvls, np.array(means)-stds, np.array(means)+stds,
                        color=_COLORS[model], alpha=0.15)
    ax.set_xlabel('SNR (dB)', fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f'  [Plot] {out_path}')

def _plot_boxplots(all_snr_metrics, noise_type, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'Benchmarks ({noise_type})', fontsize=14)
    for col, (key, title) in enumerate(zip(['rrmse_t', 'rrmse_s', 'cc'],
                                           ['RRMSE Temporal', 'RRMSE Spectral', 'CC'])):
        ax = axes[col]
        data, labels, colors = [], [], []
        for model in MODEL_NAMES:
            vals = []
            for run_m in all_snr_metrics[model]:
                for snr in run_m:
                    vals.extend(run_m[snr][key])
            data.append(vals); labels.append(_LABELS[model]); colors.append(_COLORS[model])
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='black', lw=1.5))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(title, fontweight='bold'); ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f'  [Plot] {out_path}')

# ============================================================
# ORQUESTACION
# ============================================================

def run_single(model_name, noise_type, run_idx, config, EEG_all, noise_all):
    """Ejecuta UNA repeticion usando EXACTAMENTE el flujo del main.py original."""
    datanum = config['datanum'][noise_type]
    epochs = config['epochs'][model_name][noise_type]
    combin_num = config['combin_num']
    bs = config['batch_size']
    result_loc = config['results_dir']
    folder = f'{noise_type}_{model_name}_benchmark'

    print(f'\n  >>> Run {run_idx}/10 | {model_name} + {noise_type} | {epochs} epochs')

    # 1. Preparar datos (exactamente como main.py original)
    (noise_tr, eeg_tr, noise_val, eeg_val,
     noise_ts, eeg_ts, test_std) = prepare_data(
        EEG_all, noise_all, combin_num=combin_num,
        train_per=config['train_split'], noise_type=noise_type)

    # 2. Construir modelo (exactamente como main.py original)
    model = MODEL_BUILDERS[model_name](datanum)

    # 3. Entrenar usando 'train' del original (firma exacta)
    opt = tf.optimizers.Adam(**config['optimizer'])
    saved_model, history = train(
        model, noise_tr, eeg_tr, noise_val, eeg_val,
        epochs, bs, opt, model_name, result_loc, folder, str(run_idx))

    # 4. Guardar senales usando 'save_eeg' del original (firma exacta)
    save_eeg(saved_model, result_loc, folder,
             False, False, True,
             noise_tr, eeg_tr, noise_val, eeg_val, noise_ts, eeg_ts,
             str(run_idx))

    # 5. Guardar historial (exactamente como main.py original, linea 103)
    hist_path = os.path.join(result_loc, folder, str(run_idx), 'loss_history.npy')
    np.save(hist_path, history)

    # 6. Evaluar por SNR (funcionalidad local, no en el repo original)
    snr_levels = config['snr_levels'][noise_type]
    n_snr = len(snr_levels)
    fs = config['fs'][noise_type]

    # Predecir todo el test set
    if model_name == 'fcNN':
        x_ts = noise_ts.reshape(-1, datanum)
        y_ts = eeg_ts.reshape(-1, datanum)
    else:
        x_ts = noise_ts.reshape(-1, datanum, 1)
        y_ts = eeg_ts.reshape(-1, datanum, 1)

    denoised = saved_model(x_ts, training=False).numpy()
    if model_name != 'fcNN':
        denoised = denoised.reshape(-1, datanum)
    y_ts = y_ts.reshape(-1, datanum)

    # Agrupar por nivel SNR (ciclico)
    snr_metrics = {}
    for i, snr in enumerate(snr_levels):
        idxs = np.arange(i, len(denoised), n_snr)
        m = {'rrmse_t': [], 'rrmse_s': [], 'cc': []}
        for idx in idxs:
            metrics = _compute_metrics(denoised[idx].flatten(), y_ts[idx].flatten(), fs)
            m['rrmse_t'].append(metrics['rrmse_t'])
            m['rrmse_s'].append(metrics['rrmse_s'])
            m['cc'].append(metrics['cc'])
        snr_metrics[snr] = m

    return history, snr_metrics


def orchestrate(noise_type, config):
    """Ejecuta el protocolo completo para un tipo de artefacto."""
    print(f'\n{"="*70}\n{"PROTOCOLO: " + noise_type:^70}\n{"="*70}')

    # Cargar datos (una sola vez)
    eeg_path = os.path.join(config['data_dir'], config['files']['EEG'])
    noise_path = os.path.join(config['data_dir'], config['files'][noise_type])
    if not os.path.exists(eeg_path) or not os.path.exists(noise_path):
        raise FileNotFoundError(f'Dataset no encontrado. Coloque .npy en {config["data_dir"]}/')
    EEG_all = np.load(eeg_path)
    noise_all = np.load(noise_path)
    print(f'[Data] EEG:{EEG_all.shape} {noise_type}:{noise_all.shape}')

    res_dir = os.path.join(config['results_dir'],
                           f'benchmark_{noise_type}_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(res_dir, exist_ok=True)

    all_hist = {m: [] for m in MODEL_NAMES}
    all_snr = {m: [] for m in MODEL_NAMES}

    for model in MODEL_NAMES:
        print(f'\n--- {model} ({config["epochs"][model][noise_type]} epochs) ---')
        for r in range(1, config['n_runs'] + 1):
            t0 = time.time()
            try:
                hist, snr_m = run_single(model, noise_type, r, config, EEG_all, noise_all)
                all_hist[model].append(hist)
                all_snr[model].append(snr_m)
                print(f'  [OK] Run {r} en {time.time()-t0:.1f}s')
            except Exception as e:
                print(f'  [ERROR] Run {r}: {e}')
                import traceback; traceback.print_exc()

    # Graficas
    print('\n[Plots] Generando figuras...')
    _plot_convergence(all_hist, noise_type, os.path.join(res_dir, f'fig07_convergence_{noise_type}.png'))

    # Consolidar por SNR
    cons = {m: {} for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        for snr in config['snr_levels'][noise_type]:
            cons[m][snr] = {'rrmse_t': [], 'rrmse_s': [], 'cc': []}
            for rm in all_snr[m]:
                for k in ['rrmse_t', 'rrmse_s', 'cc']:
                    cons[m][snr][k].extend(rm[snr][k])

    _plot_vs_snr(cons, 'rrmse_t', 'RRMSE Temporal', noise_type,
                 os.path.join(res_dir, f'fig08a_rrmse_t_{noise_type}.png'))
    _plot_vs_snr(cons, 'rrmse_s', 'RRMSE Spectral', noise_type,
                 os.path.join(res_dir, f'fig08b_rrmse_s_{noise_type}.png'))
    _plot_vs_snr(cons, 'cc', 'Correlation Coefficient', noise_type,
                 os.path.join(res_dir, f'fig08c_cc_{noise_type}.png'))
    _plot_boxplots(all_snr, noise_type, os.path.join(res_dir, f'fig09_boxplots_{noise_type}.png'))

    # JSON
    final = {}
    for m in MODEL_NAMES:
        final[m] = {}
        for snr in config['snr_levels'][noise_type]:
            final[m][snr] = {
                'rrmse_t_mean': float(np.mean(cons[m][snr]['rrmse_t'])),
                'rrmse_t_std': float(np.std(cons[m][snr]['rrmse_t'])),
                'rrmse_s_mean': float(np.mean(cons[m][snr]['rrmse_s'])),
                'rrmse_s_std': float(np.std(cons[m][snr]['rrmse_s'])),
                'cc_mean': float(np.mean(cons[m][snr]['cc'])),
                'cc_std': float(np.std(cons[m][snr]['cc'])),
            }
    with open(os.path.join(res_dir, f'metrics_{noise_type}.json'), 'w') as f:
        json.dump(final, f, indent=2)

    # Tabla consola
    print(f'\n{"-"*70}\n{"RESULTADOS: " + noise_type:^70}\n{"-"*70}')
    print(f'{"Modelo":<15} {"RRMSE_t":>18} {"RRMSE_s":>18} {"CC":>15}')
    print('-'*70)
    for m in MODEL_NAMES:
        r, s, c = [], [], []
        for snr in config['snr_levels'][noise_type]:
            r.extend(cons[m][snr]['rrmse_t'])
            s.extend(cons[m][snr]['rrmse_s'])
            c.extend(cons[m][snr]['cc'])
        print(f'{m:<15} {np.mean(r):.4f}+/-{np.std(r):.4f}   '
              f'{np.mean(s):.4f}+/-{np.std(s):.4f}   {np.mean(c):.4f}+/-{np.std(c):.4f}')
    print(f'\n[OK] Resultados en: {res_dir}')
    return res_dir


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'[INFO] GPU: {len(gpus)} | TF: {tf.__version__}')

    t0 = time.time()
    for nt in NOISE_TYPES:
        orchestrate(nt, CONFIG)
    print(f'\n[Done] Tiempo total: {(time.time()-t0)/3600:.1f}h')


if __name__ == '__main__':
    main()