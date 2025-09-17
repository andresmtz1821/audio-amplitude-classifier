import time
import requests
import numpy as np
import threading
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from scipy import stats as scipy_stats

##########################################
############ Data properties #############
##########################################
sampling_rate = 20
window_time = 0.5 
window_samples = int(window_time * sampling_rate) 
label_dict = {
    1.0: 'sonido_amb', 2.0: 'conversacion', 3.0: 'susurro',
    4.0: 'grito', 5.0: 'music', 6.0: 'aplausos'
}

##########################################
##### Load data and train model here #####
##########################################

processed_data = np.loadtxt("audio_features.txt")
X_train_full = processed_data[:, 1:]
y_train_full = processed_data[:, 0]

scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_train_full)

k_svm_optimal = 10
feature_selector = SelectKBest(score_func=f_classif, k=k_svm_optimal)
X_scaled_selected_full = feature_selector.fit_transform(X_scaled_full, y_train_full)
C_svm_optimal = 20.0
gamma_svm_optimal = 0.15
final_model = SVC(kernel='rbf', C=C_svm_optimal, gamma=gamma_svm_optimal, random_state=42, probability=True)
final_model.fit(X_scaled_selected_full, y_train_full)

##########################################
##### Data acquisition configuration #####
##########################################
IP_ADDRESS = '192.168.1.94' # TU IP
COMMAND = 'calibration&dB=full&time=full' # TU COMMAND CONFIRMADO
BASE_URL = f"http://{IP_ADDRESS}/get?{COMMAND}" # El puerto ya está en IP_ADDRESS


max_samp_rate_phyphox = 100 
buffer_data = [] 
buffer_lock = threading.Lock()
stop_recording_event = threading.Event()
last_sample_time_global = 0.0

def fetch_data():
    global buffer_data, last_sample_time_global
    audio_buf_name_in_json = "dB"
    time_buf_name_in_json = "time"
    sleep_time = 1.0 / max_samp_rate_phyphox 

    while not stop_recording_event.is_set():
        try:
            response = requests.get(BASE_URL, timeout=1.0)
            response.raise_for_status()
            data = response.json()

            status_measuring = data.get("status", {}).get("measuring", None)
            if status_measuring is False:
                if time.time() % 5 < sleep_time * 2:
                    print("Phyphox status 'measuring: False'. Asegúrate PLAY en Phyphox.")
                time.sleep(0.5)
                continue

            time_buffer_from_phyphox = data["buffer"].get(time_buf_name_in_json, {}).get("buffer", [])
            db_buffer_from_phyphox = data["buffer"].get(audio_buf_name_in_json, {}).get("buffer", [])

            if time_buffer_from_phyphox and db_buffer_from_phyphox:
                current_time_val = time_buffer_from_phyphox[-1]
                audio_value_val = db_buffer_from_phyphox[-1]
                
                if audio_value_val is None:
                    if len(db_buffer_from_phyphox) > 1 and db_buffer_from_phyphox[-2] is not None:
                        audio_value_val = db_buffer_from_phyphox[-2]
                        current_time_val = time_buffer_from_phyphox[-2] if isinstance(time_buffer_from_phyphox, list) and len(time_buffer_from_phyphox) > 1 else current_time_val
                    else:
                        continue
                
                current_time_float = float(current_time_val)
                audio_value_float = float(audio_value_val)

                with buffer_lock:
                    if not buffer_data or buffer_data[-1][0] < current_time_float:
                        buffer_data.append((current_time_float, audio_value_float))
                        last_sample_time_global = current_time_float
                        
                        max_buffer_len = int(max_samp_rate_phyphox * (window_time * 4)) # Más historial
                        if len(buffer_data) > max_buffer_len:
                            buffer_data = buffer_data[-max_buffer_len:]
                        # print(f"ADD: T={current_time_float:.3f}, dB={audio_value_float:.2f}, N_b={len(buffer_data)}")
            else:
                 if time.time() % 5 < sleep_time*2: print(f"Buffers '{time_buf_name_in_json}' o '{audio_buf_name_in_json}' vacíos.")
        
        except requests.exceptions.Timeout:
            if time.time() % 5 < sleep_time*2: print(f"Timeout: {BASE_URL}")
        except requests.exceptions.RequestException:
            if time.time() % 5 < sleep_time*2: print(f"Error de Red/HTTP: {BASE_URL}")
        except KeyError as e:
            if time.time() % 5 < sleep_time*2: print(f"KeyError {e}. JSON: {data if 'data' in locals() else 'No data'}")
        except Exception as e:
            if time.time() % 5 < sleep_time*2: print(f"Error en fetch_data: {e}")
        time.sleep(sleep_time)

def stop_recording_thread():
    print("Deteniendo hilo de adquisición...")
    stop_recording_event.set()
    if recording_thread.is_alive():
        recording_thread.join(timeout=2)
    print("Hilo de adquisición detenido.")

def calculate_audio_features(signal_window):
    sig = signal_window; sig = sig[~np.isnan(sig)]; features_list = []
    if len(sig) > 0:
        features_list.append(np.mean(sig)); features_list.append(np.std(sig))
        features_list.append(scipy_stats.skew(sig)); features_list.append(scipy_stats.kurtosis(sig))
        features_list.append(np.median(sig)); features_list.append(np.min(sig)); features_list.append(np.max(sig))
        features_list.append(np.ptp(sig)); features_list.append(np.var(sig))
        features_list.append(np.percentile(sig, 10)); features_list.append(np.percentile(sig, 25))
        features_list.append(np.percentile(sig, 75)); features_list.append(np.percentile(sig, 90))
        features_list.append(np.percentile(sig, 75) - np.percentile(sig, 25))
        features_list.append(np.sum(sig**2)); features_list.append(np.mean(sig**2)); features_list.append(np.sum(np.abs(sig)))
        if len(sig) > 1:
            velocity = np.diff(sig); features_list.append(np.mean(velocity)); features_list.append(np.std(velocity))
            if len(velocity) > 1:
                acceleration = np.diff(velocity); features_list.append(np.mean(acceleration)); features_list.append(np.std(acceleration))
            else: features_list.extend([0, 0])
        else: features_list.extend([0, 0, 0, 0])
        mean_level = np.mean(sig)
        zero_crossings = np.sum(np.diff(np.signbit(sig - mean_level))) if len(sig) > 1 else 0
        features_list.append(zero_crossings / len(sig) if len(sig) > 0 else 0)
        features_list.append(scipy_stats.moment(sig, moment=3)); features_list.append(scipy_stats.moment(sig, moment=4))
        features_list.append(np.mean(np.abs(sig - np.median(sig)))); features_list.append(len(sig))
        std_sig_val = np.std(sig); features_list.append(np.max(sig) / std_sig_val if std_sig_val > 0 else 0)
        features_list.append(np.percentile(sig, 90) - np.percentile(sig, 10))
        sum_abs_sig = np.sum(np.abs(sig))
        if sum_abs_sig > 0:
            weighted_sum = np.sum(np.abs(sig) * np.arange(len(sig))); features_list.append(weighted_sum / sum_abs_sig)
        else: features_list.append(0)
        features_list.append(np.mean(np.abs(np.diff(sig))) if len(sig) > 1 else 0)
        rms_val = np.sqrt(np.sum(sig**2) / len(sig)) if len(sig) > 0 else 0
        features_list.append(rms_val)
    else: features_list.extend([0] * 31)
    if len(features_list) != 31:
        while len(features_list) < 31: features_list.append(0)
        features_list = features_list[:31]
    return np.array(features_list).reshape(1, -1)

recording_thread = threading.Thread(target=fetch_data, daemon=True)
recording_thread.start()

##########################################
######### Online classification ##########
##########################################
print("\nIniciando clasificación en línea...")
print("Presiona Ctrl+C para detener.")
prediction_count = 0
# Necesitamos suficientes puntos crudos para que la interpolación a window_samples sea estable
# window_samples es 10. Tomar al menos el doble o triple de puntos crudos.
min_raw_points_needed_for_interp = window_samples * 3 # ej. 30 puntos crudos

try:
    while True:
        time.sleep(0.5) 
        
        snapshot_buffer_data_main = []
        latest_time_in_snapshot_main = 0

        with buffer_lock:
            if len(buffer_data) > 0:
                snapshot_buffer_data_main = list(buffer_data)
                latest_time_in_snapshot_main = snapshot_buffer_data_main[-1][0]

        # Condición para procesar: suficiente tiempo global y suficientes puntos en el buffer
        if not snapshot_buffer_data_main or latest_time_in_snapshot_main < window_time or len(snapshot_buffer_data_main) < min_raw_points_needed_for_interp :
            if time.time() % 2 < 0.1: print(f"Esperando datos. Buffer: {len(snapshot_buffer_data_main)}, Últ.T: {latest_time_in_snapshot_main:.2f}s (necesita > {window_time:.1f}s y >{min_raw_points_needed_for_interp} pts)")
            continue
            
        snapshot_buffer_data_main.sort(key=lambda x: x[0]) # Ordenar por tiempo
        all_times = np.array([p[0] for p in snapshot_buffer_data_main])
        all_values = np.array([p[1] for p in snapshot_buffer_data_main])

        time_window_start_for_raw = latest_time_in_snapshot_main - window_time
        relevant_indices = all_times >= time_window_start_for_raw
        t_raw = all_times[relevant_indices]
        signal_raw = all_values[relevant_indices]

        if len(t_raw) < 2 or len(np.unique(t_raw)) < 2:
            if time.time() % 2 < 0.1: print(f"Puntos crudos para interpolar ({len(t_raw)}) insuficientes o tiempos no únicos.")
            continue
            
        t_uniform_start = latest_time_in_snapshot_main - window_time 
        t_uniform_end = latest_time_in_snapshot_main
        
        if (t_uniform_end - t_uniform_start) < (window_time * 0.9): # Asegurar que la ventana tenga casi toda la duración
             if time.time() % 2 < 0.1: print(f"Rango de tiempo efectivo para interpolar muy corto: {(t_uniform_end - t_uniform_start):.2f}s")
             continue

        t_uniform = np.linspace(t_uniform_start, t_uniform_end, window_samples, endpoint=False)
        if len(t_uniform) == 0: continue
        
        try:
            interp_func = interp1d(t_raw, signal_raw, kind='linear', fill_value="extrapolate", bounds_error=False)
            interpolated_audio_window = interp_func(t_uniform)
        except ValueError as e:
            if time.time() % 2 < 0.1: print(f"Error interpolación: {e}. t_raw({len(t_raw)}), unique t_raw({len(np.unique(t_raw))})")
            continue
        
        current_features = calculate_audio_features(interpolated_audio_window)
        
        try:
            features_scaled = scaler.transform(current_features)
            features_selected = feature_selector.transform(features_scaled)
            prediction_proba = final_model.predict_proba(features_selected)
            predicted_class_idx_in_proba = np.argmax(prediction_proba[0])
            predicted_class_id = final_model.classes_[predicted_class_idx_in_proba]
            predicted_activity = label_dict.get(float(predicted_class_id), f"ID {predicted_class_id}")
            confidence = prediction_proba[0][predicted_class_idx_in_proba]
            prediction_count += 1
            print(f"🔊 Predicción #{prediction_count}: {predicted_activity.upper()} (Conf: {confidence:.2f})")
            
            if prediction_count % 5 == 1:
                audio_std = np.std(interpolated_audio_window) if len(interpolated_audio_window) > 0 else 0
                audio_range = (np.max(interpolated_audio_window) - np.min(interpolated_audio_window)) if len(interpolated_audio_window) > 0 else 0
                print(f"Stats VentProc: std={audio_std:.3f}, rango={audio_range:.2f}, N={len(interpolated_audio_window)}, T_fin_vent={latest_time_in_snapshot_main:.2f}")
        except Exception as e:
            if time.time() % 2 < 0.1: print(f"Error en predicción/proc: {e}")

except KeyboardInterrupt:
    print("\nClasificación detenida.")
finally:
    stop_recording_thread()
    
