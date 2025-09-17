# Real-Time Audio Classifier for Medical Applications

A machine learning system that classifies environmental sounds in real-time using audio amplitude data collected from mobile devices via PhyPhox. The system is designed for medical and healthcare monitoring applications.

## Overview

This project implements a Support Vector Machine (SVM) classifier that can identify six different types of environmental sounds:

- Silence/ambient noise
- Normal conversation
- Whisper
- Shout/scream
- Music
- Applause

The system processes audio data in real-time using 0.5-second windows and extracts 31 acoustic features to make predictions with confidence scores.

## Key Features

- **Real-time classification**: Processes audio data continuously with 0.5-second windows
- **Medical applications**: Designed for healthcare monitoring, emergency detection, and patient care
- **Mobile integration**: Uses PhyPhox mobile app for data collection
- **High accuracy**: Achieves 96.67% accuracy on validation data
- **Feature engineering**: Extracts 31 statistical and temporal features from audio signals

## Dataset

- **Collection method**: PhyPhox mobile application (sound pressure level sensor)
- **Total samples**: 30 recordings (5 per category)
- **Duration**: 40-60 seconds per recording
- **Features**: 31 extracted features including statistical measures, energy metrics, and temporal derivatives

## Model Performance

The final SVM model with RBF kernel achieved:
- **Validation accuracy**: 96.67%
- **Optimal parameters**: C=20.0, gamma=0.15
- **Feature selection**: SelectKBest with k=10 features

### Comparison with other algorithms:
- Gaussian Naive Bayes: 93%
- Extra Trees: 93%
- Gradient Boosting: 90%
- SVM RBF: 83% (96.67% after optimization)

## Project Structure

```
├── README.md
├── data/
│   ├── raw/                    # Original CSV files from PhyPhox
│   └── processed/              # Processed feature files
├── scripts/
│   ├── data_acquisition.py     # Data collection from PhyPhox
│   ├── data_processing.py      # Feature extraction
│   ├── communication_test.py   # PhyPhox connection testing
│   ├── data_plot.py           # Real-time visualization
│   ├── online_prototype.py    # Real-time classifier (inline model)
│   └── online_classification.py # Real-time classifier (saved models)
├── models/
│   ├── final_svm_model_svm.pkl        # Trained SVM model
│   ├── standard_scaler_audio.pkl      # Feature scaler
│   └── audio_feature_selector_svm.pkl # Feature selector
├── notebooks/
│   └── Project.ipynb          # Complete analysis and model development
└── docs/
    └── Report.pdf            # Detailed project report
```

## Requirements

- Python 3.x
- scikit-learn
- numpy
- pandas
- scipy
- requests
- matplotlib
- PhyPhox mobile app

## Usage

### Real-time Classification

Para ejecutar el clasificador de audio en tiempo real, sigue estos pasos:

#### Paso 1: Configuración de PhyPhox
1. **Instalar PhyPhox**: Descarga e instala la aplicación PhyPhox en tu dispositivo móvil:
   - [Android](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox)
   - [iOS](https://apps.apple.com/app/phyphox/id1127319693)

2. **Configurar experimento**:
   - Abre PhyPhox y busca el experimento **"Audio Amplitude"**
   - Activa la opción **"Allow remote access"** en la configuración
   - Anota la dirección IP que aparece en pantalla (ej. `192.168.1.100`)

#### Paso 2: Configuración del entorno
1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generar modelos entrenados** (si no existen):
   ```bash
   cd notebooks/
   jupyter notebook Proyecto.ipynb
   # Ejecuta todas las celdas hasta la sección "Clasificación en línea"
   ```

#### Paso 3: Configuración de IP
1. **Editar script de clasificación**:
   ```bash
   nano scripts/online_classification.py
   ```

2. **Actualizar la IP** en la línea 46:
   ```python
   IP_ADDRESS = '192.168.1.100'  # Cambia por tu IP de PhyPhox
   ```

#### Paso 4: Ejecutar clasificación en tiempo real
1. **Asegurar que los modelos estén en la carpeta correcta**:
   ```bash
   ls models/
   # Debe mostrar: final_svm_model_svm.pkl, standard_scaler_audio.pkl, audio_feature_selector_svm.pkl
   ```

2. **Iniciar PhyPhox**: Presiona ▶️ PLAY en el experimento "Audio Amplitude"

3. **Ejecutar clasificador**:
   ```bash
   cd scripts/
   python online_classification.py
   ```

#### Opciones de clasificación disponibles:

**Opción 1: Clasificador con modelos pre-entrenados** (Recomendado)
```bash
python scripts/online_classification.py
```
- Carga modelos desde archivos `.pkl`
- Mayor velocidad de inicio
- Requiere haber ejecutado el notebook previamente

**Opción 2: Clasificador con entrenamiento inline**
```bash
python scripts/online_prototype.py
```
- Entrena el modelo al inicio usando `audio_features.txt`
- Tarda más en iniciar pero no requiere archivos `.pkl`

#### Resultados esperados:
```
🔊 Predicción #1: CONVERSACION (Conf: 0.87)
🔊 Predicción #2: MUSICA (Conf: 0.92)
🔊 Predicción #3: SILENCIO_AMB (Conf: 0.95)
```

#### Troubleshooting:
- **Error "Archivo no encontrado .pkl"**: Ejecuta el notebook completo primero
- **Error "Timeout"**: Verifica que PhyPhox esté en PLAY y la IP sea correcta
- **Error "Phyphox status measuring: False"**: Presiona PLAY en PhyPhox
- **Predicciones erráticas**: Asegúrate de estar en un ambiente similar al entrenamiento

### Model Training

The complete model development process is available in `notebooks/Proyecto.ipynb`, including:
- Data loading and preprocessing
- Feature extraction
- Model comparison and selection
- Hyperparameter optimization
- Cross-validation results

## Medical Applications

This system can be applied to:
- **Emergency detection**: Identifying screams or calls for help
- **Patient monitoring**: Detecting changes in vocal patterns
- **Environmental control**: Monitoring noise levels in medical facilities
- **Telemedicine**: Remote audio-based symptom analysis
- **Rehabilitation**: Tracking progress in speech therapy

## Real-time Performance

The system shows excellent performance for:
- Silence/ambient: Perfect classification
- Music: Very good (high volume)
- Whispers: Perfect (within specific amplitude range)

Areas for improvement:
- Scream classification: Tends to confuse with music/applause
- Conversation: Requires consistent tone
- Device dependency: Performance varies between different microphones

## Authors

- Andrés Martínez Almazán (A01621042)
- Diego Arechiga Bonilla (A01621045)

**Course**: Modeling Learning with Artificial Intelligence
**Institution**: Tecnológico de Monterrey
**Professor**: Dr. Omar Mendoza Montoya
**Date**: June 2025

## License

This project was developed for academic purposes as part of a machine learning course.