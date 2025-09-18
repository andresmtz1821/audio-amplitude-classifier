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
│   ├── data\_acquisition.py     # Data collection from PhyPhox
│   ├── data\_processing.py      # Feature extraction
│   ├── communication\_test.py   # PhyPhox connection testing
│   ├── data\_plot.py            # Real-time visualization
│   ├── online\_prototype.py     # Real-time classifier (inline model)
│   └── online\_classification.py # Real-time classifier (saved models)
├── models/
│   ├── final\_svm\_model\_svm.pkl        # Trained SVM model
│   ├── standard\_scaler\_audio.pkl      # Feature scaler
│   └── audio\_feature\_selector\_svm.pkl # Feature selector
├── notebooks/
│   └── Project.ipynb          # Complete analysis and model development
└── docs/
└── Report.pdf             # Detailed project report

````

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

To run the real-time audio classifier, follow these steps:

#### Step 1: PhyPhox Setup
1. **Install PhyPhox**: Download and install the PhyPhox app on your mobile device:
   - [Android](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox)
   - [iOS](https://apps.apple.com/app/phyphox/id1127319693)

2. **Configure experiment**:
   - Open PhyPhox and search for the experiment **"Audio Amplitude"**
   - Enable the option **"Allow remote access"** in settings
   - Write down the IP address shown on the screen (e.g., `192.168.1.100`)

#### Step 2: Environment Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
````

2. **Generate trained models** (if they don’t exist):

   ```bash
   cd notebooks/
   jupyter notebook Project.ipynb
   # Run all cells up to the "Online Classification" section
   ```

#### Step 3: IP Configuration

1. **Edit classification script**:

   ```bash
   nano scripts/online_classification.py
   ```

2. **Update the IP** on line 46:

   ```python
   IP_ADDRESS = '192.168.1.100'  # Replace with your PhyPhox IP
   ```

#### Step 4: Run Real-time Classification

1. **Ensure models are in the correct folder**:

   ```bash
   ls models/
   # Must show: final_svm_model_svm.pkl, standard_scaler_audio.pkl, audio_feature_selector_svm.pkl
   ```

2. **Start PhyPhox**: Press ▶️ PLAY in the "Audio Amplitude" experiment

3. **Run classifier**:

   ```bash
   cd scripts/
   python online_classification.py
   ```

#### Available classification options:

**Option 1: Classifier with pre-trained models** (Recommended)

```bash
python scripts/online_classification.py
```

* Loads models from `.pkl` files
* Faster startup
* Requires running the notebook beforehand

**Option 2: Classifier with inline training**

```bash
python scripts/online_prototype.py
```

* Trains the model at startup using `audio_features.txt`
* Slower startup but doesn’t require `.pkl` files

#### Expected results:

```
🔊 Prediction #1: CONVERSATION (Conf: 0.87)
🔊 Prediction #2: MUSIC (Conf: 0.92)
🔊 Prediction #3: SILENCE_AMB (Conf: 0.95)
```

#### Troubleshooting:

* **Error "File not found .pkl"**: Run the complete notebook first
* **Error "Timeout"**: Check that PhyPhox is in PLAY mode and IP is correct
* **Error "Phyphox status measuring: False"**: Press PLAY in PhyPhox
* **Erratic predictions**: Ensure you’re in an environment similar to training

### Model Training

The complete model development process is available in `notebooks/Project.ipynb`, including:

* Data loading and preprocessing
* Feature extraction
* Model comparison and selection
* Hyperparameter optimization
* Cross-validation results

## Medical Applications

This system can be applied to:

* **Emergency detection**: Identifying screams or calls for help
* **Patient monitoring**: Detecting changes in vocal patterns
* **Environmental control**: Monitoring noise levels in medical facilities
* **Telemedicine**: Remote audio-based symptom analysis
* **Rehabilitation**: Tracking progress in speech therapy

## Real-time Performance

The system shows excellent performance for:

* Silence/ambient: Perfect classification
* Music: Very good (high volume)
* Whispers: Perfect (within specific amplitude range)

Areas for improvement:

* Scream classification: Tends to confuse with music/applause
* Conversation: Requires consistent tone
* Device dependency: Performance varies between different microphones

## Authors

* Andrés Martínez Almazán (A01621042)
* Diego Arechiga Bonilla (A01621045)

**Course**: Modeling Learning with Artificial Intelligence
**Institution**: Tecnológico de Monterrey
**Professor**: Dr. Omar Mendoza Montoya
**Date**: June 2025

## License

This project was developed for academic purposes as part of a machine learning course.

```

¿Quieres que además te genere el `requirements.txt` en otra celda con las dependencias exactas para que tu repo quede listo para correr?
```
