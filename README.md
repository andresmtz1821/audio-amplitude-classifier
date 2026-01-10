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
- **Interactive dashboard**: Streamlit web application for visualization and live demos

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

| Algorithm | Accuracy |
|-----------|----------|
| Gaussian Naive Bayes | 93% |
| Extra Trees | 93% |
| Gradient Boosting | 90% |
| SVM RBF (baseline) | 83% |
| SVM RBF (optimized) | 96.67% |

## Project Structure

```
audio-amplitude-classifier/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit web application
├── data/
│   ├── *.csv                       # Raw CSV files from PhyPhox
│   └── processed/
│       └── audio_features.txt      # Extracted features dataset
├── scripts/
│   ├── data_acquisition.py         # Data collection from PhyPhox
│   ├── data_processing.py          # Feature extraction
│   ├── communication_test.py       # PhyPhox connection testing
│   ├── data_plot.py                # Real-time visualization
│   ├── online_prototype.py         # Real-time classifier (inline model)
│   └── online_classification.py    # Real-time classifier (saved models)
├── models/
│   ├── final_svm_model_svm.pkl     # Trained SVM model
│   ├── standard_scaler_audio.pkl   # Feature scaler
│   └── audio_feature_selector_svm.pkl  # Feature selector
├── notebooks/
│   └── Project Development.ipynb   # Complete analysis and model development
└── docs/
    └── Report.pdf                  # Detailed project report
```

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- pandas
- scipy
- requests
- matplotlib
- streamlit
- plotly
- PhyPhox mobile app

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/audio-amplitude-classifier.git
cd audio-amplitude-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard (Recommended)

The easiest way to explore the project and test the classifier is through the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard includes:

- **Home**: Project overview, pipeline description, key metrics, and medical application context
- **Technical Analysis**: Model comparison charts, hyperparameter tuning results, feature importance analysis, and confusion matrix
- **Live Demo**: Real-time audio classification using your mobile device with PhyPhox

#### Live Demo Instructions:

1. Install PhyPhox on your mobile device ([Android](https://play.google.com/store/apps/details?id=de.rwth_aachen.phyphox) / [iOS](https://apps.apple.com/app/phyphox/id1127319693))
2. Open the **"Audio Amplitude"** experiment in PhyPhox
3. Enable **"Allow remote access"** in settings
4. Note the IP address displayed (e.g., `192.168.1.100`)
5. Press Play in PhyPhox to start recording
6. Enter the IP address in the dashboard and click "Start Classification"

### Command Line Classification

For direct command-line usage:

#### Option 1: Classifier with pre-trained models (Recommended)

```bash
python scripts/online_classification.py
```

Before running, edit the IP address in the script (line 46):
```python
IP_ADDRESS = '192.168.1.100'  # Replace with your PhyPhox IP
```

#### Option 2: Classifier with inline training

```bash
python scripts/online_prototype.py
```

This option trains the model at startup using `audio_features.txt`.

### Model Training

The complete model development process is documented in `notebooks/Project Development.ipynb`, including:

- Data loading and preprocessing
- Feature extraction pipeline
- Model comparison and selection
- Hyperparameter optimization with GridSearchCV
- Cross-validation results
- Model persistence

## Medical Applications

This system was developed with clinical applications in mind and can be applied to:

| Sound Class | Clinical Relevance |
|-------------|-------------------|
| Silence/Ambient | Baseline monitoring, equipment noise detection |
| Conversation | Patient communication assessment |
| Whisper | Low-energy speech detection in patients |
| Shout/Scream | Emergency detection, patient distress alerts |
| Music | Environmental audio identification |
| Applause | Crowd/group activity detection |

Potential use cases:

- **Emergency detection**: Identifying screams or calls for help in hospital settings
- **Patient monitoring**: Detecting changes in vocal patterns for early intervention
- **Environmental control**: Monitoring noise levels in medical facilities
- **Telemedicine**: Remote audio-based symptom analysis
- **Rehabilitation**: Tracking progress in speech therapy programs

## Real-time Performance

The system demonstrates strong performance for:

- **Silence/ambient**: Excellent classification accuracy
- **Music**: Very good detection (especially at high volume)
- **Whispers**: Perfect classification within specific amplitude range

Areas for improvement:

- Scream classification may confuse with music/applause at similar amplitudes
- Conversation requires consistent tone for best results
- Performance varies between different device microphones

## Troubleshooting

| Error | Solution |
|-------|----------|
| File not found .pkl | Run the complete notebook to generate model files |
| Connection timeout | Verify PhyPhox is in PLAY mode and IP is correct |
| Phyphox status measuring: False | Press PLAY in PhyPhox app |
| Erratic predictions | Ensure environment is similar to training conditions |

## Author

- **Andrés Martínez Almazán** (A01621042)

**Course**: Modeling Learning with Artificial Intelligence  
**Institution**: Tecnológico de Monterrey  
**Professor**: Dr. Omar Mendoza Montoya  
**Date**: June 2025

## License

This project was developed for academic purposes as part of a machine learning course at Tecnológico de Monterrey.
