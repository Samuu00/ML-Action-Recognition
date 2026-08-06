# Real-Time Gesture Recognition Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Un'architettura per il **riconoscimento di gesture/movimenti in tempo reale** da flussi video RGB/webcam. 

Il sistema adotta un approccio **Two-Stage Architecture**, disaccoppiando l'estrazione visiva delle feature dalla classificazione sequenziale temporale per garantire un'inferenza ad altissima frequenza (**< 5ms** per frame su CPU).

---

## 🏗️ Architettura del Sistema

A differenza dei classici approcci pesanti basati su 3D-CNN o Video Transformers, il sistema separa nettamente le responsabilità:

```text
┌────────────────────────────────────────────────────────┐
│                   📹 Webcam Stream                     │
└───────────────────────────┬────────────────────────────┘
                            │ (Raw Video Frames)
                            ▼
┌────────────────────────────────────────────────────────┐
│            ⚡ Threaded Frame Capture (Async I/O)       │
└───────────────────────────┬────────────────────────────┘
                            │ (Async Queued Frame)
                            ▼
┌────────────────────────────────────────────────────────┐
│             🧘 MediaPipe Pose Estimation Engine         │
└───────────────────────────┬────────────────────────────┘
                            │ (x, y, z, visibility)
                            ▼
┌────────────────────────────────────────────────────────┐
│            📐 Spatial Normalizer (Scale & Shift)       │
└───────────────────────────┬────────────────────────────┘
                            │ (Normalized Features)
                            ▼
┌────────────────────────────────────────────────────────┐
│      🔲 Temporal Sliding Window Buffer (Ring Buffer)    │
└───────────────────────────┬────────────────────────────┘
                            │ (1, Window_Size, Features)
                            ▼
┌────────────────────────────────────────────────────────┐
│           🧠 ONNX Runtime Inference Engine             │
└───────────────────────────┬────────────────────────────┘
                            │ (Raw Logits / Probabilities)
                            ▼
┌────────────────────────────────────────────────────────┐
│        📈 Prediction Smoother (Exponential Moving Avg)  │
└───────────────────────────┬────────────────────────────┘
                            │ (Filtered Prediction)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 🎯 Output Event / Application          │
└────────────────────────────────────────────────────────┘
```

1. **Feature Extraction (Stage 1):** Utilizza **MediaPipe Pose** per convertire ogni frame in un vettore spaziale compatto (33 landmark 3D).
2. **Spatial Normalization:** Applica trasformazioni matematiche per garantire l'invarianza a scala, traslazione e posizione dell'utente rispetto alla fotocamera.
3. **Temporal Sliding Window:** Accumula $N$ frame consecutivi (default: 30 frame = ~1 sec) in un buffer ad anello con complessità $O(1)$.
4. **Classification & Smoothing (Stage 2):** Esegue un'inferenza ultrarapida con **ONNX Runtime** ed stabilizza l'output con un filtro *Exponential Moving Average* anti-flicker.

---

## 📂 Struttura del Progetto

Il progetto segue i principi della **Clean Architecture / Architettura Esagonale**, mantenendo la logica di dominio completamente isolata da framework hardware o ML.

```text
gesture_recognition/
├── config/
│   └── settings.yaml             # Iperparametri, soglie, window size e configurazione cam
├── data/
│   ├── raw/                      # Video grezzi organizzati per sottocartelle di classe
│   └── processed/                # Datasets di landmark estratti e normalizzati (.npz)
├── src/
│   ├── domain/                   # Domain Core (Pure Python - No OpenCV/PyTorch)
│   │   ├── entities.py           # Dataclass immutabili: Landmark, FrameData, GesturePrediction
│   │   ├── interfaces.py         # Protocols/ABC per dipendenze disaccoppiate
│   │   └── normalizer.py         # Logica matematica pura di normalizzazione spaziale
│   ├── infrastructure/           # Implementazioni concrete delle dipendenze
│   │   ├── camera.py             # Threaded OpenCV VideoCapture (Non-blocking I/O)
│   │   ├── extractors/           # Estrattore MediaPipe Pose
│   │   ├── classifiers/          # Engine di inferenza ONNX Runtime
│   │   └── pipeline/             # Sliding Window Buffer & Prediction Smoother
│   ├── application/              # Orchestrazione Use Cases
│   │   ├── dataset_builder.py    # Pipeline di estrazione landmark da dataset video
│   │   ├── trainer.py            # Training 1D-CNN in PyTorch ed export ONNX
│   │   └── real_time_runner.py   # Main Loop di inferenza live con visual overlay
│   └── utils/
│       └── logger.py             # Logger di sistema thread-safe
├── tests/
│   ├── unit/                     # Test unitari deterministici (normalizzazione, buffer)
│   └── integration/              # Test di integrazione sulla pipeline
├── pyproject.toml
├── requirements.txt
└── main.py                       # Entry point CLI


## 🚀 Quick Start

### 1. Requisiti e Installazione

Assicurati di avere Python **3.10+** installato.

```bash
# Clona il repository
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition

# Crea ed attiva un ambiente virtuale
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

### 2. Configurazione

Modifica il file `config/settings.yaml` per adattarlo al tuo hardware e alle tue esigenze:

```yaml
camera:
  device_id: 0
  width: 1280
  height: 720

pipeline:
  window_size: 30             # Frame per finestra (1 sec @ 30 FPS)
  prediction_threshold: 0.75   # Soglia di confidenza minima
```

### 3. Esecuzione Real-Time

Per avviare la webcam e il modello di inferenza live:

```bash
python main.py --mode run --config config/settings.yaml
```

---

## 🛠️ Workflow di Addestramento Custom

Per addestrare il sistema su nuove gesture personalizzate:

### Passaggio 1: Organizza i Video
Posiziona i tuoi video di addestramento nella cartella `data/raw/` strutturati per classe:

```text
data/raw/
├── wave_hand/
│   ├── video1.mp4
│   └── video2.mp4
├── swipe_left/
│   ├── video1.mp4
│   └── video2.mp4
└── no_gesture/
    └── video1.mp4
```

### Passaggio 2: Estrazione Feature
Esegui il builder per estrarre e normalizzare i landmark da tutti i video:

```bash
python main.py --mode build-dataset
```

### Passaggio 3: Addestramento ed Esportazione ONNX
Utilizza i file in `src/application/trainer.py` per addestrare la rete PyTorch `TemporalGestureCNN` ed esportarla direttamente nel formato `data/gesture_model.onnx`.

---

## 🧪 Testing

Esegui la suite di test unitari e d'integrazione:

```bash
# Esegui tutti i test
pytest

# Esegui con coverage report
pytest --cov=src tests/
```
