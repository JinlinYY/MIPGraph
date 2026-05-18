# MIPGraph — Ionic Liquid Property Prediction UI

Web-based prediction platform powered by **MIPGraph** (Mechanism-Factorized Ion-Pair Graph Learning Framework).  
Input cation–anion SMILES and operating conditions (T, P) to instantly obtain six thermophysical property predictions:

> Density · Electrical Conductivity · Heat Capacity · Surface Tension · Thermal Conductivity · Viscosity

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MIPGraph_UI.git
cd MIPGraph_UI
```

### 2. Install dependencies

```bash
pip install -r requirements_app.txt
```

> **PyTorch note:** The command above installs the CPU build of PyTorch.  
> For GPU / CUDA support, follow the official guide first:  
> https://pytorch.org/get-started/locally/

### 3. Download the pre-trained weights

Download the checkpoint archive from the link below and place it at:

```
outputs/checkpoints/finetune_viscosity_from_weak_seed42/best_model.pt
```

**Download:** [[ paste your link here — HuggingFace / Google Drive / Baidu Netdisk ]]

---

## Usage

### Windows — double-click

```
start.bat
```

### macOS / Linux

```bash
bash start.sh
```

### Manual

```bash
cd MIPGraph_UI
python scripts/serve_screening_ui.py
```

Then open your browser at **http://127.0.0.1:8765**

---

## Project Structure

```
MIPGraph_UI/
├── scripts/
│   └── serve_screening_ui.py   # FastAPI server (entry point)
├── static/
│   ├── index.html              # Web UI
│   └── assets/
│       ├── app.js
│       └── style.css
├── src/                        # Model source code
│   ├── chem/                   # SMILES parsing, graph featurization, 3D geometry
│   ├── data/                   # Dataset, scaler, splits
│   ├── models/                 # MIPGraph model architecture
│   ├── training/               # Loss, metrics, trainer
│   └── utils/                  # IO, logging, seeding
├── configs/
│   └── default.yaml            # Model configuration
├── outputs/
│   └── checkpoints/            # Pre-trained model weights (download separately)
├── ui_model_runtime.py         # Model loading & inference helper
├── requirements_app.txt        # Python dependencies
├── start.bat                   # One-click launch (Windows)
└── start.sh                    # One-click launch (macOS/Linux)
```

---

## System Requirements

| Item | Minimum |
|---|---|
| Python | 3.9+ |
| RAM | 4 GB |
| Disk | ~500 MB (model weights + dependencies) |
| GPU | Not required (CPU inference supported) |
