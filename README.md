# SIH 26131: Crop Disease and Pest Detection

Local AI backend for crop disease classification (MobileNetV3 Small) and pest detection (YOLO).

## Project structure

```
SIH_26131/
├── ai/
│   ├── models/
│   │   ├── plant_disease_mobilenetv3_28.pth
│   │   └── pest_best.pt
│   ├── configs/
│   │   ├── disease_class_names.json
│   │   ├── pest_class_names.json
│   │   ├── actions.json
│   │   └── crop_pests.json
│   ├── inference.py
│   └── requirements.txt
├── uploads/
├── main.py
├── test_api.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies from the project root.

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (Command Prompt):

```bat
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the trained weights in `ai/models/` before running inference:

- `ai/models/plant_disease_mobilenetv3_28.pth`
- `ai/models/pest_best.pt`

## Local pipeline test

From the project root, with the virtual environment activated:

```bash
python test_api.py
```

The script uses any `.jpg` / `.jpeg` / `.png` file already in `uploads/`. If none is present, it writes a dummy green image to `uploads/sample_test.jpg` and runs `analyze_crop_image()` on it.

## Launch the API

From the project root:

```bash
uvicorn main:app --reload --port 8000
```

Health check:

- `GET http://127.0.0.1:8000/health`
- Response: `{"status": "online", "model_device": "<cpu|cuda>"}`

## `POST /predict`

Upload a single image file as multipart form data. The field name must be `file`.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@uploads/sample_test.jpg"
```

JSON response:

```json
{
  "crop": "string",
  "disease": "string",
  "disease_confidence": 0.0,
  "pest_detected": "string or null",
  "pest_confidence": 0.0,
  "recommended_action": "string",
  "raw_pest_detections": []
}
```
