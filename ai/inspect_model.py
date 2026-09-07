import json
import os
import torch
import sys

BASE_DIR = r"c:\Users\Manoj Kumar Agurla\Desktop\SIH_26131"
MODEL_PATH = os.path.join(BASE_DIR, "ai", "models", "plant_disease_mobilenetv3_28.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "ai", "configs", "disease_class_names.json")

print("--- DIAGNOSTIC AUDIT ---")
if os.path.exists(MODEL_PATH):
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except Exception as e:
        print("Failed to load weights_only=False, trying without:", e)
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
    print("Checkpoint Type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Keys in Checkpoint:", list(ckpt.keys()))
        if "class_to_idx" in ckpt:
            print("Found class_to_idx in checkpoint!")
            print(ckpt["class_to_idx"])
        if "classes" in ckpt:
            print("Found classes in checkpoint!")
            print(ckpt["classes"])
        if "idx_to_class" in ckpt:
            print("Found idx_to_class in checkpoint!")
            print(ckpt["idx_to_class"])
            
        state_dict = ckpt
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                print(f"Using {k} for state_dict")
                break
                
        for k in list(state_dict.keys()):
            if "classifier" in k.lower():
                print(f"Shape of {k}: {state_dict[k].shape}")
else:
    print(f"Model path not found: {MODEL_PATH}")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)
    print("Total Classes in disease_class_names.json:", len(classes))
    if isinstance(classes, list):
        for i, k in enumerate(classes):
            print(f"  {i}: {k}")
    elif isinstance(classes, dict):
        for k in sorted(classes.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            print(f"  {k}: {classes[k]}")
else:
    print(f"Config path not found: {CONFIG_PATH}")
