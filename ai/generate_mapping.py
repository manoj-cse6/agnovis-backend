import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from torchvision import transforms
from PIL import Image
MODEL_PATH = BASE_DIR / "ai" / "models" / "plant_disease_mobilenetv3_28.pth"
CONFIG_PATH = BASE_DIR / "ai" / "configs" / "disease_class_names.json"

# Candidate sources for class mapping
DRIVE_CONFIG = Path(r"c:\Users\Manoj Kumar Agurla\My Drive\SIH_26131\03_Configs\disease_class_names.json")
DRIVE_TEST_DIR = Path(r"c:\Users\Manoj Kumar Agurla\My Drive\SIH_26131\04_Test_Images")

def get_class_names():
    if DRIVE_CONFIG.exists():
        with open(DRIVE_CONFIG, "r", encoding="utf-8") as f:
            classes = json.load(f)
            return classes
    return None

def main():
    classes = get_class_names()
    print(f"Loaded {len(classes)} classes from training artifact:")
    for i, name in enumerate(classes):
        print(f"  {i}: {name}")

    mapping_dict = {str(i): name for i, name in enumerate(classes)}
    
    # Save to ai/configs/disease_class_names.json
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, indent=2)
    print(f"\nSuccessfully wrote aligned mapping to {CONFIG_PATH}")

    # Verify with model and test images if available
    if DRIVE_TEST_DIR.exists() and MODEL_PATH.exists():
        from ai.inference import _build_mobilenetv3_small
        model = _build_mobilenetv3_small(28)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_images = sorted([
            p for p in DRIVE_TEST_DIR.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        print(f"\nEvaluating on {len(test_images)} test images from {DRIVE_TEST_DIR}:")
        correct = 0
        for img_path in test_images:
            actual = re.sub(r"__\d+$", "", img_path.stem)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                conf, idx = torch.max(probs, 0)
            
            predicted = mapping_dict[str(idx.item())]
            is_match = (predicted == actual)
            if is_match:
                correct += 1
            print(f"[{'PASS' if is_match else 'FAIL'}] Actual: {actual:<45} | Pred ({idx.item():2d}): {predicted:<45} | Conf: {conf.item()*100:.2f}%")
        
        print(f"\nAccuracy: {correct}/{len(test_images)} ({correct/len(test_images)*100:.1f}%)")

    # Audit actions and remedies for all 28 classes
    from ai.inference import _parse_disease_label, _recommended_action, _lookup_recommendations, load_models
    load_models()
    print("\n" + "="*80)
    print("ALL 28 CLASSES ACTIONS & REMEDIES AUDIT:")
    print("="*80)
    for idx_str, label in sorted(mapping_dict.items(), key=lambda x: int(x[0])):
        crop, disease = _parse_disease_label(label)
        act = _recommended_action(label, crop, disease, None)
        recs = _lookup_recommendations(label, disease, None)
        sev = recs["disease"]["severity_level"]
        print(f"[{int(idx_str):2d}] {label:<45} -> Crop: {crop:<12} | Disease: {disease:<30} | Sev: {sev:<6}")

if __name__ == "__main__":
    main()
