 
from pathlib import Path

DATA_DIR = Path("data/chest_xray")
classes = ["NORMAL", "PNEUMONIA"]
splits = ["train", "val", "test"]

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Check that data/chest_xray exists."
    )

print("CHEST X-RAY DATASET SUMMARY\n")

for split in splits:
    print(f"{split.upper()}")

    for class_name in classes:
        folder = DATA_DIR / split / class_name

        if not folder.exists():
            print(f"  {class_name}: Folder missing")
            continue

        images = [
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        print(f"  {class_name}: {len(images)} images")

    print()
