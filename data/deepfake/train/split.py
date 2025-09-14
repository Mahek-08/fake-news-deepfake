# split_deepfake_dataset.py
import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
base_dir = "data/deepfake"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Create val folders
os.makedirs(os.path.join(val_dir, "real"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "fake"), exist_ok=True)

def split_and_copy(label, test_size=0.2):
    files = os.listdir(os.path.join(train_dir, label))
    train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)
    
    # Move validation files
    for f in val_files:
        src = os.path.join(train_dir, label, f)
        dst = os.path.join(val_dir, label, f)
        shutil.move(src, dst)
    
    print(f"✅ {label}: {len(train_files)} train, {len(val_files)} val")

if __name__ == "__main__":
    split_and_copy("real", test_size=0.2)
    split_and_copy("fake", test_size=0.2)
    print("Dataset split completed.")