import os
import shutil
import hashlib
import argparse
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import kagglehub
except ImportError:
    print("Error: 'kagglehub' is required. Please install it with 'pip install kagglehub'")
    exit(1)

def get_file_hash(filepath):
    """Generate MD5 hash of an image for duplicate detection."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def determine_class(filepath):
    """
    Identify class (TB or Normal) based on the file's path or name.
    """
    path_lower = filepath.lower()
    
    # Normal indicators
    normal_keywords = ['normal', 'healthy', 'control', 'negative']
    if any(keyword in path_lower for keyword in normal_keywords):
        return 'Normal'
    
    # TB indicators
    tb_keywords = ['tuberculosis', 'positive']
    if re.search(r'\btb\b', path_lower) or any(keyword in path_lower for keyword in tb_keywords):
        return 'TB'
        
    return 'Unknown'

def copy_files(file_list, dest_dir, split_name, class_name):
    """Helper function to safely copy files into the structured dataset directory."""
    dest_subdir = os.path.join(dest_dir, split_name, class_name)
    os.makedirs(dest_subdir, exist_ok=True)
    
    for img_path in tqdm(file_list, desc=f"Copying {class_name} ({split_name})", unit="file", leave=False):
        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_subdir, filename)
        
        # Prevent filename collisions
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            img_hash_snippet = get_file_hash(img_path)[:6]
            dest_path = os.path.join(dest_subdir, f"{name}_{img_hash_snippet}{ext}")
        
        try:
            shutil.copy2(img_path, dest_path)
        except Exception as e:
            print(f"Error copying {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Auto TB Dataset Downloader and Setup")
    parser.add_argument('--dest', type=str, default='dataset', help="Destination directory for the final organized dataset")
    parser.add_argument('--skip_download', action='store_true', help="Skip kagglehub download step")
    args = parser.parse_args()

    dest_dir = os.path.abspath(args.dest)
    raw_dir = ""

    # 1. Download dataset using kagglehub
    if not args.skip_download:
        print("\n--- Phase 1: Downloading Datasets ---")
        print("Using KaggleHub to fetch the Tuberculosis Chest X-ray Dataset...")
        try:
            # This dataset contains thousands of TB and Normal images
            path = kagglehub.dataset_download('tawsifurrahman/tuberculosis-tb-chest-xray-dataset')
            print(f"Dataset successfully downloaded/located at: {path}")
            raw_dir = path
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please ensure you have internet access.")
            return
    else:
        # If skip download, we assume the dataset was already downloaded to kagglehub's cache or a custom path
        print("Skipping download as requested. Please provide raw dir if you want to use a specific path.")
        # Try to find it in the cache
        raw_dir = kagglehub.dataset_download('tawsifurrahman/tuberculosis-tb-chest-xray-dataset')
        print(f"Using cached dataset at: {raw_dir}")

    # 2. Scan for valid images
    print("\n--- Phase 2: Scanning and Processing ---")
    valid_exts = {'.png', '.jpg', '.jpeg'}
    all_images = []
    
    print(f"Scanning '{raw_dir}' for images...")
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                all_images.append(os.path.join(root, file))

    total_images_found = len(all_images)
    print(f"Total valid images found in raw_data: {total_images_found}")

    if total_images_found == 0:
        print("No images found. Something went wrong with the download.")
        return

    # 3. Detect and remove duplicate images using hashing
    print("\nDetecting duplicates using hashing...")
    unique_images = []
    seen_hashes = set()
    duplicates_removed = 0

    for img_path in tqdm(all_images, desc="Hashing images", unit="file"):
        img_hash = get_file_hash(img_path)
        if img_hash is None:
            continue
            
        if img_hash in seen_hashes:
            duplicates_removed += 1
        else:
            seen_hashes.add(img_hash)
            unique_images.append(img_path)

    print(f"Duplicates automatically removed: {duplicates_removed}")
    print(f"Unique images available: {len(unique_images)}")

    # 4. Classify images into TB, Normal
    print("\nClassifying images based on folder/file names...")
    tb_images = []
    normal_images = []
    unknown_images = []

    for img_path in unique_images:
        cls = determine_class(img_path)
        if cls == 'TB':
            tb_images.append(img_path)
        elif cls == 'Normal':
            normal_images.append(img_path)
        else:
            unknown_images.append(img_path)

    print(f"Total classified TB     : {len(tb_images)}")
    print(f"Total classified Normal : {len(normal_images)}")

    if not tb_images and not normal_images:
        print("Error: Could not classify any images. Directory structure or filenames are not recognized.")
        return

    # 5. Split the dataset (70% Train, 15% Validation, 15% Test)
    def split_data(images_list):
        if len(images_list) < 3:
            return images_list, [], []
            
        train_data, temp_data = train_test_split(images_list, test_size=0.30, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)
        
        return train_data, val_data, test_data

    tb_train, tb_val, tb_test = split_data(tb_images)
    normal_train, normal_val, normal_test = split_data(normal_images)

    # 6. Create final dataset structure
    splits = ['train', 'val', 'test']
    classes = ['TB', 'Normal']
    
    print(f"\n--- Phase 3: Creating Dataset Hierarchy at '{dest_dir}' ---")
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    # Copy files
    copy_files(tb_train, dest_dir, 'train', 'TB')
    copy_files(tb_val, dest_dir, 'val', 'TB')
    copy_files(tb_test, dest_dir, 'test', 'TB')

    copy_files(normal_train, dest_dir, 'train', 'Normal')
    copy_files(normal_val, dest_dir, 'val', 'Normal')
    copy_files(normal_test, dest_dir, 'test', 'Normal')

    # 7. Print final report
    print("\n" + "="*50)
    print(" " * 12 + "FINAL DATASET REPORT")
    print("="*50)
    print(f"Total Images Downloaded (Raw) : {total_images_found}")
    print(f"Duplicates Removed            : {duplicates_removed}")
    print(f"Total Unique Images Processed : {len(unique_images)}")
    print(f"Unclassified/Ignored          : {len(unknown_images)}")
    print("-" * 50)
    print(f"Total TB Images               : {len(tb_images)}")
    print(f"Total Normal Images           : {len(normal_images)}")
    print("="*50)
    print(f"Train split (70%)             : TB={len(tb_train):<5} | Normal={len(normal_train):<5}")
    print(f"Validation split (15%)        : TB={len(tb_val):<5} | Normal={len(normal_val):<5}")
    print(f"Test split (15%)              : TB={len(tb_test):<5} | Normal={len(normal_test):<5}")
    print("="*50)
    print(f"Success: Dataset is ready in the '{dest_dir}/' folder.")
    print("You can now proceed to train your TB detection model!")

if __name__ == "__main__":
    main()
