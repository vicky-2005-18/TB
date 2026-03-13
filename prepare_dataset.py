import os
import shutil
import hashlib
import argparse
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_file_hash(filepath):
    """
    Generate MD5 hash of an image for duplicate detection.
    Reads the file in chunks to ensure memory efficiency.
    """
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return None

def determine_class(filepath):
    """
    Identify class (TB or Normal) based on the file's path or name.
    """
    path_lower = filepath.lower()
    
    # Check for Normal class
    if 'normal' in path_lower:
        return 'Normal'
    
    # Check for TB class using regex for word boundary to avoid false positives 
    # or explicit 'tuberculosis'
    if re.search(r'\btb\b', path_lower) or 'tuberculosis' in path_lower:
        return 'TB'
        
    return 'Unknown'

def copy_files(file_list, dest_dir, split_name, class_name):
    """
    Helper function to safely copy files into the structured dataset directory.
    """
    dest_subdir = os.path.join(dest_dir, split_name, class_name)
    
    for img_path in tqdm(file_list, desc=f"Copying {class_name} ({split_name})", unit="file", leave=False):
        filename = os.path.basename(img_path)
        dest_path = os.path.join(dest_subdir, filename)
        
        # Prevent filename collisions in case identically named files exist in different source folders
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            # Use part of the file hash to guarantee uniqueness
            img_hash_snippet = get_file_hash(img_path)[:6]
            dest_path = os.path.join(dest_subdir, f"{name}_{img_hash_snippet}{ext}")
        
        try:
            shutil.copy2(img_path, dest_path)
        except Exception as e:
            print(f"Error copying {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare TB Dataset")
    parser.add_argument('--source', type=str, default='.', help="Source directory containing mixed images (default is current folder)")
    parser.add_argument('--dest', type=str, default='dataset', help="Destination directory for the organized dataset (default is 'dataset')")
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.dest)

    valid_exts = {'.png', '.jpg', '.jpeg'}
    
    print(f"Scanning for images in '{source_dir}' ...")
    
    # 1. & 2. Scan dataset folders and identify valid image formats
    all_images = []
    
    for root, dirs, files in os.walk(source_dir):
        # Skip the destination directory to avoid recursive processing
        if dest_dir in os.path.abspath(root):
            continue
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                all_images.append(os.path.join(root, file))

    total_images_found = len(all_images)
    print(f"Total valid images found: {total_images_found}")

    if total_images_found == 0:
        print("No images found. Exiting.")
        return

    # 3. & 4. Detect and remove duplicate images using image hashing
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
            # Automatically skipping adding the duplicate to our dataset
        else:
            seen_hashes.add(img_hash)
            unique_images.append(img_path)

    print(f"Duplicates automatically removed: {duplicates_removed}")
    print(f"Unique images available: {len(unique_images)}")

    # 5. Organize dataset into two main classes: TB, Normal
    print("\nClassifying images into TB and Normal...")
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

    print(f"Total TB images      : {len(tb_images)}")
    print(f"Total Normal images  : {len(normal_images)}")
    if unknown_images:
        print(f"Warning: {len(unknown_images)} images could not be classified automatically and will be ignored.")

    if not tb_images and not normal_images:
        print("Error: Could not classify any images. Ensure files or folders contain 'TB' or 'Normal' in their name.")
        return

    # 7. Split the dataset automatically (70% Train, 15% Validation, 15% Test)
    def split_data(images_list):
        if len(images_list) < 3:
            return images_list, [], []
            
        # 70% Train, 30% Temp
        train_data, temp_data = train_test_split(images_list, test_size=0.30, random_state=42)
        # Split Temp in half: 15% Val, 15% Test
        val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)
        
        return train_data, val_data, test_data

    tb_train, tb_val, tb_test = split_data(tb_images)
    normal_train, normal_val, normal_test = split_data(normal_images)

    # 6. Create a clean dataset folder structure
    splits = ['train', 'val', 'test']
    classes = ['TB', 'Normal']
    
    print(f"\nCreating dataset hierarchy at '{dest_dir}'...")
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    # Copy files into structured split folders
    # 8. Uniqueness inherently guaranteed across splits due to hashing earlier and independent split sets
    copy_files(tb_train, dest_dir, 'train', 'TB')
    copy_files(tb_val, dest_dir, 'val', 'TB')
    copy_files(tb_test, dest_dir, 'test', 'TB')

    copy_files(normal_train, dest_dir, 'train', 'Normal')
    copy_files(normal_val, dest_dir, 'val', 'Normal')
    copy_files(normal_test, dest_dir, 'test', 'Normal')

    # 9. Print final summary
    print("\n" + "="*45)
    print(" " * 10 + "FINAL DATASET SUMMARY")
    print("="*45)
    print(f"Total Original Images : {total_images_found}")
    print(f"Duplicates Removed    : {duplicates_removed}")
    print(f"Total Unique Images   : {len(unique_images)}")
    print(f"Unclassified Ignored  : {len(unknown_images)}")
    print("-" * 45)
    print(f"Total classified TB     : {len(tb_images)}")
    print(f"Total classified Normal : {len(normal_images)}")
    print("="*45)
    print(f"Train split (70%)     : TB={len(tb_train):<5} | Normal={len(normal_train):<5}")
    print(f"Validation split (15%): TB={len(tb_val):<5} | Normal={len(normal_val):<5}")
    print(f"Test split (15%)      : TB={len(tb_test):<5} | Normal={len(normal_test):<5}")
    print("="*45)
    print(f"Ready: Use 'dataset/' for training your model!")

if __name__ == "__main__":
    main()
