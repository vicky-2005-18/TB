import os
import pandas as pd
from tqdm import tqdm
from utils.radiomics_extractor import extract_radiomics_features

def main():
    print("Starting Radiomics Feature Extraction...")
    
    # Dataset path according to README structure
    base_dir = "dataset"
    categories = ["TB", "Normal"]
    
    if not os.path.exists(base_dir):
        print(f"Error: Could not find '{base_dir}' directory.")
        print("Please ensure your data is stored in dataset/TB/ and dataset/Normal/")
        return
        
    all_features = []
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        
        if not os.path.exists(cat_dir):
            print(f"Warning: Category completely missing at {cat_dir}")
            continue
            
        label = 1 if category == "TB" else 0
        
        # Determine all images in this category across train/val/test if using split, 
        # or flat folders if using simple structure.
        # Assuming simple structure per README: dataset/TB and dataset/Normal
        image_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing category: {category} ({len(image_files)} images found)...")
        
        for img_name in tqdm(image_files, desc=f"Extracting {category}"):
            img_path = os.path.join(cat_dir, img_name)
            
            try:
                features = extract_radiomics_features(img_path)
                
                if features is not None:
                    # Append necessary metadata (label, filename)
                    features['image_name'] = img_name
                    features['label'] = label
                    all_features.append(features)
                    
            except Exception as e:
                print(f"Failed extracting from {img_path}: {e}")
                
    if not all_features:
        print("No features extracted. Please check your dataset.")
        return
        
    # Convert list of dicts to a pandas DataFrame
    df = pd.DataFrame(all_features)
    
    # Simple validation output
    print("\nFeature extraction complete!")
    print(f"Dataset Shape: {df.shape[0]} samples with {df.shape[1] - 2} features.")
    
    # Save the dataframe to a CSV file
    output_path = os.path.join(base_dir, "radiomics_features.csv")
    df.to_csv(output_path, index=False)
    print(f"Features saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
