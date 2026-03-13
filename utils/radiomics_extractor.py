import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis

def extract_radiomics_features(image_path):
    """
    Extracts statistical and texture (GLCM) features from an X-ray image 
    using scikit-image and scipy. This replaces PyRadiomics since PyRadiomics 
    requires C++ build tools on Windows which often fail to install.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        dict: A dictionary of extracted texture and statistical features.
    """
    # 1. Read the image using OpenCV and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Resize for consistency
    img = cv2.resize(img, (224, 224))
    
    features = {}
    
    # --- FIRST ORDER STATISTICAL FEATURES ---
    img_flat = img.flatten()
    features['firstorder_Mean'] = np.mean(img_flat)
    features['firstorder_Variance'] = np.var(img_flat)
    features['firstorder_StandardDeviation'] = np.std(img_flat)
    features['firstorder_Skewness'] = skew(img_flat)
    features['firstorder_Kurtosis'] = kurtosis(img_flat)
    features['firstorder_Entropy'] = shannon_entropy(img)
    features['firstorder_Minimum'] = np.min(img_flat)
    features['firstorder_Maximum'] = np.max(img_flat)
    features['firstorder_Median'] = np.median(img_flat)
    features['firstorder_RootMeanSquared'] = np.sqrt(np.mean(img_flat**2))
    
    # --- GLCM TEXTURE FEATURES (Gray Level Co-occurrence Matrix) ---
    # To compute GLCM, we usually reduce the number of gray levels to speed it up and reduce noise
    # Binning to 32 discrete gray levels
    img_binned = (img // 8).astype(np.uint8) 
    
    # Calculate GLCM at distances [1, 2, 4] and angles [0, 45, 90, 135] degrees
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(img_binned, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
    
    # Helper to average over all distances and angles for a compact feature set
    def get_glcm_prop(prop_name):
        return np.mean(graycoprops(glcm, prop_name))
        
    features['glcm_Contrast'] = get_glcm_prop('contrast')
    features['glcm_Dissimilarity'] = get_glcm_prop('dissimilarity')
    features['glcm_Homogeneity'] = get_glcm_prop('homogeneity')
    features['glcm_Energy'] = get_glcm_prop('energy')
    features['glcm_Correlation'] = get_glcm_prop('correlation')
    features['glcm_ASM'] = get_glcm_prop('ASM')
    
    # We can also add specifically the maximum correlation across angles to catch directional textures
    features['glcm_MaxCorrelation'] = np.max(graycoprops(glcm, 'correlation'))
    features['glcm_MaxContrast'] = np.max(graycoprops(glcm, 'contrast'))

    # Return the 18 constructed features 
    return features
