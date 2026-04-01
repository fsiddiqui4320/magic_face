import os
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

def test_face_detection(img_path, root_dir):
    print(f"Testing image: {img_path}")
    print(f"Using root_dir: {root_dir}")
    
    app = FaceAnalysis(name='antelopev2', root=root_dir, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640)) # Use CPU (ctx_id=-1) for debugging if GPU is flaky
    
    im_pil = Image.open(img_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    
    face_info = app.get(img_cv)
    print(f"Found {len(face_info)} faces.")
    
    if len(face_info) > 0:
        for i, face in enumerate(face_info):
            print(f"Face {i} bbox: {face['bbox']}")
            print(f"Face {i} det_score: {face['det_score']}")
    else:
        print("No faces found!")

if __name__ == "__main__":
    img_13 = r"C:\Users\faris3\MagicFace\test_images\processed_identities\WM_13.jpg"
    img_1 = r"C:\Users\faris3\MagicFace\test_images\processed_identities\WM_1.jpg"
    
    # Try with utils/third_party_files
    root_utils = r"C:\Users\faris3\MagicFace\utils\third_party_files"
    
    print("--- Testing WM_13 with utils models ---")
    test_face_detection(img_13, root_utils)
    
    print("\n--- Testing WM_1 with utils models ---")
    test_face_detection(img_1, root_utils)
