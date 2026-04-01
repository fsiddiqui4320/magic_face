from PIL import Image
import os

def resize_image(input_path, output_path, target_size=(512, 512)):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return
    
    img = Image.open(input_path)
    # Ensure it's in RGB (if it's a PNG with transparency)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img.save(output_path)
    print(f"Resized image saved to {output_path} (Resolution: {target_size})")

if __name__ == "__main__":
    # Using the path observed in the directory listing
    input_img = r"C:\Users\faris3\MagicFace\test_images\processed_identities\WM_5.jpg"
    output_img = r"C:\Users\faris3\MagicFace\test_images\WM_5_resized.png"
    
    resize_image(input_img, output_img)
