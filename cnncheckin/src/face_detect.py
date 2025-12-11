import cv2
import sys
import os

def detect_face(input_path, output_path):
    """
    Detects the largest face in the image and saves the cropped version.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Load image
        img = cv2.imread(input_path)
        if img is None:
            print("Error: Could not load image")
            sys.exit(1)
            
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Load Haar Cascade
        # We try to find the standard xml file in likely locations or use cv2 data
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("Error: Could not load haarcascade")
            sys.exit(1)
            
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No faces detected")
            sys.exit(2)  # Exit code 2 for no face found
            
        # Find largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add margin (20%)
        margin_w = int(w * 0.2)
        margin_h = int(h * 0.2)
        
        img_h, img_w = img.shape[:2]
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(img_w, x + w + margin_w)
        y2 = min(img_h, y + h + margin_h)
        
        # Crop
        face_img = img[y1:y2, x1:x2]
        
        # Save
        cv2.imwrite(output_path, face_img)
        print(f"Face saved to {output_path}")
        sys.exit(0)
        
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 face_detect.py <input_image> <output_image>")
        sys.exit(1)
        
    detect_face(sys.argv[1], sys.argv[2])
