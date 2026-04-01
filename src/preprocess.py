import cv2
import numpy as np
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Step 1 Remove Black borders
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours , _ =  cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    img = img[y: y+h, x: x+w]
    
    #Step 2 Ben Graham - Fix uneven lighting
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=30)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # Step 3 Resize and clip pixel values
    img = np.clip(img, 0, 255).astype(np.uint8) # pexel below 0 becomes 0 , pixel above 255 becomes 255
    img = cv2.resize(img, (224, 224))

    return img


if __name__ == "__main__":
    input_dir = "data/raw/train_images"
    output_dir = "data/processed"

    images = os.listdir(input_dir)
    total = len(images)

    for i, filename in enumerate(images):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = preprocess_image(input_path)
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if i % 100 ==0:
            print(f"Processed {i}/{total} images")

    print("Done")