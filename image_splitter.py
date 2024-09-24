import cv2
import numpy as np
from PIL import Image
import os

def extract_shapes(image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        extracted_shape = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_shape = extracted_shape[y:y+h, x:x+w]
        result_image = cv2.cvtColor(cropped_shape, cv2.COLOR_BGRA2RGBA)
        output_path = os.path.join(output_folder, f'shape_{i}.png')
        Image.fromarray(result_image).save(output_path)

    print("Shapes extracted successfully!")

image_path = 'image.png'  # Replace with the path to your image
output_folder = 'shapes'  # Folder where extracted shapes will be saved

extract_shapes(image_path, output_folder)
