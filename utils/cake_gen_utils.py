import imghdr

from PIL import Image
import os

input_dir = "../data/cakes"

for file in os.listdir(input_dir):
    if file.endswith(('.jpg', '.png', '.jpeg', '.JPG')):
        img_path = ""
        original_path = os.path.join(input_dir, file)
        try:
            actual_type = imghdr.what(original_path)
            print(f"File: {file} - Detected type: {actual_type}")
            img_path = os.path.join(input_dir, file)
            img = Image.open(img_path)
            img = img.resize((512, 512))
            print("Saving img: ", img_path)
            img.save(img_path)
        except Exception as e:
            print(f"Error with img: {img_path}. Motivo {str(e)}", )


