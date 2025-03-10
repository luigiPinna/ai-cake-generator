from PIL import Image
import os

input_dir = "cake-generator/data/torte"

for file in os.listdir(input_dir):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img.save(img_path)