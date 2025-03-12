import os
import subprocess
from pathlib import Path
import imghdr

import numpy as np
import torch
from PIL import Image

def convert_images(input_dir):
    """Convert iphone images to jpg bse img"""
    input_dir = Path(input_dir)

    for file in input_dir.glob('*.JP*G'):
        try:
            output_file = file.parent / f"{file.stem}_converted.jpg"

            # Usa exiftool per estrarre e riconvertire l'immagine
            cmd = ['exiftool', '-b', '-PreviewImage', str(file)]

            with open(output_file, 'wb') as f:
                subprocess.run(cmd, stdout=f, check=True)

            print(f"Successfully converted: {file} -> {output_file}")

        except subprocess.CalledProcessError as e:
            print(f"Error converting {file}: {str(e)}")
        except Exception as e:
            print(f"Unexpected error with {file}: {str(e)}")


def resize_img(input_dir, size):
    """Resize all images in a folder"""
    for file in os.listdir(input_dir):
        if file.endswith(('.jpg', '.png', '.jpeg', '.JPG')):
            img_path = ""
            original_path = os.path.join(input_dir, file)
            try:
                actual_type = imghdr.what(original_path)
                print(f"File: {file} - Detected type: {actual_type}")
                img_path = os.path.join(input_dir, file)
                img = Image.open(img_path)
                img = img.resize((size, size))
                print("Saving img: ", img_path)
                img.save(img_path)
            except Exception as e:
                print(f"Error with img: {img_path}. Motivo {str(e)}", )



def convert_images_to_rgba(input_dir, output_dir=None):
    """
    Converts all images in a directory from RGB (3 channels) to RGBA (4 channels).

    Returns:
        int: Number of successfully converted images
    """
    # If output_dir is not specified, use input_dir (overwrites original files)
    if output_dir is None:
        output_dir = input_dir
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    # Common image file extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    count = 0

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # Open the image
                img = Image.open(input_path)

                # Convert to RGBA (4 channels)
                if img.mode != 'RGBA':
                    # Convert to RGBA
                    rgba_img = img.convert('RGBA')

                    # Save the converted image
                    # Use PNG to preserve alpha channel
                    output_path = os.path.splitext(output_path)[0] + '.png'
                    rgba_img.save(output_path)
                    count += 1
                    print(f"Converted: {filename} -> {os.path.basename(output_path)}")
                else:
                    print(f"Already in RGBA format: {filename}")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")

    return count


def inspect_image(input_dir):
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        # Costruisci il percorso completo
        image_path = os.path.join(input_dir, filename)

        # Verifica che sia un file e non una directory
        if not os.path.isfile(image_path):
            continue

        # Verifica che sia un'immagine
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            continue

        try:
            img = Image.open(image_path)
            print(f"\nImage: {filename}")
            print(f"Image mode: {img.mode}")
            print(f"Image size: {img.size}")

            # Converti in RGBA
            img_rgba = img.convert('RGBA')
            print(f"After conversion mode: {img_rgba.mode}")

            # Converti in array numpy
            img_array = np.array(img_rgba)
            print(f"Numpy array shape: {img_array.shape}")

            # Converti in tensor
            img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            print(f"Tensor shape: {img_tensor.shape}")

            # Stampa solo la forma del tensor invece dell'intero tensor
            print("Tensor shape:", img_tensor.shape)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":

    INPUT_DIR = "../data/cakes"
    SIZE = 512

    # convrt images
    #convert_images(INPUT_DIR)

    # resize imgs
    #resize_img(INPUT_DIR, SIZE)


    # Run the conversion from RGB (3 channels) to RGBA (4 channels).
    # num_converted = convert_images_to_rgba(INPUT_DIR)
    # print(f"\nConversion completed! {num_converted} images converted to RGBA format.")

    # Check img format
    inspect_image(INPUT_DIR)



