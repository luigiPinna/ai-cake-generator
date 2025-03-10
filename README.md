# Cake Generator AI Project

An AI-powered cake image generator project that uses Stable Diffusion to create realistic custom cake images based on text descriptions.

## Overview

This project allows you to fine-tune a Stable Diffusion model using your own collection of cake photos and then generate new, unique cake designs through a simple chat interface. Perfect for bakery owners, cake designers, or anyone who wants to experiment with AI-generated pastry designs.

## Features

- **Custom Model Training**: Fine-tune Stable Diffusion on your own collection of cake photos
- **Text-to-Image Generation**: Create photorealistic cake images by describing them in natural language
- **Chat Interface**: User-friendly chat interface to request and generate cake images
- **Flexible Deployment**: Run locally on your own machine or on free cloud services

## Requirements

- Python 3.9+ (Python 3.10 recommended for best compatibility)
- GPU for training (can use Google Colab or other free GPU services)
- 20-30 high-quality cake images for training
- 10GB+ free disk space

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cake-generator.git
   cd cake-generator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter dependency conflicts, try the simplified installation:
   ```bash
   pip install huggingface-hub==0.17.3
   pip install torch>=2.2.0 torchvision>=0.13.0 numpy>=1.26.0 pillow>=9.0.0 tqdm>=4.65.0
   pip install transformers==4.33.0 accelerate==0.22.0 diffusers==0.23.0 gradio==3.48.0
   ```

## Project Structure

```
cake-generator/
│
├── data/                  # Folder for cake images
│   └── cakes/             # Subfolder with images
│
├── models/                # Folder where trained models will be saved
│
├── scripts/
│   ├── train.py           # Script for model training
│   └── generate.py        # Script for generating new images
│
├── app/
│   └── chat_interface.py  # Chat interface for interacting with the model
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Data Preparation

1. **Collect Images**: Gather 20-30 high-quality photos of cakes, ideally from different angles and with good lighting.

2. **Organize Images**:
   ```bash
   mkdir -p data/torte
   cp /path/to/your/cake/photos/* data/torte/
   ```

3. **Resize Images** (optional but recommended):
   ```python
   from PIL import Image
   import os
   
   input_dir = "data/torte"
   
   for file in os.listdir(input_dir):
       if file.endswith(('.jpg', '.png', '.jpeg')):
           img_path = os.path.join(input_dir, file)
           img = Image.open(img_path)
           img = img.resize((512, 512))
           img.save(img_path)
   ```

## Training the Model

1. **Start the training process:**
   ```bash
   python scripts/train.py --data_dir=./data/torte --output_dir=./models/torte_model --use_lora
   ```

   For more advanced training (requires more resources):
   ```bash
   python scripts/train.py --data_dir=./data/torte --output_dir=./models/torte_model --max_train_steps=800
   ```

2. **Training parameters:**
   - `--data_dir`: Directory with cake images
   - `--output_dir`: Where to save the trained model
   - `--base_model`: Base model to fine-tune (default: "runwayml/stable-diffusion-v1-5")
   - `--use_lora`: Use LoRA for efficient training (recommended for systems with limited resources)
   - `--max_train_steps`: Number of training steps (higher = better results but more time)

## Generating Images

1. **Generate a cake image:**
   ```bash
   python scripts/generate.py --model_path=./models/torte_model --prompt="chocolate cake with vanilla frosting and strawberry decorations" --output_path=./chocolate_cake.png
   ```

2. **Generation parameters:**
   - `--model_path`: Path to your trained model
   - `--prompt`: Text description of the cake
   - `--output_path`: Where to save the generated image
   - `--num_images`: Number of images to generate (default: 1)
   - `--seed`: Random seed for reproducibility

## Chat Interface

1. **Launch the chat interface:**
   ```bash
   python app/chat_interface.py
   ```

2. **Access the interface:**
   - The interface will be available at http://127.0.0.1:7860 in your web browser
   - Type cake descriptions in the text box and press "Send" to generate images

## Using Google Colab (Alternative for Training)

If you don't have a powerful GPU, you can use Google Colab's free GPU:

1. Upload your cake images to Google Drive
2. Create a new Colab notebook
3. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Install dependencies and run training in the notebook
5. Download the trained model to your local machine

## Troubleshooting

- **Out of Memory Errors**: Reduce image resolution or use `--use_lora` option
- **Slow Training**: Training can take 1-3 hours with a GPU; be patient
- **Low-Quality Images**: Increase `--max_train_steps` or add more training images
- **Installation Issues**: Try installing packages one by one or use a different Python version

## Extensions & Improvements

- Add more training images for better results
- Use SDXL as base model for higher quality images
- Implement ControlNet for better control over cake shape/structure
- Deploy as a web service using Hugging Face Spaces

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for Diffusers library
- Stability AI for Stable Diffusion
- The open-source AI community

---

For questions, suggestions, or improvements, please open an issue or contact the repository maintainer.