import gradio as gr
import torch
import os
import sys
import time
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler

# Add the main folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class CakeGeneratorChat:
    def __init__(self, model_path="../models/torte_model"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        print(f"Loading model on {self.device}...")

        # Instead of loading the whole pipeline at once, we load components separately
        # and then combine them, similar to our training approach

        # Check if we have a full pipeline or just fine-tuned UNet
        if os.path.exists(os.path.join(self.model_path, "unet")) and not os.path.exists(
                os.path.join(self.model_path, "text_encoder")):
            print("Found fine-tuned UNet only. Loading base components from SD 1.4...")

            # Load base components
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="tokenizer"
            )

            self.text_encoder = CLIPTextModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="text_encoder"
            ).to(self.device)

            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae"
            ).to(self.device)

            self.scheduler = DDPMScheduler.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="scheduler"
            )

            # Load fine-tuned UNet
            print(f"Loading fine-tuned UNet from {self.model_path}...")
            self.unet = UNet2DConditionModel.from_pretrained(
                os.path.join(self.model_path, "unet")
            ).to(self.device)

            # Create the pipeline manually
            from transformers import CLIPFeatureExtractor

            # Load feature extractor
            feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="feature_extractor"
            )

            self.pipeline = StableDiffusionPipeline(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
                requires_safety_checker=False
            )

        else:
            # Try to load the full pipeline
            print("Attempting to load full pipeline...")
            try:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None
                ).to(self.device)
            except Exception as e:
                print(f"Error loading full pipeline: {e}")
                print("Falling back to base model with memory optimizations...")

                # If loading fails, use base model
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    safety_checker=None
                ).to(self.device)

        # Apply memory optimizations
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()

        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()

        print("Model loaded successfully!")

    def generate_cake(self, prompt):
        # Enhance the prompt
        enhanced_prompt = f"A realistic photo of an artisanal cake: {prompt}, high quality, detailed"

        # Generate the image
        start_time = time.time()

        try:
            # Clear cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

            with torch.no_grad():
                image = self.pipeline(
                    prompt=enhanced_prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=30,  # Reduced from 50 to 30 for speed
                    height=512,
                    width=512,
                ).images[0]

            end_time = time.time()

            # Calculate generation time
            generation_time = round(end_time - start_time, 2)

            # Prepare the response
            response = f"I generated your cake in {generation_time} seconds."

            return image, response

        except Exception as e:
            # Handle any errors during generation
            error_message = f"Sorry, I couldn't generate the cake. Error: {str(e)}"
            print(error_message)

            # Return a placeholder or None for the image
            return None, error_message


def main():
    # Initialize the cake generator
    cake_generator = CakeGeneratorChat()

    # Chat history
    chat_history = []

    # Message processing function
    def chat(message, history):
        history.append((message, ""))

        if any(keyword in message.lower() for keyword in ["torta", "dolce", "cake", "genera", "crea"]):
            # This is a cake generation request
            image, response = cake_generator.generate_cake(message)
            history[-1] = (history[-1][0], response)
            return history, image
        else:
            # This is a normal conversation
            response = "Hello! I'm your cake generation assistant. Describe the cake you want and I'll create it for you."
            history[-1] = (history[-1][0], response)
            return history, None

    # Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# AI Cake Generator")
        gr.Markdown("Describe the cake you want and the AI will generate it for you!")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot([], elem_id="chatbot", height=400)
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Describe the cake you want...",
                    container=False
                )
                btn = gr.Button("Send")

            with gr.Column(scale=2):
                output_image = gr.Image(label="Your generated cake", height=400)

        btn.click(chat, inputs=[msg, chatbot], outputs=[chatbot, output_image])
        msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, output_image])

    # Launch the interface
    demo.launch(share=True)


if __name__ == "__main__":
    main()