import gradio as gr
import torch
import os
import sys
import time
from diffusers import StableDiffusionPipeline

# Add the main folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class CakeGeneratorChat:
    def __init__(self, model_path="../models/torte_model"):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        print("Loading model...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")
        print("Model loaded successfully!")

    def generate_cake(self, prompt):
        # Enhance the prompt
        enhanced_prompt = f"A realistic photo of an artisanal cake: {prompt}, high quality, detailed"

        # Generate the image
        start_time = time.time()
        with torch.no_grad():
            image = self.pipeline(
                prompt=enhanced_prompt,
                num_images_per_prompt=1,
                num_inference_steps=50,
            ).images[0]
        end_time = time.time()

        # Calculate generation time
        generation_time = round(end_time - start_time, 2)

        # Prepare the response
        response = f"I generated your cake in {generation_time} seconds."

        return image, response


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
