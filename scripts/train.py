import os
import argparse

import numpy as np
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDPMScheduler
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from tqdm.auto import tqdm
from PIL import Image
import random


def parse_args():
   parser = argparse.ArgumentParser(description="Train Stable Diffusion on cake images")
   parser.add_argument("--data_dir", type=str, default="../data/cakes", help="Directory containing cake images")
   parser.add_argument("--output_dir", type=str, default="../models/torte_model",
                       help="Directory to save the model")
   parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base model to use")
   parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
   parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
   parser.add_argument("--learning_rate", type=float, default=2e-6, help="Initial learning rate")
   parser.add_argument("--max_train_steps", type=int, default=400, help="Total number of training steps")
   parser.add_argument("--use_lora", action="store_true", help="Use LoRA for training (requires fewer resources)")
   return parser.parse_args()


def main():
   args = parse_args()

   # Create output directory if it doesn't exist
   os.makedirs(args.output_dir, exist_ok=True)

   # Load the base model
   pipeline = StableDiffusionPipeline.from_pretrained(
       args.base_model,
       torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
   )

   if torch.cuda.is_available():
       pipeline = pipeline.to("cuda")

   # Training method: LoRA or full DreamBooth
   if args.use_lora:
       print("Using LoRA for training...")
       # LoRA configuration
       for name, module in pipeline.unet.named_modules():
           if name.endswith("attn1") or name.endswith("attn2"):
               module.processor = LoRAAttnProcessor(
                   hidden_size=module.processor.hidden_size,
                   cross_attention_dim=module.processor.cross_attention_dim if hasattr(module.processor,
                                                                                       "cross_attention_dim") else None,
                   rank=16,
               )

       # Optimizer
       optimizer_cls = torch.optim.AdamW
       optimizer = optimizer_cls(
           pipeline.unet.parameters(),
           lr=args.learning_rate,
       )
   else:
       print("Using DreamBooth for training...")
       # DreamBooth (full training)
       optimizer_cls = torch.optim.AdamW
       optimizer = optimizer_cls(
           pipeline.unet.parameters(),
           lr=args.learning_rate,
       )

   # Dataset creation (simplified)
   image_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
                  if f.endswith(('.jpg', '.png', '.jpeg'))]

   print(f"Found {len(image_paths)} images for training")

   # Simplified training loop
   progress_bar = tqdm(range(args.max_train_steps))
   progress_bar.set_description("Training steps")

   for step in range(args.max_train_steps):
       # Select a random image for each batch
       img_path = random.choice(image_paths)
       img = Image.open(img_path).convert("RGB").resize((args.resolution, args.resolution))

       # Input preparation
       input_img = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
       input_img = input_img.permute(2, 0, 1).unsqueeze(0).to(pipeline.device)

       # Training step
       optimizer.zero_grad()

       # Add training logic here (simplified for demonstration purposes)
       noise = torch.randn_like(input_img)
       timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (1,), device=pipeline.device)
       noisy_images = pipeline.scheduler.add_noise(input_img, noise, timesteps)

       # Pass through UNet
       noise_pred = pipeline.unet(noisy_images, timesteps).sample

       # Calculate loss
       loss = torch.nn.functional.mse_loss(noise_pred, noise)
       loss.backward()
       optimizer.step()

       progress_bar.update(1)

       # Save the model every 100 steps and at the end
       if step % 100 == 0 or step == args.max_train_steps - 1:
           pipeline.save_pretrained(args.output_dir)
           print(f"Model saved to {args.output_dir}")

   # Save the final model
   pipeline.save_pretrained(args.output_dir)
   print(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
   main()