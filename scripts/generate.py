import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def parse_args():
   parser = argparse.ArgumentParser(description="Generate cake images with the trained model")
   parser.add_argument("--model_path", type=str, default="../models/torte_model",
                       help="Path to the trained model")
   parser.add_argument("--prompt", type=str, required=True, help="Text description of the cake to generate")
   parser.add_argument("--output_path", type=str, default="generated_cake.png",
                       help="Path where to save the image")
   parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
   parser.add_argument("--seed", type=int, default=None, help="Seed for generation")
   return parser.parse_args()


def main():
   args = parse_args()

   # Load the trained model
   print(f"Loading model from {args.model_path}...")
   pipeline = StableDiffusionPipeline.from_pretrained(
       args.model_path,
       torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
   )

   if torch.cuda.is_available():
       pipeline = pipeline.to("cuda")

   # Enhance the prompt
   enhanced_prompt = f"A realistic photo of an artisanal cake: {args.prompt}, high quality, detailed"

   # Generate the image
   print(f"Generating {args.num_images} images with prompt: '{enhanced_prompt}'")

   generator = None
   if args.seed is not None:
       generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

   images = pipeline(
       prompt=enhanced_prompt,
       num_images_per_prompt=args.num_images,
       generator=generator,
       num_inference_steps=50,
   ).images

   # Save the images
   for i, image in enumerate(images):
       output_path = args.output_path if args.num_images == 1 else f"{args.output_path.split('.')[0]}_{i}.{args.output_path.split('.')[1]}"
       image.save(output_path)
       print(f"Image saved at {output_path}")


if __name__ == "__main__":
   main()