import os
import argparse
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
import gc

from diffusers.models.attention_processor import LoRAAttnProcessor
from tqdm.auto import tqdm
from PIL import Image
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion on cake images")
    parser.add_argument("--data_dir", type=str, default="../data/cakes")
    parser.add_argument("--output_dir", type=str, default="../models/torte_model")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Enable memory efficient attention
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        local_files_only=False,
        force_download=False,
        resume_download=True,
        proxies=None,
        timeout=100000
    )

    # Memory optimizations
    pipeline.enable_attention_slicing(1)
    pipeline.enable_vae_slicing()
    pipeline.enable_model_cpu_offload()

    # LoRA setup with reduced parameters
    unet = pipeline.unet
    for name, module in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            module.processor = LoRAAttnProcessor(
                hidden_size=module.to_q.in_features,
                cross_attention_dim=module.to_k.in_features if hasattr(module, "to_k") else None,
                rank=4  # Reduced rank
            )

    # Optimizer with reduced learning rate
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=1e-5,
        weight_decay=0.01
    )

    # Load and prepare text encoder
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer

    # Get image paths
    image_paths = [
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    print(f"Found {len(image_paths)} images for training")

    # Training loop
    progress_bar = tqdm(range(args.max_train_steps))

    for step in range(args.max_train_steps):
        # Clear memory
        if step % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        try:
            # Load and prepare image
            img_path = random.choice(image_paths)
            img = Image.open(img_path).convert("RGB").resize((args.resolution, args.resolution))
            img_tensor = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            # Prepare text input
            text_input = tokenizer(
                "a photo of a cake",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )

            # Training step
            optimizer.zero_grad()

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input.input_ids)[0]

            # Add noise
            noise = torch.randn_like(img_tensor)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (1,))
            noisy_images = pipeline.scheduler.add_noise(img_tensor, noise, timesteps)

            # Get model prediction
            noise_pred = unet(
                noisy_images,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss and optimize
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            progress_bar.update(1)

            # Save checkpoint
            if (step + 1) % 25 == 0:
                pipeline.save_pretrained(args.output_dir)
                print(f"\nCheckpoint saved at step {step + 1}")

        except Exception as e:
            print(f"Error at step {step}: {str(e)}")
            continue

    # Save final model
    pipeline.save_pretrained(args.output_dir)
    print("Training completed!")


if __name__ == "__main__":
    main()