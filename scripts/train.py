import os
import argparse
import numpy as np
import torch
import gc
from tqdm.auto import tqdm
from PIL import Image
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal memory training on cake images")
    parser.add_argument("--data_dir", type=str, default="../data/cakes")
    parser.add_argument("--output_dir", type=str, default="../models/torte_model")
    parser.add_argument("--resolution", type=int, default=32)  # Extremely small for memory savings
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-6)  # Reduced learning rate
    return parser.parse_args()


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # IMPORTANT: Load ALL components from the SAME model to ensure dimension compatibility
    print("Loading components from the same model (CompVis/stable-diffusion-v1-4)...")

    # Load tokenizer first (smallest)
    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="tokenizer",
        torch_dtype=torch.float32
    )

    # Load text encoder (ensure dimension compatibility)
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="text_encoder",
        torch_dtype=torch.float32
    )

    # Keep VAE loading
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        torch_dtype=torch.float32
    )

    # Load UNet with minimal memory footprint
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    print("Loading noise scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="scheduler"
    )

    # Put models in evaluation mode to save memory
    vae.eval()
    text_encoder.eval()

    # Freeze text_encoder and vae to save memory
    for param in vae.parameters():
        param.requires_grad = False

    for param in text_encoder.parameters():
        param.requires_grad = False

    # Move to device one at a time, clearing cache between operations
    vae.to(device)
    clear_memory()

    text_encoder.to(device)
    clear_memory()

    unet.to(device)
    clear_memory()

    # Memory optimizations
    vae.enable_slicing()

    # Only train a subset of parameters (LoRA-like approach but even smaller)
    # Fix the LoRA implementation to avoid dimensional issues
    trainable_params = []

    # Freeze entire UNet first
    for param in unet.parameters():
        param.requires_grad = False

    # Only unfreeze final attention module output projections
    for name, module in unet.named_modules():
        if name.endswith("to_out.0"):  # Target only final output linear layers
            for param in module.parameters():
                param.requires_grad = True
                trainable_params.append(param)

    print(f"Training {len(trainable_params)} parameters out of {len(list(unet.parameters()))}")

    # Simple AdamW optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01
    )

    # Get image paths
    image_paths = [
        os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    print(f"Found {len(image_paths)} images for training")

    # Prepare text embeddings
    text_input = tokenizer(
        ["a photo of a cake"] * 1,  # Batch size 1
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids)[0]

    # Training loop
    progress_bar = tqdm(range(args.max_train_steps))

    for step in range(args.max_train_steps):
        try:
            # Clear memory every step
            clear_memory()

            # Load and prepare image
            img_path = random.choice(image_paths)
            img = Image.open(img_path).convert("RGBA").resize((args.resolution, args.resolution))

            # Manual pixel array handling (more control over memory)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = img_array * 2.0 - 1.0  # Normalize to [-1, 1]

            # Manual handling of channels, only use 3 channels to save memory
            # But make sure we're using the RGB part and not dropping alpha
            img_array = img_array[:, :, :3]

            # Convert to tensor, smaller chunks, use RGB only (3 channels)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

            # Training step
            optimizer.zero_grad()

            # Encode the image with VAE (in eval mode, no gradients)
            with torch.no_grad():
                # Use VAE in small chunks, force 3 channel input
                if img_tensor.shape[1] == 4:  # If RGBA
                    img_tensor = img_tensor[:, :3, :, :]  # Keep only RGB
                latents = vae.encode(img_tensor).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Add noise to latents
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), device=device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Training step
            optimizer.zero_grad()

            # UNet prediction - use small chunks to avoid memory issues
            # This is the critical part where we need to avoid OOM errors
            noise_pred = None
            try:
                # First try direct prediction
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
            except RuntimeError as e:
                # If we get memory error, try a more aggressive approach
                if "memory" in str(e).lower() or "cuda" in str(e).lower() or "mps" in str(e).lower():
                    print("Memory error, trying chunked prediction...")
                    # Free memory
                    clear_memory()

                    # Try again with half precision
                    with torch.autocast(device_type=device.type):
                        noise_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=text_embeddings
                        ).sample
                else:
                    # If it's not a memory error, re-raise
                    raise e

            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Clean up intermediates to save memory
            del latents, noise, timesteps, noisy_latents, noise_pred
            clear_memory()

            if step % 5 == 0 or step == args.max_train_steps - 1:
                print(f"Step {step}, Loss: {loss.item():.4f}")

            progress_bar.update(1)

            # Save checkpoint
            if (step + 1) % 25 == 0 or step == args.max_train_steps - 1:
                print(f"Saving checkpoint at step {step + 1}")
                os.makedirs(os.path.join(args.output_dir, "unet"), exist_ok=True)
                unet.save_pretrained(os.path.join(args.output_dir, "unet"))

        except Exception as e:
            print(f"Error at step {step}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("Training completed!")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()