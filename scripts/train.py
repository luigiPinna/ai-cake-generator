import os
import argparse
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch
import gc
from tqdm.auto import tqdm
from PIL import Image
import random


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stable Diffusion on cake images")
    parser.add_argument("--data_dir", type=str, default="../data/cakes")
    parser.add_argument("--output_dir", type=str, default="../models/torte_model")
    parser.add_argument("--resolution", type=int, default=64)  # Further reduced to 64
    parser.add_argument("--max_train_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Note: We would enable NaN checks on MPS, but it's not available in this PyTorch version
    # if device.type == "mps":
    #     torch.backends.mps.enable_check_nan()

    # Use a lightweight SD model to reduce memory usage
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        use_safetensors=False,  # Changed from True to False
        safety_checker=None,
        # The following options reduce memory usage
        low_cpu_mem_usage=True
        # Removed revision="fp16" since it's causing issues
    )

    # Memory optimizations
    pipeline.enable_attention_slicing(1)
    pipeline.enable_vae_slicing()

    # For Apple Silicon, we need to be careful with mixed precision
    # Keep everything in float32 for compatibility with MPS
    pipeline = pipeline.to(device)

    # Move pipeline to device
    pipeline = pipeline.to(device)

    # LoRA setup with reduced parameters
    unet = pipeline.unet
    for name, module in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            module.processor = LoRAAttnProcessor(
                hidden_size=module.to_q.in_features,
                cross_attention_dim=module.to_k.in_features if hasattr(module, "to_k") else None,
                rank=2  # Reduced from 4 to 2 for memory efficiency
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        try:
            # Load and prepare image - FIXED to use RGBA
            img_path = random.choice(image_paths)
            img = Image.open(img_path).convert("RGBA").resize((args.resolution, args.resolution))

            # Convert RGBA image to numpy array and normalize
            img_array = np.array(img).astype(np.float32) / 255.0

            # Normalize to [-1, 1] range
            img_array = img_array * 2.0 - 1.0

            # Convert to tensor with correct channel ordering
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)

            # Ensure we have 4 channels
            if img_tensor.shape[1] != 4:
                print(f"Warning: Image has {img_tensor.shape[1]} channels instead of 4")
                # Add alpha channel if missing
                if img_tensor.shape[1] == 3:
                    alpha = torch.ones((1, 1, args.resolution, args.resolution), device=device)
                    img_tensor = torch.cat([img_tensor, alpha], dim=1)

            # Prepare text input
            text_input = tokenizer(
                "a photo of a cake",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Training step
            optimizer.zero_grad()

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(text_input.input_ids)[0]

            # Add noise
            noise = torch.randn_like(img_tensor)
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (1,), device=device)
            noisy_images = pipeline.scheduler.add_noise(img_tensor, noise, timesteps)

            # Debug info
            if step == 0:
                print(f"Image tensor shape: {img_tensor.shape}")
                print(f"Noisy images shape: {noisy_images.shape}")
                print(f"Expected UNet input channels: 4")

            # Get model prediction
            # Use chunking to reduce memory usage
            noise_pred = unet(
                noisy_images,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            # Calculate loss and optimize
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

            progress_bar.update(1)

            # Save checkpoint
            if (step + 1) % 25 == 0:
                pipeline.save_pretrained(args.output_dir)
                print(f"\nCheckpoint saved at step {step + 1}")

        except Exception as e:
            print(f"Error at step {step}: {str(e)}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            continue

    # Save final model
    pipeline.save_pretrained(args.output_dir)
    print("Training completed!")


if __name__ == "__main__":
    main()