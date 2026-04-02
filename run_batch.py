"""
run_batch.py — MagicFace batch processor.

Loads all models ONCE and loops through all images, avoiding the massive
overhead of reloading model weights per image that subprocess-based notebooks cause.

Usage:
    python run_batch.py --expression angry
    python run_batch.py --expression fearful
    python run_batch.py --expression angry --categories dominant
    python run_batch.py --expression angry fearful --categories dominant submissive

Outputs:
    edited_images/angry/<category>/
    edited_images/fearful/<category>/
"""

import os
import sys
import argparse
import types

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'utils'))

from mgface.pipelines_mgface.pipeline_mgface import MgPipeline
from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel

# Load preprocessing models on import (once for the whole run)
print("Loading preprocessing models (InsightFace)...")
from preprocess import crop_one_image

print("Loading background models (BiSeNet + D3DFR)...")
from retrieve_bg import make_bg_for_one_image

# ── Constants ─────────────────────────────────────────────────────────────────

IND_DICT = {
    'AU1': 0, 'AU2': 1, 'AU4': 2,  'AU5': 3,  'AU6': 4,  'AU9': 5,
    'AU12': 6, 'AU15': 7, 'AU17': 8, 'AU20': 9, 'AU25': 10, 'AU26': 11,
}

# Closed-mouth expression profiles.
# AU25 (Lips Part) and AU26 (Jaw Drop) are intentionally excluded.
EXPRESSIONS = {
    'angry':   {'aus': 'AU4+AU5',              'variations': '5+5'},
    'fearful': {'aus': 'AU1+AU2+AU4+AU5+AU20', 'variations': '5+5+5+5+5'},
}

CATEGORIES = {
    'dominant':   os.path.join(PROJECT_ROOT, 'processed_identities', 'processed_dominant_identities'),
    'submissive': os.path.join(PROJECT_ROOT, 'processed_identities', 'processed_submissive_identities'),
}

N_IMAGES = 50
DEVICE   = 'cuda'
SEED     = 424

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_au_vector(aus_str, variations_str):
    au_prompt = np.zeros((12,))
    for au, val in zip(aus_str.split('+'), variations_str.split('+')):
        au_prompt[IND_DICT[au]] = int(val)
    return au_prompt


def load_pipeline():
    print("\nLoading diffusion pipeline (this may take a minute)...")
    model_id = 'sd-legacy/stable-diffusion-v1-5'

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder='vae', cache_dir=PROJECT_ROOT).to(DEVICE)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder='text_encoder', cache_dir=PROJECT_ROOT).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder='tokenizer', cache_dir=PROJECT_ROOT)
    unet_ID = UNetID2DConditionModel.from_pretrained(
        'mengtingwei/magicface', subfolder='ID_enc',
        use_safetensors=True, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, cache_dir=PROJECT_ROOT)
    unet_deno = UNetDeno2DConditionModel.from_pretrained(
        'mengtingwei/magicface', subfolder='denoising_unet',
        use_safetensors=True, low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True, cache_dir=PROJECT_ROOT)

    for m in (vae, text_encoder, unet_ID, unet_deno):
        m.requires_grad_(False)

    pipeline = MgPipeline.from_pretrained(
        model_id,
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        unet_ID=unet_ID, unet_deno=unet_deno,
        safety_checker=None, torch_dtype=torch.float16,
        cache_dir=PROJECT_ROOT,
    ).to(DEVICE)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    prompt = 'A close up of a person.'
    input_ids = tokenizer(
        [prompt], max_length=tokenizer.model_max_length,
        padding='max_length', truncation=True, return_tensors='pt'
    ).input_ids
    prompt_embeds = text_encoder(input_ids.to(DEVICE))[0]

    print("Pipeline ready.\n")
    return pipeline, prompt_embeds


def run_inference(pipeline, prompt_embeds, cropped_path, bg_path, au_vector, out_path):
    transform = transforms.ToTensor()
    source = transform(Image.open(cropped_path)).unsqueeze(0)
    bg     = transform(Image.open(bg_path)).unsqueeze(0)

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    tor_exp   = torch.from_numpy(au_vector).unsqueeze(0)

    result = pipeline(
        prompt_embeds=prompt_embeds,
        source=source, bg=bg, au=tor_exp,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    result.save(out_path)


# ── Main processing loop ──────────────────────────────────────────────────────

def process(pipeline, prompt_embeds, expression, categories):
    exp_cfg   = EXPRESSIONS[expression]
    au_vector = make_au_vector(exp_cfg['aus'], exp_cfg['variations'])

    print(f"\n{'='*60}")
    print(f"Expression : {expression.upper()}")
    print(f"AUs        : {exp_cfg['aus']}  (variations: {exp_cfg['variations']})")
    print(f"Categories : {', '.join(categories)}")
    print(f"{'='*60}\n")

    for category in categories:
        input_dir  = CATEGORIES[category]
        output_dir = os.path.join(PROJECT_ROOT, 'edited_images', expression, category)
        tmp_dir    = os.path.join(PROJECT_ROOT, 'test_images', category)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tmp_dir,    exist_ok=True)

        image_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.jpg'))[:N_IMAGES]
        print(f"[{category.upper()}] {len(image_files)} images  →  {output_dir}\n")

        for i, img_name in enumerate(image_files, 1):
            base        = os.path.splitext(img_name)[0]
            input_path  = os.path.join(input_dir, img_name)
            cropped     = os.path.join(tmp_dir, f"{base}_cropped.png")
            bg          = os.path.join(tmp_dir, f"{base}_bg.png")
            out_path    = os.path.join(output_dir, f"{base}.png")

            prefix = f"  [{i:>3}/{len(image_files)}] {img_name}"

            if os.path.exists(out_path):
                print(f"{prefix}  SKIP (already done)")
                continue

            print(f"{prefix}", end='  ', flush=True)

            fake_args = types.SimpleNamespace(img_path=input_path, save_path=cropped)
            crop_one_image(fake_args)
            print("cropped", end=' → ', flush=True)

            fake_args.save_path = bg
            make_bg_for_one_image(fake_args)
            print("bg", end=' → ', flush=True)

            run_inference(pipeline, prompt_embeds, cropped, bg, au_vector, out_path)
            print("done")

        print(f"\n[{category.upper()}] Finished. Results in: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description='MagicFace batch processor')
    parser.add_argument(
        '--expression', nargs='+', choices=['angry', 'fearful'],
        default=['angry'],
        help='Expression(s) to generate (default: angry)')
    parser.add_argument(
        '--categories', nargs='+', choices=['dominant', 'submissive'],
        default=['dominant', 'submissive'],
        help='Category/categories to process (default: both)')
    args = parser.parse_args()

    pipeline, prompt_embeds = load_pipeline()

    for expression in args.expression:
        process(pipeline, prompt_embeds, expression, args.categories)

    print("\nAll done!")


if __name__ == '__main__':
    main()
