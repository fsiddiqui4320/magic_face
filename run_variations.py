"""
run_variations.py — MagicFace expression variation generator.

Generates 4 emotions × 3 AU variations for specified identities,
outputting to variation_test/{wm_id}/{emotion}{n}.png.

Usage:
    python run_variations.py
    python run_variations.py --identities WM_136
    python run_variations.py --force
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

# ── Constants ─────────────────────────────────────────────────────────────────

IND_DICT = {
    'AU1': 0, 'AU2': 1, 'AU4': 2,  'AU5': 3,  'AU6': 4,  'AU9': 5,
    'AU12': 6, 'AU15': 7, 'AU17': 8, 'AU20': 9, 'AU25': 10, 'AU26': 11,
}

# Identity → category mapping. Category determines the source image directory.
IDENTITIES = {
    'WM_136': 'dominant',
    'WM_344': 'submissive',
}

CATEGORY_DIRS = {
    'dominant':   os.path.join(PROJECT_ROOT, 'processed_identities', 'processed_dominant_identities'),
    'submissive': os.path.join(PROJECT_ROOT, 'processed_identities', 'processed_submissive_identities'),
}

# Each list contains 3 profiles in order (variation 1, 2, 3).
VARIATIONS = {
    'happy': [
        {'aus': 'AU12+AU6',       'variations': '3+2'},
        {'aus': 'AU12+AU6',       'variations': '5+4'},
        {'aus': 'AU12+AU6+AU25',  'variations': '5+5+-2'},
    ],
    'sad': [
        {'aus': 'AU1+AU4+AU15', 'variations': '3+3+2'},
        {'aus': 'AU1+AU4+AU15', 'variations': '5+5+3'},
        {'aus': 'AU1+AU4+AU15', 'variations': '5+5+5'},
    ],
    'angry': [
        {'aus': 'AU4+AU5+AU25+AU12', 'variations': '5+4+-2+-2'},
        {'aus': 'AU4+AU5+AU25+AU12', 'variations': '5+4+-2+-1'},
        {'aus': 'AU4+AU5+AU25',      'variations': '5+4+-2'},
    ],
    'fearful': [
        {'aus': 'AU1+AU4+AU5+AU20+AU25',     'variations': '4+4+4+2+-2'},
        {'aus': 'AU1+AU2+AU4+AU5+AU20+AU25', 'variations': '4+1+4+3+2+-2'},
        {'aus': 'AU1+AU4+AU5+AU20+AU25',     'variations': '5+5+4+3+-2'},
    ],
}

DEVICE = 'cuda'
SEED   = 424


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_au_vector(aus_str, variations_str):
    au_prompt = np.zeros((12,))
    for au, val in zip(aus_str.split('+'), variations_str.split('+')):
        au_prompt[IND_DICT[au]] = int(val)
    return au_prompt


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline():
    from mgface.pipelines_mgface.pipeline_mgface import MgPipeline
    from mgface.pipelines_mgface.unet_ID_2d_condition import UNetID2DConditionModel
    from mgface.pipelines_mgface.unet_deno_2d_condition import UNetDeno2DConditionModel

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


# ── Preprocessing and batch processing ───────────────────────────────────────

def preprocess_identity(wm_id, category, force=False):
    """Return (cropped_path, bg_path), generating them if needed.

    Files are cached at test_images/{category}/{wm_id}_cropped.png
    and test_images/{category}/{wm_id}_bg.png — same location used
    by run_batch.py, so prior preprocessing is automatically reused.
    """
    from preprocess import crop_one_image
    from retrieve_bg import make_bg_for_one_image

    source_path = os.path.join(CATEGORY_DIRS[category], f'{wm_id}.jpg')
    tmp_dir     = os.path.join(PROJECT_ROOT, 'test_images', category)
    os.makedirs(tmp_dir, exist_ok=True)

    cropped = os.path.join(tmp_dir, f'{wm_id}_cropped.png')
    bg      = os.path.join(tmp_dir, f'{wm_id}_bg.png')

    if not os.path.exists(cropped) or force:
        fake_args = types.SimpleNamespace(img_path=source_path, save_path=cropped)
        crop_one_image(fake_args)

    if not os.path.exists(bg) or force:
        fake_args = types.SimpleNamespace(img_path=source_path, save_path=bg)
        make_bg_for_one_image(fake_args)

    return cropped, bg


def process_all_variations(pipeline, prompt_embeds, identities, force=False):
    """Run all 4 emotions × 3 variations for each identity in `identities`."""
    for wm_id in identities:
        category = IDENTITIES[wm_id]
        out_dir  = os.path.join(PROJECT_ROOT, 'variation_test', wm_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Identity : {wm_id}  ({category})")
        print(f"Output   : {out_dir}")
        print(f"{'='*60}")

        print("  Preprocessing (crop + bg)...", end=' ', flush=True)
        cropped, bg = preprocess_identity(wm_id, category, force=force)
        print("done")

        for emotion, profiles in VARIATIONS.items():
            for n, profile in enumerate(profiles, 1):
                out_path = os.path.join(out_dir, f'{emotion}{n}.png')
                label    = f'  {emotion}{n}'

                if os.path.exists(out_path) and not force:
                    print(f'{label}  SKIP (exists — use --force to rerun)')
                    continue

                au_vector = make_au_vector(profile['aus'], profile['variations'])
                print(f'{label}  AUs={profile["aus"]}  vals={profile["variations"]}',
                      end='  →  ', flush=True)
                run_inference(pipeline, prompt_embeds, cropped, bg, au_vector, out_path)
                print('done')

        print(f"\n  {wm_id} complete. 12 images in {out_dir}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MagicFace expression variation generator')
    parser.add_argument(
        '--identities', nargs='+',
        default=list(IDENTITIES.keys()),
        choices=list(IDENTITIES.keys()),
        help=f'Which identities to process (default: all — {list(IDENTITIES.keys())})')
    parser.add_argument(
        '--force', action='store_true',
        help='Reprocess even if output already exists')
    args = parser.parse_args()

    print(f"Identities : {args.identities}")
    print(f"Force      : {args.force}")

    print("\nLoading preprocessing models (InsightFace)...")
    import preprocess  # noqa: F401 — triggers InsightFace load
    print("Loading background models (BiSeNet + D3DFR)...")
    import retrieve_bg  # noqa: F401 — triggers BiSeNet + D3DFR load

    pipeline, prompt_embeds = load_pipeline()
    process_all_variations(pipeline, prompt_embeds, args.identities, force=args.force)
    print("\nAll done!")


if __name__ == '__main__':
    main()
