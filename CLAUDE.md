# MagicFace — CLAUDE.md

## Project Overview

MagicFace is a facial expression editing model conditioned on Facial Action Units (AUs). Given a source face image and a background image, the model generates a new image with the specified expression changes. It is based on Stable Diffusion v1-5 with two custom UNets: an ID encoder and a denoising UNet.

Paper: https://arxiv.org/pdf/2501.02260  
HuggingFace weights: `mengtingwei/magicface`

---

## Architecture

- **ID UNet** (`mgface/pipelines_mgface/unet_ID_2d_condition.py`) — encodes identity/appearance from the source image.
- **Denoising UNet** (`mgface/pipelines_mgface/unet_deno_2d_condition.py`) — diffusion denoiser conditioned on AU vector + identity features.
- **Pipeline** (`mgface/pipelines_mgface/pipeline_mgface.py`) — `MgPipeline`, wraps VAE + CLIP text encoder + both UNets. Extends the diffusers `StableDiffusionPipeline` interface.
- **BiSeNet** (`utils/model.py`) — face parsing network used to isolate background.
- **D3DFR ResNet50** (`third_party/d3dfr/`) — 3D face reconstruction used to extract 68 facial landmarks for background generation.

---

## AU Mapping

12 AUs supported, indexed 0–11:

| AU | Index | Description |
|----|-------|-------------|
| AU1 | 0 | Inner Brow Raiser |
| AU2 | 1 | Outer Brow Raiser |
| AU4 | 2 | Brow Lowerer |
| AU5 | 3 | Upper Lid Raiser |
| AU6 | 4 | Cheek Raiser |
| AU9 | 5 | Nose Wrinkler |
| AU12 | 6 | Lip Corner Puller |
| AU15 | 7 | Lip Corner Depressor |
| AU17 | 8 | Chin Raiser |
| AU20 | 9 | Lip Stretcher |
| AU25 | 10 | Lips Part |
| AU26 | 11 | Jaw Drop |

AU variation values are integers (typically 1–5), representing intensity.

---

## Pipeline: 3-Step Workflow

### Step 1 — Preprocess (crop face)
```bash
python utils/preprocess.py --img_path <input.jpg> --save_path <cropped.png>
```
Uses InsightFace `antelopev2` to detect the face, crops and warps to 512×512.

### Step 2 — Retrieve Background
```bash
python utils/retrieve_bg.py --img_path <input.jpg> --save_path <bg.png>
```
Uses BiSeNet (79999_iter.pth) for face parsing + D3DFR for landmark extraction. Outputs a background + landmark overlay image.

### Step 3 — Inference
```bash
python inference.py \
  --img_path <cropped.png> \
  --bg_path <bg.png> \
  --au_test AU12 \
  --AU_variation 5 \
  --saved_path edited_images/
```
For multiple AUs, use `+` as separator:
```bash
--au_test AU1+AU2+AU5 --AU_variation 5+5+5
```

---

## Batch Processing

**Primary workflow — double-click on Windows:**

| File | What it does |
|------|-------------|
| `run_angry.bat` | Angry expression, both categories (50 × 2 = 100 images) |
| `run_fearful.bat` | Fearful expression, both categories (50 × 2 = 100 images) |

Both `.bat` files activate the `magicface` conda env and run `run_batch.py`.

**`run_batch.py`** loads all models once (much faster than the old per-image subprocess approach) and processes every image in a loop. It skips already-completed images, so it's safe to re-run after an interruption.

Advanced usage from command line:
```bash
# One expression, one category
python run_batch.py --expression angry --categories dominant

# Both expressions, both categories in one shot
python run_batch.py --expression angry fearful --categories dominant submissive
```

**Notebooks** (`batch_process.ipynb`, `batch_process_fear.ipynb`, `process.ipynb`) are kept as reference / for spot-checking individual images. They still use subprocess per image and are slower.

### Source Directories (relative to `magic_face/`)

| Category | Source |
|----------|--------|
| Dominant | `processed_identities/processed_dominant_identities/` |
| Submissive | `processed_identities/processed_submissive_identities/` |

### Output Structure

```
magic_face/
  edited_images/
    dominant/        ← angry dominant faces
    submissive/      ← angry submissive faces
  edited_images_fear/
    dominant/        ← fearful dominant faces
    submissive/      ← fearful submissive faces
  test_images/
    dominant/        ← intermediate cropped + bg files
    submissive/      ← intermediate cropped + bg files
```

### Model Behavior Notes (learned from empirical testing)

The MagicFace paper only demonstrates two expressions in training/evaluation:
- **Happiness**: AU6+AU12
- **Sadness**: AU1+AU4+AU15

This means AU4 is learned as a **sadness AU** by the model. Applying AU4 without AU5
causes the model to generate sadness features including drooping/closed eyelids — even
though FACS theory says AU4 should only lower the brow.

**AU5 is required alongside AU4** to counteract the sadness-eye-droop artifact and keep
eyes naturally open. The intensity of AU5 controls the balance:
- AU5 at 5 → eyes too wide, reads as fearful
- AU5 at 2 → eyes naturally open, reads as angry

AUs that cause problems in this model (do not use):
- **AU9** (Nose Wrinkler) — activates orbicularis oculi, fully closes eyes
- **AU17** (Chin Raiser) — causes unnatural downward facial drag
- **AU15** alone with AU4 — triggers full sadness prototype (closed eyes)
- **AU25** / **AU26** — open the mouth

### AU Profiles

| Expression | AUs | Variations | Notes |
|------------|-----|------------|-------|
| Angry | `AU4+AU5+AU25+AU12+AU2` | `5+4+-2+-2+-1` | AU4 lowers inner brow; AU5 keeps eyes open; AU25=-2 closes lips; AU12=-2 tightens lip corners (approximates AU23); AU2=-1 lowers outer brows (distinguishes angry from confused). NOTE: AU2=-2 causes severe eye distortion — keep at -1 max. |
| Fearful | `AU1+AU2+AU4+AU5+AU20+AU25` | `5+5+5+5+3+-2` | Fear brow shape + wide eyes + lip stretcher (reduced); AU25=-2 closes mouth |

**Mouths must stay closed.** AU25 and AU26 are always excluded.

---

## Required Model Weights

Download from HuggingFace before running:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="mengtingwei/magicface", local_dir="./")
```

Key files needed:
- `79999_iter.pth` — BiSeNet face parsing weights (place in project root)
- `checkpoints/third_party/d3dfr_res50_nofc.pth` — D3DFR ResNet50
- `checkpoints/third_party/BFM_model_front.mat` — Basel Face Model
- `ID_enc/` — ID UNet safetensors
- `denoising_unet/` — Denoising UNet safetensors
- `third_party_files/models/antelopev2/` — InsightFace model files

---

## Dependencies

```
torch==2.1.1+cu118
torchvision==0.16.1+cu118
diffusers==0.25.1
transformers==4.42.4
insightface==0.7.3
huggingface-hub==0.25.2
einops==0.8.0
opencv-python==4.10.0.84
Pillow==11.1.0
scipy==1.15.0
```

GPU (CUDA) is required — `device = 'cuda'` is hardcoded in `inference.py` and `utils/retrieve_bg.py`.

---

## .gitignore Notes

Images, weights, and compiled files are excluded from git:
- `*.jpg`, `*.png`, `*.pth`, `*.safetensors`, `*.onnx`
- `test_images/`, `edited_images/`, `edited_images_fear/`
- HuggingFace cache dirs (`models--*`)
