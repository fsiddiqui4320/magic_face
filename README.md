# MagicFace — CASPER Lab Expression Editing Pipeline

MagicFace is a facial expression editing model conditioned on Facial Action Units (AUs). Given a neutral face image, it generates a new version of that face with a specified expression — without changing identity, lighting, or background. This repo contains the CASPER Lab pipeline for generating angry, fearful, happy, and sad versions of face stimuli across dominance categories.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Download Model Weights](#download-model-weights)
4. [Source Data](#source-data)
5. [Running Batch Processing](#running-batch-processing)
6. [Running Variation Testing](#running-variation-testing)
7. [Output Structure](#output-structure)
8. [AU Profiles Reference](#au-profiles-reference)

---

## Prerequisites

- **CUDA GPU** — required (inference is GPU-only)
- **[Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)**
- **Git**

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/fsiddiqui4320/magic_face.git
cd magic_face
```

**2. Create and activate the conda environment**

```bash
conda create -n magicface python=3.10 -y
conda activate magicface
```

**3. Install PyTorch with CUDA 11.8**

```bash
pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**4. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## Download Model Weights

Model weights are hosted on HuggingFace and must be downloaded before running anything. This downloads ~10 GB.

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="mengtingwei/magicface", local_dir="./")
```

Run this once from the `magic_face/` directory with the `magicface` env active. It will populate:

| File/Folder | Purpose |
|-------------|---------|
| `79999_iter.pth` | BiSeNet face parser — isolates background |
| `checkpoints/third_party/d3dfr_res50_nofc.pth` | 3D face reconstruction for landmark extraction |
| `checkpoints/third_party/BFM_model_front.mat` | Basel Face Model |
| `ID_enc/` | Identity encoder UNet weights |
| `denoising_unet/` | Expression denoising UNet weights |
| `third_party_files/models/antelopev2/` | InsightFace face detection model |

---

## Source Data

Place input face images (`.jpg`) into the appropriate subdirectory:

```
magic_face/
  processed_identities/
    processed_dominant_identities/    ← dominant face images (e.g. WM_001.jpg)
    processed_submissive_identities/  ← submissive face images (e.g. WM_002.jpg)
```

Images should be clear frontal photos with a neutral expression. The pipeline will automatically crop, align, and extract backgrounds before editing.

---

## Running Batch Processing

`run_batch.py` processes a full set of faces for a given expression. It loads all models once and loops through every image in a category, making it significantly faster than running images individually.

**Generate angry faces:**
```bash
conda activate magicface
python run_batch.py --expression angry --categories dominant submissive
```

**Generate fearful faces:**
```bash
python run_batch.py --expression fearful --categories dominant submissive
```

**Process a single category:**
```bash
python run_batch.py --expression angry --categories dominant
```

**Limit number of images (e.g. for testing):**
```bash
python run_batch.py --expression angry --n_images 5
```

**Reprocess images that already exist:**
```bash
python run_batch.py --expression angry --force
```

The script prints the AU vector before processing so you can verify the expression profile being applied.

---

## Running Variation Testing

`run_variations.py` generates multiple AU intensity variations for two specific test identities (one dominant, one submissive). This is used to evaluate and compare different expression profiles before committing to a full batch.

**Run all variations for both test identities:**
```bash
conda activate magicface
python run_variations.py
```

**Run for a single identity:**
```bash
python run_variations.py --identities WM_136
```

**Reprocess existing outputs:**
```bash
python run_variations.py --force
```

This generates 12 images per identity (4 emotions × 3 variations each), covering happy, sad, angry, and fearful at low/medium/high intensities.

---

## Output Structure

```
magic_face/
  edited_images/
    angry/
      dominant/         ← angry versions of dominant faces
      submissive/       ← angry versions of submissive faces
    fearful/
      dominant/         ← fearful versions of dominant faces
      submissive/       ← fearful versions of submissive faces

  variation_test/
    WM_136/             ← dominant test identity
      happy1.png        ← subtle happy
      happy2.png        ← medium happy
      happy3.png        ← full happy (closed mouth guard)
      sad1.png
      sad2.png
      sad3.png
      angry1.png        ← current locked profile
      angry2.png        ← reduced lip frown
      angry3.png        ← pure brow anger
      fearful1.png      ← no AU2 (oblique fear brow)
      fearful2.png      ← minimal AU2
      fearful3.png      ← maximum AU4 (strongest fear signal)
    WM_344/             ← submissive test identity
      (same structure)

  test_images/          ← intermediate files (auto-generated, not needed after processing)
    dominant/
    submissive/
```

---

## AU Profiles Reference

MagicFace edits expressions by setting Action Unit (AU) intensities. The pipeline supports 12 AUs:

| AU | Description |
|----|-------------|
| AU1 | Inner Brow Raiser |
| AU2 | Outer Brow Raiser |
| AU4 | Brow Lowerer |
| AU5 | Upper Lid Raiser |
| AU6 | Cheek Raiser |
| AU12 | Lip Corner Puller (smile) |
| AU15 | Lip Corner Depressor |
| AU20 | Lip Stretcher |
| AU25 | Lips Part |
| AU26 | Jaw Drop |

Values range from -10 to +10. Negative values invert the action (e.g. AU12=-2 tightens lip corners instead of pulling them into a smile).

### Expression Profiles (Batch Processing)

| Expression | AUs | Values | Notes |
|------------|-----|--------|-------|
| Angry | `AU4+AU5+AU25+AU12` | `5+4+-2+-2` | AU4 lowers brow; AU5 keeps eyes open; AU25=-2 closes mouth; AU12=-2 tightens lip corners |
| Fearful | `AU1+AU2+AU4+AU5+AU20+AU25` | `2+2+2+2+3+-2` | Fear brow shape + wide eyes + lip stretcher; AU25=-2 closes mouth |

### Variation Profiles (run_variations.py)

| Variation | AUs | Values |
|-----------|-----|--------|
| happy1 | `AU12+AU6` | `3+2` |
| happy2 | `AU12+AU6` | `5+4` |
| happy3 | `AU12+AU6+AU25` | `5+5+-2` |
| sad1 | `AU1+AU4+AU15` | `3+3+2` |
| sad2 | `AU1+AU4+AU15` | `5+5+3` |
| sad3 | `AU1+AU4+AU15` | `5+5+5` |
| angry1 | `AU4+AU5+AU25+AU12` | `5+4+-2+-2` |
| angry2 | `AU4+AU5+AU25+AU12` | `5+4+-2+-1` |
| angry3 | `AU4+AU5+AU25` | `5+4+-2` |
| fearful1 | `AU1+AU4+AU5+AU20+AU25` | `4+4+4+2+-2` |
| fearful2 | `AU1+AU2+AU4+AU5+AU20+AU25` | `4+1+4+3+2+-2` |
| fearful3 | `AU1+AU4+AU5+AU20+AU25` | `5+5+4+3+-2` |

### Hard Constraints

These AUs must never be used — they produce artifacts in this model:

| AU | Why |
|----|-----|
| AU9 (Nose Wrinkler) | Fully closes eyes |
| AU17 (Chin Raiser) | Causes unnatural downward facial drag |
| AU25/AU26 positive | Opens the mouth — all stimuli must have closed mouths |
| AU2 ≤ -2 | Causes severe eye distortion |

### Key Model Behavior

**AU4 is treated as a sadness AU.** MagicFace was trained with sadness defined as AU1+AU4+AU15, so the model interprets AU4 as part of a sadness prototype. Applying AU4 alone causes the eyes to droop or close. **AU5 must always accompany AU4** in angry and fearful profiles to counteract this effect and keep eyes naturally open.
