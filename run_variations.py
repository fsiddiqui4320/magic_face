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
