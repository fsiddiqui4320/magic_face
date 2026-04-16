import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

# Import the constants we're about to write
from run_variations import IND_DICT, VARIATIONS, IDENTITIES, make_au_vector

FORBIDDEN_AUS = {'AU9', 'AU17'}

def _all_profiles():
    """Yield (emotion, n, profile) for every variation."""
    for emotion, profiles in VARIATIONS.items():
        for n, profile in enumerate(profiles, 1):
            yield emotion, n, profile


def test_all_au_names_are_valid():
    """Every AU name referenced in VARIATIONS must exist in IND_DICT."""
    for emotion, n, profile in _all_profiles():
        for au in profile['aus'].split('+'):
            assert au in IND_DICT, f"{emotion}{n}: unknown AU '{au}'"


def test_no_forbidden_aus():
    """AU9 (closes eyes) and AU17 (face drag) must never appear."""
    for emotion, n, profile in _all_profiles():
        for au in profile['aus'].split('+'):
            assert au not in FORBIDDEN_AUS, f"{emotion}{n}: forbidden AU '{au}'"


def test_au25_au26_never_positive():
    """AU25 and AU26 open the mouth — must always be <= 0 when present."""
    for emotion, n, profile in _all_profiles():
        aus = profile['aus'].split('+')
        vals = [int(v) for v in profile['variations'].split('+')]
        for au, val in zip(aus, vals):
            if au in ('AU25', 'AU26'):
                assert val <= 0, f"{emotion}{n}: {au}={val} opens mouth"


def test_au2_never_below_minus_one():
    """AU2 at -2 or below causes eye distortion."""
    for emotion, n, profile in _all_profiles():
        aus = profile['aus'].split('+')
        vals = [int(v) for v in profile['variations'].split('+')]
        for au, val in zip(aus, vals):
            if au == 'AU2':
                assert val >= -1, f"{emotion}{n}: AU2={val} causes eye distortion"


def test_make_au_vector_shape_and_values():
    """make_au_vector returns a length-12 array with correct indices set."""
    vec = make_au_vector('AU12+AU6', '5+4')
    assert vec.shape == (12,)
    assert vec[IND_DICT['AU12']] == 5
    assert vec[IND_DICT['AU6']] == 4
    assert vec[IND_DICT['AU1']] == 0   # untouched AU stays 0


def test_make_au_vector_negative_values():
    vec = make_au_vector('AU25+AU12', '-2+-1')
    assert vec[IND_DICT['AU25']] == -2
    assert vec[IND_DICT['AU12']] == -1


def test_identities_categories_are_valid():
    """Each identity must map to 'dominant' or 'submissive'."""
    for wm_id, category in IDENTITIES.items():
        assert category in ('dominant', 'submissive'), \
            f"{wm_id} has invalid category '{category}'"
