import gc
import hashlib
import os
import queue
import threading
import warnings

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm

import re
import random


def run_mdx(model_params, output_dir, model_name, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process the audio
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    roformer_output_format = 'wav'
    
    print(f"output_dir: {output_dir}")
    prompt = f'audio-separator "{filename}" --model_filename {model_name} --output_dir="{output_dir}" --output_format={roformer_output_format} --normalization=0.9'
    os.system(prompt)

    vocals_file = f"{base_name}_Vocals.wav"
    instrumental_file = f"{base_name}_Instrumental.wav"

    main_filepath = None
    invert_filepath = None

    if not exclude_main:
        main_filepath = os.path.join(output_dir, vocals_file)
        if os.path.exists(os.path.join(output_dir, f"{base_name}_(Vocals)_{model_name.replace('.9755.ckpt', '')}.wav")):
            os.rename(os.path.join(output_dir, f"{base_name}_(Vocals)_{model_name.replace('.9755.ckpt', '')}.wav"), main_filepath)

    if not exclude_inversion:
        invert_filepath = os.path.join(output_dir, instrumental_file)
        if os.path.exists(os.path.join(output_dir, f"{base_name}_(Instrumental)_{model_name.replace('.9755.ckpt', '')}.wav")):
            os.rename(os.path.join(output_dir, f"{base_name}_(Instrumental)_{model_name.replace('.9755.ckpt', '')}.wav"), invert_filepath)

    if not keep_orig:
        os.remove(filename)

    return main_filepath, invert_filepath
