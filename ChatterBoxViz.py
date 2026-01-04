import random
from typing import Optional, Tuple, List, Set, Dict, Any
import gradio as gr
import numpy as np
import torchaudio as ta
import torch
import nltk
import re
import os
import tempfile
import time
import traceback
import json
import shutil
import sys
import hashlib
import contextlib
import logging
from pathlib import Path
from functools import lru_cache
from huggingface_hub import hf_hub_download
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

# Silence HF warnings related to SDPA/output_attentions and legacy cache.
# Upstream fix pending in chatterbox PR #398 (https://github.com/resemble-ai/chatterbox/pull/398).
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

fastapi_app = FastAPI()
class TTSRequest(BaseModel):
    text: str

import platform, subprocess

def play_audio_file(path: str):
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", path])
        elif system == "Linux":
            subprocess.run(["aplay", path])
        elif system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"Could not play audio: {e}")

@fastapi_app.post("/api/gui-tts")
def gui_tts(req: TTSRequest):
    try:
        gen = process_text_to_speech(
            long_text_input=req.text,
            audio_prompt_file_path=current_settings["audio_prompt_file_path"],

            # base
            cfg_weight_input=current_settings["cfg_weight_input"],
            exaggeration_input=current_settings["exaggeration_input"],

            # turbo
            top_p_input=current_settings["top_p_input"],
            top_k_input=current_settings["top_k_input"],
            repetition_penalty_input=current_settings["repetition_penalty_input"],
            norm_loudness_input=current_settings["norm_loudness_input"],

            # shared
            temperature_input=current_settings["temperature_input"],
            target_chars_input=current_settings["target_chars_input"],
            max_chars_input=current_settings["max_chars_input"],
            device_choice=current_settings["device_choice"],
            seed_num=current_settings["seed_num"],
            model_choice=current_settings["model_choice"],
        )

        last_result = None
        for update in gen:
            last_result = update

        audio_path = last_result[0] if last_result and last_result[0] else current_settings["output_filepath"]

        if audio_path:
            play_audio_file(audio_path)
            return {"success": True, "path": audio_path}
        else:
            return {"success": False, "error": "No audio file produced"}
    except Exception as e:
        return {"success": False, "error": str(e)}





# Run FastAPI in a separate thread when Gradio starts
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=5001)

current_model_kind_loaded = None
# Pre-compile regex patterns for better performance
WHITESPACE_REGEX = re.compile(r'\s+')
NON_WORD_CHARS = re.compile(r'[^\w\s-]')
current_settings = {
    "audio_prompt_file_path": None,
    "cfg_weight_input": None,
    "exaggeration_input": None,
    "temperature_input": None,
    "target_chars_input": None,
    "max_chars_input": None,
    "device_choice": None,
    "seed_num": None,
    "output_filepath": None,
}

def update_setting(key):
    def _inner(value):
        current_settings[key] = value
    return _inner


# Convert strings to sets for O(1) lookups
PUNCTUATION_SPLIT_STRONGLY = {".", "!", "?"}
PUNCTUATION_SPLIT_WEAKLY = {",", ";", ":"}
PUNCTUATION_ALL = PUNCTUATION_SPLIT_STRONGLY.union(PUNCTUATION_SPLIT_WEAKLY)

# Cache for directory listings and file existence
_dir_cache = {}
_file_cache = {}
_settings_cache = {}

if not os.path.exists("outputs"):
    os.makedirs("outputs")
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
APP_DATA_DIR = "app_data"
SETTINGS_FILE = os.path.join(APP_DATA_DIR, "settings.json")
PERSISTENT_PROMPT_FILENAME = "last_used_prompt.wav"
PERSISTENT_PROMPT_PATH = os.path.join(APP_DATA_DIR, PERSISTENT_PROMPT_FILENAME)

SETTINGS_DIR = os.path.join(APP_DATA_DIR, "settings_profiles")
SETTINGS_PROMPTS_DIR = os.path.join(SETTINGS_DIR, "prompts")

# Audio validation constants
MAX_AUDIO_DURATION_SECONDS = 30
MIN_AUDIO_DURATION_SECONDS = 0.5
SUPPORTED_SAMPLE_RATES = [16000, 22050, 24000, 44100, 48000]
MAX_PROMPT_AUDIO_FILE_SIZE_MB = 100  # Added for clarity
MIN_SAVED_AUDIO_FILE_SIZE_BYTES = 1024  # For output verification

MAX_TEXT_LENGTH = 50000
MIN_CHUNK_SIZE = 10
MAX_OUTPUT_FILE_SIZE_MB = 500  # Will be used to warn user


# --- Context Manager for Suppressing Output ---
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Safely suppress stdout/stderr with proper error handling"""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    devnull_out = None
    devnull_err = None

    try:
        # Use 'w' without encoding for os.devnull
        devnull_out = open(os.devnull, 'w')
        devnull_err = open(os.devnull, 'w')
        sys.stdout = devnull_out
        sys.stderr = devnull_err
        yield
    except Exception as e:
        logger.error(f"Error in suppress_stdout_stderr: {e}")
        # Restore original streams immediately if an error occurs within the try block
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        raise  # Re-raise the exception
    finally:
        # Restore original streams first
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Then safely close devnull files
        for devnull_file in [devnull_out, devnull_err]:
            if devnull_file and not devnull_file.closed:
                try:
                    devnull_file.flush()
                    devnull_file.close()
                except Exception as e:
                    logger.warning(f"Error closing devnull file: {e}")


# --- Global Model State & NLTK ---
tts_model = None
current_device_loaded = None


def calculate_file_hash(filepath: str, hash_algo: str = "md5") -> Optional[str]:
    """Calculate the hash of a file."""
    if not filepath or not os.path.exists(filepath):
        logger.error(f"Cannot calculate hash: File not found at '{filepath}'")
        return None

    hasher = hashlib.new(hash_algo)
    try:
        with open(filepath, 'rb') as f:
            while True:
                buf = f.read(65536)  # Read in 64k chunks
                if not buf:
                    break
                hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Could not calculate {hash_algo} hash for {filepath}: {e}")
        return None


def setup_nltk():
    """Setup NLTK with proper error handling"""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer found")
    except (nltk.downloader.DownloadError, LookupError):
        try:
            logger.info("NLTK punkt tokenizer not found. Downloading...")
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt tokenizer downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download NLTK punkt tokenizer: {e}")
            raise gr.Error("Failed to setup text processing. Please check your internet connection and try again.")


# Initialize NLTK
setup_nltk()
sent_tokenize = nltk.sent_tokenize


# --- Validation Functions ---
def validate_audio_file(file_path: str) -> tuple[bool, str]:
    """Validate audio file format, duration, and sample rate"""
    if not file_path or not os.path.exists(file_path):
        return False, "Audio file not found"

    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_PROMPT_AUDIO_FILE_SIZE_MB:
            return False, f"Audio file too large ({file_size_mb:.1f}MB). Maximum size is {MAX_PROMPT_AUDIO_FILE_SIZE_MB}MB."

        # Load and validate audio
        waveform, sample_rate = ta.load(file_path)

        # Check duration
        duration = waveform.shape[1] / sample_rate
        if duration < MIN_AUDIO_DURATION_SECONDS:
            return False, f"Audio too short ({duration:.1f}s). Minimum duration is {MIN_AUDIO_DURATION_SECONDS}s."
        if duration > MAX_AUDIO_DURATION_SECONDS:
            return False, f"Audio too long ({duration:.1f}s). Maximum duration is {MAX_AUDIO_DURATION_SECONDS}s."

        # Check channels (convert to mono if stereo)
        if waveform.shape[0] > 2:
            return False, f"Audio has too many channels ({waveform.shape[0]}). Maximum is 2 (stereo)."
        # Note: ChatterboxTTS will handle conversion to mono if needed during its processing.

        # Check sample rate
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            logger.warning(f"Unusual sample rate: {sample_rate}Hz for prompt. Supported rates: {SUPPORTED_SAMPLE_RATES}. Model will resample.")
            # Chatterbox will resample to its internal SR, so this is more of a warning.

        return True, f"Valid audio prompt: {duration:.1f}s, {sample_rate}Hz, {waveform.shape[0]} channel(s)"

    except Exception as e:
        return False, f"Error reading audio file: {str(e)}"


def validate_text_input(text: str) -> tuple[bool, str]:
    """Validate text input"""
    if not text or not text.strip():
        return False, "Text input cannot be empty"

    text = text.strip()
    if len(text) < 3:
        return False, "Text too short. Please enter at least 3 characters."

    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_LENGTH} characters."

    # Check for unusual characters that might break processing
    if '\x00' in text:
        return False, "Text contains null characters which are not supported"

    return True, f"Valid text: {len(text)} characters"


def validate_parameters(cfg_weight: float, exaggeration: float, temperature: float,
                        target_chars: int, max_chars: int) -> tuple[bool, str]:
    """Validate generation parameters"""
    if not (0.0 <= cfg_weight <= 1.0):
        return False, f"CFG weight must be between 0.0 and 2.0 (got {cfg_weight})"

    if not (0 <= exaggeration <= 1000):
        return False, f"Exaggeration must be between 0.0 and 2.0 (got {exaggeration})"

    if not (0.05 <= temperature <= 2.0):
        return False, f"Temperature must be between 0.05 and 2.0 (got {temperature})"

    if not (MIN_CHUNK_SIZE <= target_chars <= 500):  # Max target chars can be adjusted
        return False, f"Target chars must be between {MIN_CHUNK_SIZE} and 500 (got {target_chars})"

    if not (target_chars < max_chars <= 1000):  # Max max_chars can be adjusted
        return False, f"Max chars must be greater than target chars and ≤ 1000 (got {max_chars})"

    return True, "Parameters valid"


def check_disk_space(required_mb: float = 100) -> tuple[bool, str]:
    """Check available disk space (cross-platform)."""
    try:
        # Determine path to check: APP_DATA_DIR or current dir if it doesn't exist yet.
        # For shutil.disk_usage, it's better to check a directory that is expected to exist.
        # If APP_DATA_DIR might not exist yet, default to current directory for the check.
        check_path = APP_DATA_DIR
        if not os.path.exists(check_path) or not os.path.isdir(check_path):
            check_path = '.'  # Check current working directory if APP_DATA_DIR is not valid yet
            logger.info(f"APP_DATA_DIR ('{APP_DATA_DIR}') not found or not a directory, checking disk space for current directory ('{os.path.abspath(check_path)}').")

        # shutil.disk_usage returns a named tuple with total, used, and free bytes.
        # 'free' is what's available to the user (might be less than total - used for non-superusers).
        usage = shutil.disk_usage(check_path)
        available_mb = usage.free / (1024 * 1024)

        if available_mb < required_mb:
            return False, f"Insufficient disk space in '{os.path.abspath(check_path)}'. Required: {required_mb:.1f}MB, Available: {available_mb:.1f}MB"

        return True, f"Sufficient disk space: {available_mb:.1f}MB available in '{os.path.abspath(check_path)}'"
    except AttributeError:
        # This might catch issues if shutil.disk_usage is somehow unavailable (very rare on Python 3.3+)
        logger.warning("shutil.disk_usage not available (unexpected). Skipping disk space check.")
        return True, "Disk space check skipped (shutil.disk_usage unavailable)"
    except FileNotFoundError:
        # If the check_path itself becomes invalid between os.path.exists and shutil.disk_usage (race condition, rare)
        # or if '.' is somehow not a valid path (extremely rare).
        logger.warning(f"Path '{check_path}' not found during disk space check. Skipping check.")
        return True, "Disk space check skipped (path not found)"
    except Exception as e:
        logger.warning(f"Could not check disk space using shutil.disk_usage: {e}")
        return True, "Disk space check skipped (warning)"


# --- Helper Functions ---
def set_seed(seed: int):
    """Set random seed with validation"""
    try:
        seed = int(seed)
        if not (0 <= seed <= 2 ** 32 - 1):  # Common range for seeds
            raise ValueError(f"Seed must be between 0 and 2**32 - 1, got {seed}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Global seed set to: {seed}")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        # Not raising gr.Error as it might be non-critical for some operations,
        # but it's important for reproducibility. The calling function should handle UI.
        raise


def ensure_app_data_dir():
    """Create all necessary app data directories with error handling"""
    try:
        # Create base, settings, and prompts directories
        for path_str in [APP_DATA_DIR, SETTINGS_DIR, SETTINGS_PROMPTS_DIR]:
            path_obj = Path(path_str)
            path_obj.mkdir(parents=True, exist_ok=True)

        # Test write permissions in the base directory
        test_file = Path(APP_DATA_DIR) / '.write_test'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"No write permission to {APP_DATA_DIR}: {e}")

        logger.info(f"App data directories ready: {APP_DATA_DIR}, {SETTINGS_DIR}, {SETTINGS_PROMPTS_DIR}")
    except Exception as e:
        logger.error(f"Error setting up app data directories: {e}")
        raise gr.Error(f"Cannot create/access application data directories: {e}. Please check permissions.")


def save_settings(
    device, cfg, exg, temp,
    long_text, target_chars, max_chars,
    seed_value,
    prompt_successfully_persisted_this_run,
    model_choice, top_p, top_k, repetition_penalty, norm_loudness
):
    """Save settings only if they have changed, with atomic write and conditional backup."""
    temp_settings_file_path = None

    try:
        ensure_app_data_dir()

        # Normalize + validate model choice
        model_choice = (model_choice or "turbo").lower()
        if model_choice not in ("base", "turbo"):
            model_choice = "turbo"

        new_settings = {
            "device": device,

            # Base params (still saved even if turbo; harmless + keeps backward compat)
            "cfg_weight": float(cfg),
            "exaggeration": float(exg),

            # Shared
            "temperature": float(temp),
            "long_text": str(long_text)[:50000],
            "target_chars_per_chunk": int(target_chars),
            "max_chars_per_chunk": int(max_chars),
            "seed": int(seed_value),

            # New: model selection
            "model_choice": model_choice,

            # New: turbo params
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "norm_loudness": bool(norm_loudness),

            "has_persistent_prompt": prompt_successfully_persisted_this_run and os.path.exists(PERSISTENT_PROMPT_PATH),
            "last_saved_timestamp": time.time(),
        }

        # Compare without timestamp
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f_old:
                    existing_settings = json.load(f_old)

                existing_cmp = dict(existing_settings)
                existing_cmp.pop("last_saved_timestamp", None)

                new_cmp = dict(new_settings)
                new_cmp.pop("last_saved_timestamp", None)

                if existing_cmp == new_cmp:
                    logger.info(f"Settings have not changed. Skipping save to {SETTINGS_FILE}.")
                    return
            except json.JSONDecodeError:
                logger.warning(f"Could not decode existing settings file '{SETTINGS_FILE}' for comparison. Will overwrite.")
            except Exception as e_read:
                logger.warning(f"Error reading existing settings file '{SETTINGS_FILE}' for comparison: {e_read}. Will proceed to save/overwrite.")

        logger.info(f"Settings have changed or no existing valid settings. Proceeding to save to {SETTINGS_FILE}.")

        fd, temp_settings_file_path = tempfile.mkstemp(
            suffix=".json", prefix="settings_tmp_", dir=APP_DATA_DIR
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f_temp:
            json.dump(new_settings, f_temp, indent=4, ensure_ascii=False)

        # Backup before overwrite
        if os.path.exists(SETTINGS_FILE):
            backup_file = f"{SETTINGS_FILE}.{int(time.time())}.backup"
            try:
                shutil.copy2(SETTINGS_FILE, backup_file)
                logger.info(f"Created settings backup: {backup_file}")
            except Exception as e_backup:
                logger.warning(f"Could not create settings backup for '{SETTINGS_FILE}': {e_backup}")

        shutil.move(temp_settings_file_path, SETTINGS_FILE)
        temp_settings_file_path = None
        logger.info(f"Settings saved successfully to {SETTINGS_FILE}")

    except Exception as e:
        logger.error(f"Error saving settings: {e}\n{traceback.format_exc()}")
    finally:
        if temp_settings_file_path and os.path.exists(temp_settings_file_path):
            try:
                os.remove(temp_settings_file_path)
                logger.info(f"Cleaned up temporary settings file: {temp_settings_file_path}")
            except Exception as e_clean:
                logger.warning(f"Could not remove temporary settings file {temp_settings_file_path}: {e_clean}")

        cleanup_old_backups(APP_DATA_DIR, SETTINGS_FILE, keep_latest_n=10)

def cleanup_old_backups(directory: str, base_filename: str, keep_latest_n: int):
    try:
        base_name = os.path.basename(base_filename)
        backups = sorted(
            [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(".backup")],
            key=lambda f_name: os.path.getmtime(os.path.join(directory, f_name))
        )

        if len(backups) > keep_latest_n:
            for old_backup in backups[:-keep_latest_n]:  # Keep the N newest ones
                try:
                    os.remove(os.path.join(directory, old_backup))
                    logger.info(f"Removed old settings backup: {old_backup}")
                except Exception as e_remove:
                    logger.warning(f"Could not remove old backup {old_backup}: {e_remove}")
    except Exception as e:
        logger.warning(f"Error during backup cleanup: {e}")

def load_settings():
    """Load settings with comprehensive error handling and validation."""
    defaults = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Base defaults
        "cfg_weight": 0.2,
        "exaggeration": 0.8,

        # Shared
        "temperature": 0.8,
        "long_text": "",
        "target_chars_per_chunk": 100,
        "max_chars_per_chunk": 200,
        "seed": 0,

        # New
        "model_choice": "turbo",
        "top_p": 0.95,
        "top_k": 1000,
        "repetition_penalty": 1.2,
        "norm_loudness": True,

        "audio_prompt_path": None,
    }

    try:
        ensure_app_data_dir()

        if not os.path.exists(SETTINGS_FILE):
            logger.info(f"No settings file found at {SETTINGS_FILE}, using defaults.")
            return defaults

        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            settings = json.load(f)

        loaded_values = {}

        for key, default_value in defaults.items():
            if key not in settings:
                loaded_values[key] = default_value
                continue

            value = settings.get(key)
            try:
                if key in ["cfg_weight", "exaggeration", "temperature", "top_p", "repetition_penalty"]:
                    loaded_values[key] = float(value)

                elif key in ["target_chars_per_chunk", "max_chars_per_chunk", "seed", "top_k"]:
                    loaded_values[key] = int(value)

                elif key == "norm_loudness":
                    loaded_values[key] = bool(value)

                elif key == "model_choice":
                    v = str(value).lower()
                    loaded_values[key] = v if v in ("base", "turbo") else "turbo"

                elif key == "device":
                    v = str(value).lower()
                    if v not in ["cuda", "cpu"]:
                        logger.warning(f"Invalid device '{value}' in settings, using default '{default_value}'.")
                        loaded_values[key] = default_value
                    elif v == "cuda" and not torch.cuda.is_available():
                        logger.warning("CUDA specified in settings but not available, switching to CPU.")
                        loaded_values[key] = "cpu"
                    else:
                        loaded_values[key] = v

                elif key == "long_text":
                    loaded_values[key] = str(value)

                elif key == "audio_prompt_path":
                    # handled below
                    loaded_values[key] = None

                else:
                    loaded_values[key] = value

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid value for '{key}' in settings ('{value}'), using default '{default_value}'. Error: {e}"
                )
                loaded_values[key] = default_value

        # Clamp a few things to safe ranges (optional but avoids weird saved values)
        loaded_values["top_p"] = max(0.0, min(1.0, float(loaded_values["top_p"])))
        loaded_values["top_k"] = max(0, min(1000, int(loaded_values["top_k"])))
        loaded_values["repetition_penalty"] = max(1.0, min(2.0, float(loaded_values["repetition_penalty"])))

        # Handle persistent prompt (same as your current logic)
        if settings.get("has_persistent_prompt", False) and os.path.exists(PERSISTENT_PROMPT_PATH):
            is_valid, msg = validate_audio_file(PERSISTENT_PROMPT_PATH)
            if is_valid:
                loaded_values["audio_prompt_path"] = PERSISTENT_PROMPT_PATH
                logger.info(f"Loaded persistent audio prompt: {PERSISTENT_PROMPT_PATH} ({msg})")
            else:
                logger.warning(f"Persistent prompt '{PERSISTENT_PROMPT_PATH}' is invalid ({msg}). Removing it.")
                try:
                    os.remove(PERSISTENT_PROMPT_PATH)
                except Exception as e:
                    logger.error(f"Could not remove invalid persistent prompt '{PERSISTENT_PROMPT_PATH}': {e}")
                loaded_values["audio_prompt_path"] = None
        else:
            loaded_values["audio_prompt_path"] = None

        logger.info("Settings loaded successfully.")
        return loaded_values

    except json.JSONDecodeError as e:
        logger.error(f"Corrupt settings file '{SETTINGS_FILE}': {e}. Attempting to handle.")
        return handle_corrupt_settings(defaults)
    except Exception as e:
        logger.error(f"Error loading settings from '{SETTINGS_FILE}': {e}\n{traceback.format_exc()}. Using defaults.")
        return defaults



def handle_corrupt_settings(defaults):
    """Handle corrupt settings file by renaming it and returning defaults."""
    try:
        corrupt_filename = f"{SETTINGS_FILE}.corrupt_{int(time.time())}"
        if os.path.exists(SETTINGS_FILE):  # Check again before renaming
            os.rename(SETTINGS_FILE, corrupt_filename)
            logger.info(f"Renamed corrupt settings file to {corrupt_filename}.")
        else:
            logger.info("Corrupt settings file was already removed or renamed.")

        # Remove persistent prompt as we can't trust its validity if settings are corrupt
        if os.path.exists(PERSISTENT_PROMPT_PATH):
            logger.info("Removing potentially related persistent prompt due to corrupt settings.")
            os.remove(PERSISTENT_PROMPT_PATH)
    except Exception as e:
        logger.error(f"Could not handle corrupt settings file operations: {e}")

    # Return a fresh copy of defaults
    return defaults.copy()


# --- Text Processing ---
@lru_cache(maxsize=1024)
def _find_intelligent_split_point(text_segment: str, target_chars: int, max_chars: int) -> int:
    """Find optimal text split point with better error handling and performance optimizations"""
    try:
        segment_len = len(text_segment)
        if segment_len <= max_chars:
            return segment_len

        # Define search range
        search_end = min(segment_len - 1, max_chars - 1)
        search_start = max(0, target_chars - (max_chars - target_chars) // 2)
        search_start = max(0, min(search_start, search_end - 1))

        # Search for strong punctuation first (most desirable split points)
        for i in range(search_end, search_start - 1, -1):
            if text_segment[i] in PUNCTUATION_SPLIT_STRONGLY:
                if i + 1 < segment_len and text_segment[i + 1].isspace():
                    return i + 1
                elif i == segment_len - 1:
                    return i + 1

        # Then search for weak punctuation
        for i in range(search_end, search_start - 1, -1):
            if text_segment[i] in PUNCTUATION_SPLIT_WEAKLY:
                if i + 1 < segment_len and text_segment[i + 1].isspace():
                    return i + 1
                elif i == segment_len - 1:
                    return i + 1

        # Finally, look for whitespace
        whitespace_end = min(segment_len - 1, max_chars - 1)
        whitespace_start = max(0, MIN_CHUNK_SIZE - 1)

        if whitespace_end > whitespace_start:
            # Find last space in the range
            space_pos = text_segment.rfind(' ', whitespace_start, whitespace_end + 1)
            if space_pos != -1:
                return space_pos + 1

        # Fallback: hard split at max_chars
        return min(segment_len, max_chars)

    except Exception as e:
        logger.error(f"Error in _find_intelligent_split_point: {e}. Falling back to max_chars.")
        return min(len(text_segment), max_chars)


def get_settings_profiles() -> list[str]:
    """Returns a list of available settings profile names."""
    try:
        ensure_app_data_dir()  # Ensure the app data directory exists
        if not os.path.exists(SETTINGS_DIR):
            return []

        profiles = [
            f[:-5]  # Remove .json extension
            for f in os.listdir(SETTINGS_DIR)
            if f.endswith(".json") and os.path.isfile(os.path.join(SETTINGS_DIR, f))
        ]
        profiles.sort()  # Sort alphabetically
        return profiles
    except Exception as e:
        logging.error(f"Error getting settings profiles: {str(e)}")
        return []

def save_profile(
    profile_name: str,
    text_val, audio_prompt_path,

    # base
    cfg_val, exag_val,

    # turbo
    top_p_val, top_k_val, repetition_penalty_val, norm_loudness_val,

    # shared
    temp_val,
    target_chars_val, max_chars_val,
    device_val, seed_val,
    model_choice_val,
):
    """Saves the current UI state as a named profile."""
    if not profile_name or not profile_name.strip():
        raise gr.Error("Profile name cannot be empty.")

    sanitized_name = re.sub(r"[^\w\s-]", "", profile_name).strip()
    if not sanitized_name:
        raise gr.Error("Invalid characters in profile name. Use letters, numbers, spaces, or hyphens.")

    model_choice_val = (model_choice_val or "turbo").lower()
    if model_choice_val not in ("base", "turbo"):
        model_choice_val = "turbo"

    logger.info(f"Saving settings profile: '{sanitized_name}'")
    ensure_app_data_dir()

    settings_data = {
        "text": text_val,

        # base
        "cfg_weight": float(cfg_val),
        "exaggeration": float(exag_val),

        # turbo
        "top_p": float(top_p_val),
        "top_k": int(top_k_val),
        "repetition_penalty": float(repetition_penalty_val),
        "norm_loudness": bool(norm_loudness_val),

        # shared
        "temperature": float(temp_val),
        "target_chars_per_chunk": int(target_chars_val),
        "max_chars_per_chunk": int(max_chars_val),
        "device": str(device_val),
        "seed": int(seed_val),
        "model_choice": model_choice_val,

        "has_audio_prompt": False,
    }

    # Handle the audio prompt
    if audio_prompt_path and os.path.exists(audio_prompt_path):
        try:
            prompt_dest_path = os.path.join(SETTINGS_PROMPTS_DIR, f"{sanitized_name}.wav")
            shutil.copy(audio_prompt_path, prompt_dest_path)
            settings_data["has_audio_prompt"] = True
            logger.info(f"Copied audio prompt to {prompt_dest_path}")
        except Exception as e:
            logger.error(f"Could not save audio prompt for profile '{sanitized_name}': {e}")
            gr.Warning(f"Could not save audio prompt for profile '{sanitized_name}': {e}")

    profile_json_path = os.path.join(SETTINGS_DIR, f"{sanitized_name}.json")
    try:
        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=4)
    except Exception as e:
        raise gr.Error(f"Failed to write settings file: {e}")

    gr.Info(f"Profile '{sanitized_name}' saved successfully!")
    updated_profiles = get_settings_profiles()
    return gr.update(choices=updated_profiles, value=sanitized_name)



def load_profile(profile_name: str):
    """Loads a profile and returns gr.update objects for all UI components."""
    if not profile_name:
        # return empty updates for all components (match count below)
        return (gr.update(),) * 15

    logger.info(f"Loading settings profile: '{profile_name}'")
    profile_json_path = os.path.join(SETTINGS_DIR, f"{profile_name}.json")

    if not os.path.exists(profile_json_path):
        raise gr.Error(f"Profile '{profile_name}' not found.")

    with open(profile_json_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    # Audio prompt
    audio_prompt_to_load = None
    if settings.get("has_audio_prompt"):
        prompt_path = os.path.join(SETTINGS_PROMPTS_DIR, f"{profile_name}.wav")
        if os.path.exists(prompt_path):
            audio_prompt_to_load = prompt_path
            logger.info(f"Found associated audio prompt: {prompt_path}")
        else:
            logger.warning(f"Profile '{profile_name}' expects a prompt, but file not found: {prompt_path}")
            gr.Warning(f"Audio prompt for profile '{profile_name}' is missing.")

    model_choice = str(settings.get("model_choice", "turbo")).lower()
    if model_choice not in ("base", "turbo"):
        model_choice = "turbo"

    gr.Info(f"Profile '{profile_name}' loaded.")

    return (
        gr.update(value=settings.get("text", "")),
        gr.update(value=audio_prompt_to_load),

        # base
        gr.update(value=float(settings.get("cfg_weight", 0.2))),
        gr.update(value=float(settings.get("exaggeration", 0.8))),

        # turbo
        gr.update(value=float(settings.get("top_p", 0.95))),
        gr.update(value=int(settings.get("top_k", 1000))),
        gr.update(value=float(settings.get("repetition_penalty", 1.2))),
        gr.update(value=bool(settings.get("norm_loudness", True))),

        # shared
        gr.update(value=float(settings.get("temperature", 0.8))),
        gr.update(value=int(settings.get("target_chars_per_chunk", 100))),
        gr.update(value=int(settings.get("max_chars_per_chunk", 200))),
        gr.update(value=settings.get("device", "cuda")),
        gr.update(value=int(settings.get("seed", 0))),
        gr.update(value=model_choice),

        # also update the name textbox for easy renaming
        gr.update(value=profile_name),
    )



def delete_profile(profile_name: str):
    """Deletes a settings profile and its associated prompt."""
    if not profile_name:
        raise gr.Error("No profile selected to delete.")

    logger.warning(f"Attempting to delete profile: '{profile_name}'")

    # Delete JSON file
    profile_json_path = os.path.join(SETTINGS_DIR, f"{profile_name}.json")
    if os.path.exists(profile_json_path):
        os.remove(profile_json_path)

    # Delete associated prompt file
    prompt_path = os.path.join(SETTINGS_PROMPTS_DIR, f"{profile_name}.wav")
    if os.path.exists(prompt_path):
        os.remove(prompt_path)

    gr.Info(f"Profile '{profile_name}' deleted.")

    updated_profiles = get_settings_profiles()
    return gr.update(choices=updated_profiles, value=None), gr.update(value="")  # Update dropdown and clear name box


def intelligent_chunk_text(long_text: str, target_chars: int, max_chars: int) -> list[str]:
    """Chunk text intelligently with comprehensive error handling"""
    try:
        if not long_text or not long_text.strip():
            return []

        long_text = re.sub(r'\s+', ' ', long_text).strip()  # Normalize whitespace

        if not (MIN_CHUNK_SIZE <= target_chars < max_chars):  # Basic sanity check
            logger.error(f"Invalid chunking parameters: target={target_chars}, max={max_chars}. Using fallback.")
            # Fallback to simple splitting or return error indicator if preferred
            return [long_text[i:i + max_chars] for i in range(0, len(long_text), max_chars)]

        sentences = sent_tokenize(long_text)
        if not sentences:  # Should not happen if long_text is not empty
            logger.warning("NLTK sent_tokenize returned no sentences for non-empty text. Using full text as one chunk if possible.")
            return [long_text] if len(long_text) <= max_chars and len(long_text) >= MIN_CHUNK_SIZE else []

        chunks = []
        current_buffer = []
        current_buffer_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if sentence_length > max_chars:  # Sentence itself is too long
                if current_buffer:  # Flush buffer before handling long sentence
                    chunks.append(" ".join(current_buffer).strip())
                    current_buffer = []
                    current_buffer_length = 0

                # Split the oversized sentence
                temp_segment_start_idx = 0
                while temp_segment_start_idx < sentence_length:
                    remaining_sentence_part = sentence[temp_segment_start_idx:]
                    if not remaining_sentence_part.strip(): break  # Avoid empty trailing parts

                    if len(remaining_sentence_part) <= max_chars:
                        chunks.append(remaining_sentence_part.strip())
                        break
                    else:
                        split_at = _find_intelligent_split_point(remaining_sentence_part, target_chars, max_chars)
                        chunk_to_add = remaining_sentence_part[:split_at].strip()
                        if chunk_to_add:  # Ensure non-empty chunk
                            chunks.append(chunk_to_add)
                        temp_segment_start_idx += split_at
                        # Skip any leading whitespace for the next part of the split sentence
                        while temp_segment_start_idx < sentence_length and sentence[temp_segment_start_idx].isspace():
                            temp_segment_start_idx += 1
            else:  # Sentence fits or can be combined
                # If adding this sentence exceeds max_chars for the buffer
                if current_buffer_length + (1 if current_buffer else 0) + sentence_length > max_chars:
                    if current_buffer:  # Flush current buffer
                        chunks.append(" ".join(current_buffer).strip())
                    current_buffer = [sentence]  # Start new buffer with current sentence
                    current_buffer_length = sentence_length
                else:  # Add sentence to buffer
                    current_buffer.append(sentence)
                    current_buffer_length += (1 if len(current_buffer) > 1 else 0) + sentence_length

                # If buffer is full enough (>= target_chars)
                if current_buffer_length >= target_chars:
                    chunks.append(" ".join(current_buffer).strip())
                    current_buffer = []
                    current_buffer_length = 0

        if current_buffer:  # Flush any remaining sentences in buffer
            chunks.append(" ".join(current_buffer).strip())

        # Final filter for chunk size and non-empty
        valid_chunks = [chunk for chunk in chunks if MIN_CHUNK_SIZE <= len(chunk.strip()) <= max_chars]

        if not valid_chunks and long_text:  # If all filtering removed everything, but there was text
            logger.warning(f"Intelligent chunking resulted in no valid chunks for text of length {len(long_text)}. This might indicate issues with chunking parameters or text structure.")
            # As a last resort, try to return the original text if it's somehow valid as a single chunk
            if MIN_CHUNK_SIZE <= len(long_text) <= max_chars:
                return [long_text]

        logger.info(f"Text chunked into {len(valid_chunks)} valid chunks.")
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in intelligent_chunk_text: {e}\n{traceback.format_exc()}. Falling back to simple splitting.")
        # Fallback: simple splitting by max_chars
        try:
            return [long_text[i:i + max_chars] for i in range(0, len(long_text), max_chars) if long_text[i:i + max_chars].strip()]
        except Exception as fallback_error:
            logger.error(f"Fallback chunking also failed: {fallback_error}")
            return []


# --- Model Management ---

def ensure_model_loaded(device_choice: str, model_choice: str):
    """Load TTS model (base or turbo) with comprehensive error handling."""
    global tts_model, current_device_loaded, current_model_kind_loaded

    model_choice = (model_choice or "turbo").lower()
    if model_choice not in ("base", "turbo"):
        model_choice = "turbo"

    try:
        if (
            tts_model is not None
            and current_device_loaded == device_choice
            and current_model_kind_loaded == model_choice
        ):
            logger.info(f"Model already loaded ({model_choice}) on {current_device_loaded}")
            return f"Model already loaded ({model_choice}) on {current_device_loaded}"

        logger.info(f"Attempting to load Chatterbox model '{model_choice}' on device: {device_choice}")

        if device_choice == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA selected, but not available. Please select CPU or check CUDA installation.")
            try:
                torch.cuda.empty_cache()
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("CUDA device seems responsive.")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    mem_info = torch.cuda.memory_stats() if torch.cuda.is_available() else {}
                    logger.error(f"CUDA OOM during test allocation: {e}. Memory stats: {mem_info}")
                    raise RuntimeError(f"Insufficient GPU memory to load model. Details: {e}")
                raise RuntimeError(f"GPU error, cannot load model. Details: {e}")
        elif device_choice != "cpu":
            raise ValueError(f"Invalid device choice: {device_choice}. Must be 'cuda' or 'cpu'.")

        # Suppress verbose loader output
        with suppress_stdout_stderr():
            if model_choice == "turbo":
                tts_model_candidate = ChatterboxTurboTTS.from_pretrained(device=device_choice)
            else:
                tts_model_candidate = ChatterboxTTS.from_pretrained(device=device_choice)

        tts_model = tts_model_candidate
        current_device_loaded = device_choice
        current_model_kind_loaded = model_choice

        logger.info(f"Model loaded successfully ({model_choice}) on {current_device_loaded}")
        return f"Model loaded successfully ({model_choice}) on {current_device_loaded}"

    except Exception as e:
        error_msg_lower = str(e).lower()
        specific_error = ""
        if "out of memory" in error_msg_lower:
            specific_error = "Out of memory. Try CPU, a smaller model, or free up memory."
        elif any(kw in error_msg_lower for kw in ["download", "connection", "url", "http"]):
            specific_error = "Network error during model download. Check internet connection and Hugging Face Hub status."
        elif "permission" in error_msg_lower or "access is denied" in error_msg_lower:
            specific_error = "File permission error. Check permissions for model cache directory (usually ~/.cache/huggingface/)."
        elif "safetensors_rust" in error_msg_lower or ".so" in error_msg_lower or "libc+" in error_msg_lower:
            specific_error = f"Error with compiled components (e.g. Safetensors). Ensure dependencies are correctly installed. Original error: {e}"

        full_error_message = f"Failed to load model: {specific_error if specific_error else e}"
        logger.error(f"{full_error_message}\n{traceback.format_exc()}")

        # Cleanup
        tts_model = None
        current_device_loaded = None
        current_model_kind_loaded = None
        if device_choice == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise gr.Error(full_error_message)



# --- Main Processing Function ---
def _validate_initial_inputs(long_text_input, cfg_weight_input, exaggeration_input, temperature_input,
                             target_chars_input, max_chars_input, status_messages, progress):
    """Validates all non-file inputs and updates status."""
    logger.info("Validating inputs...")
    status_messages.append("🔄 Validating inputs...")
    # yield None, "\n".join(status_messages) # Yielding from helper is tricky with main generator

    text_valid, text_msg = validate_text_input(long_text_input)
    if not text_valid:
        status_messages.append(f"❌ Text validation failed: {text_msg}")
        return False
    status_messages.append(f"✅ {text_msg}")

    params_valid, params_msg = validate_parameters(
        cfg_weight_input, exaggeration_input, temperature_input,
        target_chars_input, max_chars_input
    )
    if not params_valid:
        status_messages.append(f"❌ Parameter validation failed: {params_msg}")
        logger.error(f"Parameter validation failed from UI/initial values: {params_msg}")
        return False
    status_messages.append(f"✅ {params_msg}")

    space_ok, space_msg = check_disk_space(required_mb=100)
    status_messages.append(f"{('✅' if space_ok else '⚠️')} {space_msg}")
    if not space_ok: logger.warning(f"Disk space warning: {space_msg}")

    progress(0.05)
    return True


def _handle_audio_prompt(audio_prompt_file_path_from_ui: Optional[str], status_messages, progress) -> Tuple[Optional[str], bool]:
    """
    Handles audio prompt logic: validation, persistence, using existing.
    Returns: (path_to_use_for_generation, prompt_successfully_persisted_this_run_flag)
    """
    status_messages.append("🔄 Processing audio prompt...")
    # yield None, "\n".join(status_messages) # Cannot yield directly from here if main fn is generator

    current_prompt_for_gen = None
    persisted_this_run = False

    if audio_prompt_file_path_from_ui:
        abs_ui_path = os.path.abspath(audio_prompt_file_path_from_ui)
        abs_persistent_path = os.path.abspath(PERSISTENT_PROMPT_PATH)

        status_messages.append(f"🔄 Validating UI-provided audio prompt: '{os.path.basename(audio_prompt_file_path_from_ui)}'...")
        # yield None, "\n".join(status_messages) # yield from main
        audio_valid, audio_msg = validate_audio_file(audio_prompt_file_path_from_ui)

        if not audio_valid:
            status_messages.append(f"❌ Invalid audio prompt from UI: {audio_msg}")
            return None, False  # Signal error
        status_messages.append(f"✅ UI-provided prompt is valid: {audio_msg}")

        is_ui_path_already_persistent = os.path.exists(PERSISTENT_PROMPT_PATH) and \
                                        abs_ui_path == abs_persistent_path

        if is_ui_path_already_persistent:
            status_messages.append(f"ℹ️ Using existing persistent prompt (UI path matches): {PERSISTENT_PROMPT_FILENAME}")
            current_prompt_for_gen = PERSISTENT_PROMPT_PATH
            persisted_this_run = True
        else:
            should_copy_to_persistent = True
            if os.path.exists(PERSISTENT_PROMPT_PATH):
                hash_ui_prompt = calculate_file_hash(audio_prompt_file_path_from_ui)
                hash_persistent_prompt = calculate_file_hash(PERSISTENT_PROMPT_PATH)
                if hash_ui_prompt and hash_persistent_prompt and hash_ui_prompt == hash_persistent_prompt:
                    should_copy_to_persistent = False
                    current_prompt_for_gen = PERSISTENT_PROMPT_PATH
                    persisted_this_run = True
                    status_messages.append(f"ℹ️ UI prompt content matches existing persistent prompt. No copy needed. Using: {PERSISTENT_PROMPT_FILENAME}")
                else:
                    status_messages.append(f"ℹ️ UI prompt content differs or first time. Will update persistent prompt.")
            else:
                status_messages.append(f"ℹ️ No existing persistent prompt. Will create from UI prompt.")

            if should_copy_to_persistent:
                status_messages.append(f"💾 Updating/creating persistent prompt from '{os.path.basename(audio_prompt_file_path_from_ui)}'...")
                # yield None, "\n".join(status_messages) # yield from main
                try:
                    ensure_app_data_dir()
                    shutil.copy(audio_prompt_file_path_from_ui, PERSISTENT_PROMPT_PATH)
                    current_prompt_for_gen = PERSISTENT_PROMPT_PATH
                    persisted_this_run = True
                    status_messages.append(f"✅ Prompt saved/updated as persistent: {PERSISTENT_PROMPT_FILENAME}")
                except Exception as e_copy:
                    logger.warning(f"Could not copy UI prompt to persistent storage: {e_copy}. Using UI-provided path for this session.", exc_info=True)
                    status_messages.append(f"⚠️ Could not save prompt persistently (Error: {str(e_copy)[:50]}...). Using UI-provided path.")
                    current_prompt_for_gen = audio_prompt_file_path_from_ui  # Fallback
                    persisted_this_run = False
    else:  # audio_prompt_file_path_from_ui is None
        status_messages.append("ℹ️ No audio prompt file selected. Will use default/inbuilt voice.")
        current_prompt_for_gen = None
        persisted_this_run = False

    progress(0.25)
    return current_prompt_for_gen, persisted_this_run


def _setup_generation_parameters_and_seed(status_messages, seed_num, cfg_weight_input, exaggeration_input, temperature_input,
                                          target_chars_input, max_chars_input, current_prompt_path_for_generation, progress):
    """Sets up generation parameters, seed and logs them."""
    status_messages.append("🔄 Setting up generation parameters & seed...")
    # yield None, "\n".join(status_messages) # yield from main

    status_messages.append(f"🎯 Using prompt: {os.path.basename(current_prompt_path_for_generation) if current_prompt_path_for_generation else 'Inbuilt/Default'}")
    status_messages.append(f"⚙️ Parameters: CFG={cfg_weight_input}, Exag={exaggeration_input}, Temp={temperature_input}")
    status_messages.append(f"✂️ Chunking: Target={target_chars_input}, Max={max_chars_input}")
    try:
        actual_seed = int(seed_num)
        if actual_seed == 0:
            actual_seed = int(time.time() * 1000) % (2 ** 32)
            status_messages.append(f"🌱 Using random seed: {actual_seed}")
        else:
            status_messages.append(f"🌱 Using fixed seed: {actual_seed}")
        set_seed(actual_seed)  # Call global set_seed
        print(f"CONSOLE: Seed set for generation. User input: {seed_num}, Actual seed used: {actual_seed}")
    except ValueError as e_val:
        status_messages.append(f"⚠️ Invalid seed value '{seed_num}'. Using random. Error: {e_val}")
        actual_seed = int(time.time() * 1000) % (2 ** 32);
        set_seed(actual_seed)
        print(f"CONSOLE: Fallback random seed set: {actual_seed}")
    except Exception as e_seed:
        status_messages.append(f"⚠️ Error setting seed: {e_seed}. Not reproducible.")
        logger.warning("Could not set seed", exc_info=True)
    progress(0.30)
    return  # Parameters are passed by value, seed is global

def _set_generation_seed(status_messages, seed_num):
    """Sets up generation parameters, seed and logs them."""
    status_messages.append("🔄 Setting seed...")
    try:
        actual_seed = int(seed_num)
        if actual_seed == 0:
            actual_seed = int(time.time() * 1000) % (2 ** 32)
            status_messages.append(f"🌱 Using random seed: {actual_seed}")
        else:
            status_messages.append(f"🌱 Using fixed seed: {actual_seed}")
        set_seed(actual_seed)  # Call global set_seed
        print(f"CONSOLE: Seed set for generation. User input: {seed_num}, Actual seed used: {actual_seed}")
    except ValueError as e_val:
        status_messages.append(f"⚠️ Invalid seed value '{seed_num}'. Using random. Error: {e_val}")
        actual_seed = int(time.time() * 1000) % (2 ** 32);
        set_seed(actual_seed)
        print(f"CONSOLE: Fallback random seed set: {actual_seed}")
    except Exception as e_seed:
        status_messages.append(f"⚠️ Error setting seed: {e_seed}. Not reproducible.")
        logger.warning("Could not set seed", exc_info=True)
    return

def _perform_tts_generation_loop(
    long_text_input: str,
    target_chars_input: int,
    max_chars_input: int,
    current_prompt_path_for_generation: Optional[str],
    model_choice: str,

    # base
    cfg_weight_input: float,
    exaggeration_input: float,

    # turbo
    top_p_input: float,
    top_k_input: int,
    repetition_penalty_input: float,
    norm_loudness_input: bool,

    # shared
    temperature_input: float,

    status_messages: list,
    progress: gr.Progress
):
    """
    Generator: yields (None, status_text) while synthesizing.
    Returns: (all_wavs_list, failed_chunks_count)
    """
    global tts_model

    model_choice = (model_choice or "turbo").lower()
    if model_choice not in ("base", "turbo"):
        model_choice = "turbo"

    # --- Chunking ---
    status_messages.append("🔄 Chunking text...")
    yield None, "\n".join(status_messages)

    try:
        chunks = intelligent_chunk_text(long_text_input, target_chars_input, max_chars_input)
        if not chunks:
            status_messages.append("❌ Error: Text chunking resulted in no processable chunks.")
            yield None, "\n".join(status_messages)
            return None, 0

        status_messages.append(f"Text divided into {len(chunks)} chunks.")
        progress(0.32, desc="Chunking complete.")
        yield None, "\n".join(status_messages)
    except Exception as e_chunk_text:
        status_messages.append(f"❌ Error during text chunking: {e_chunk_text}")
        logger.error("Text chunking failed", exc_info=True)
        yield None, "\n".join(status_messages)
        return None, 0

    # --- Generation Loop ---
    all_wavs: list[torch.Tensor] = []
    failed_chunks_count = 0
    total_chunks = len(chunks)
    loop_progress_start = 0.32
    loop_progress_range = 0.58

    for i, chunk_text in enumerate(chunks):
        progress(
            loop_progress_start + (loop_progress_range * (i / total_chunks)),
            desc=f"Synthesizing chunk {i + 1}/{total_chunks}"
        )

        temp_status_for_yield = status_messages[:] + [
            f"Synthesizing chunk {i + 1}/{total_chunks} (len {len(chunk_text)})..."
        ]
        yield None, "\n".join(temp_status_for_yield)

        print(f"CONSOLE: Chunk {i + 1}/{total_chunks} ('{chunk_text[:25].replace(chr(10), ' ')}...')")

        for attempt in range(1, 3):
            try:
                with suppress_stdout_stderr():
                    # Build kwargs based on model
                    if model_choice == "turbo":
                        kwargs = dict(
                            temperature=float(temperature_input),
                            top_p=float(top_p_input),
                            top_k=int(top_k_input),
                            repetition_penalty=float(repetition_penalty_input),
                            norm_loudness=bool(norm_loudness_input),
                        )
                    else:
                        kwargs = dict(
                            temperature=float(temperature_input),
                            cfg_weight=float(cfg_weight_input),
                            exaggeration=float(exaggeration_input),
                        )

                    if current_prompt_path_for_generation is not None:
                        kwargs["audio_prompt_path"] = current_prompt_path_for_generation

                    wav_chunk = tts_model.generate(chunk_text, **kwargs)

                all_wavs.append(wav_chunk)
                status_messages.append(f"✅ Chunk {i + 1}/{total_chunks} synthesized.")
                break

            except Exception as e_chunk:
                logger.warning(
                    f"Attempt {attempt} for chunk {i + 1} failed.",
                    exc_info=True if attempt == 2 else False
                )
                if attempt == 2:
                    status_messages.append(f"❌ Error on chunk {i + 1}: {str(e_chunk)[:100]}")
                    status_messages.append(f"⏭️ Skipping chunk {i + 1}. Adding 0.5s silence.")
                    failed_chunks_count += 1
                    silence_device = getattr(tts_model, "device", "cpu")
                    silence_sr = getattr(tts_model, "sr", 24000)
                    all_wavs.append(torch.zeros((1, int(silence_sr * 0.5)), device=silence_device))
                else:
                    time.sleep(0.2)

        yield None, "\n".join(status_messages)

    progress(0.90, desc="All chunks processed.")

    if not all_wavs or all(w.numel() == 0 for w in all_wavs):
        status_messages.append("❌ Error: No audio data generated after processing all chunks.")
        yield None, "\n".join(status_messages)
        return None, failed_chunks_count

    if failed_chunks_count > 0:
        status_messages.append(f"⚠️ Gen completed with {failed_chunks_count}/{total_chunks} failed chunks.")
        yield None, "\n".join(status_messages)

    return all_wavs, failed_chunks_count



# --- Main Processing Function (Refactored) ---
def process_text_to_speech(
    long_text_input: str,
    audio_prompt_file_path: Optional[str],

    # base
    cfg_weight_input: float,
    exaggeration_input: float,

    # turbo
    top_p_input: float,
    top_k_input: int,
    repetition_penalty_input: float,
    norm_loudness_input: bool,

    # shared
    temperature_input: float,
    target_chars_input: int,
    max_chars_input: int,
    device_choice: str,
    seed_num: int,
    model_choice: str,

    progress: gr.Progress = gr.Progress(track_tqdm=False)
):
    status_messages = []
    generated_audio_final_path = None
    global tts_model
    global current_settings

    try:
        progress(0, desc="Initializing...")
        try:
            ensure_app_data_dir()
            status_messages.append("✅ App data dir ready.")
        except Exception as e:
            status_messages.append(f"❌ Critical Error setting up app data: {str(e)}")
            logger.error("Critical Error setting up app data directory", exc_info=True)
            yield None, "\n".join(status_messages)
            return
        yield None, "\n".join(status_messages)

        # Stage 1: Validations
        if not _validate_initial_inputs(
            long_text_input,
            cfg_weight_input, exaggeration_input, temperature_input,
            target_chars_input, max_chars_input,
            status_messages, progress
        ):
            yield None, "\n".join(status_messages)
            return
        yield None, "\n".join(status_messages)

        # Stage 2: Model Loading
        progress(0.05, desc="Loading model...")
        status_messages.append(f"🔄 Loading TTS model '{model_choice}' on {device_choice}...")
        yield None, "\n".join(status_messages)
        try:
            model_load_status = ensure_model_loaded(device_choice, model_choice)
            status_messages.append(f"✅ {model_load_status}")
        except gr.Error as e_gr:
            status_messages.append(f"❌ Model loading failed: {str(e_gr)}")
            yield None, "\n".join(status_messages)
            return
        except Exception as e_model_load:
            status_messages.append(f"❌ Unexpected model loading error: {str(e_model_load)}")
            logger.error("Unexpected model loading error", exc_info=True)
            yield None, "\n".join(status_messages)
            return
        progress(0.20, desc="Model loaded.")
        yield None, "\n".join(status_messages)

        # Stage 3: Audio Prompt Handling
        current_prompt_path_for_generation, prompt_successfully_persisted_this_run = _handle_audio_prompt(
            audio_prompt_file_path, status_messages, progress
        )
        if current_prompt_path_for_generation is False:
            yield None, "\n".join(status_messages)
            return
        yield None, "\n".join(status_messages)

        # Inbuilt voice conditioning if no prompt
        if current_prompt_path_for_generation is None and tts_model:
            status_messages.append("✅ Setting up inbuilt voice conditioning...")
            yield None, "\n".join(status_messages)
            try:
                REPO_ID = "ResembleAI/chatterbox"
                conds_filename = "conds.pt"
                with suppress_stdout_stderr():
                    local_path_conds = hf_hub_download(repo_id=REPO_ID, filename=conds_filename)
                load_device = current_device_loaded if current_device_loaded else device_choice
                if hasattr(tts_model, "conds") and hasattr(tts_model.conds, "load"):
                    with suppress_stdout_stderr():
                        tts_model.conds = tts_model.conds.load(local_path_conds, load_device)
                    status_messages.append(f"✅ Inbuilt voice conditioning loaded for '{load_device}'.")
                else:
                    status_messages.append("⚠️ Model has no 'conds.load' for inbuilt voice.")
            except Exception as e_inbuilt:
                status_messages.append(f"❌ Error setting up inbuilt voice: {str(e_inbuilt)[:100]}...")
                logger.error("Error setting up inbuilt voice", exc_info=True)
            yield None, "\n".join(status_messages)

        # Stage 4: Seed / logging (keep your helper, but note it logs base params)
        _model = (model_choice or "turbo").lower()
        _set_generation_seed(
            status_messages, seed_num
        )
        # Add a model-specific param line (so logs stay meaningful)
        if _model == "turbo":
            status_messages.append(
                f"⚙️ Turbo: top_p={top_p_input:.2f}, top_k={int(top_k_input)}, "
                f"rep_pen={repetition_penalty_input:.2f}, "
                f"norm_loudness={'on' if norm_loudness_input else 'off'}, "
                f"temp={temperature_input:.2f}"
            )
        else:
            status_messages.append(
                f"⚙️ Base: cfg={cfg_weight_input}, exag={exaggeration_input}, temp={temperature_input}"
            )
        status_messages.append(f"✂️ Chunking: Target={target_chars_input}, Max={max_chars_input}")
        yield None, "\n".join(status_messages)

        # Stage 5: Generation
        progress(0.30, desc="Starting TTS Synthesis...")
        all_wavs, failed_chunks_count = yield from _perform_tts_generation_loop(
            long_text_input, target_chars_input, max_chars_input,
            current_prompt_path_for_generation,
            model_choice,
            cfg_weight_input, exaggeration_input,
            top_p_input, top_k_input, repetition_penalty_input, norm_loudness_input,
            temperature_input,
            status_messages,
            progress
        )
        if all_wavs is None:
            return

        # Stage 6: Merge + Save (unchanged from your logic)
        progress(0.90, desc="Merging audio...")
        status_messages.append("🔊 Merging audio chunks...")
        yield None, "\n".join(status_messages)

        try:
            merge_device = getattr(tts_model, "device", "cpu")
            processed_wavs = [wav.to(merge_device) for wav in all_wavs]
            merged_wav = torch.cat(processed_wavs, dim=1)
        except RuntimeError as e_merge:
            status_messages.append(f"❌ Error merging audio: {str(e_merge)}. Try CPU merge.")
            yield None, "\n".join(status_messages)
            try:
                cpu_wavs = [wav.cpu() for wav in all_wavs]
                merged_wav = torch.cat(cpu_wavs, dim=1)
                status_messages.append("✅ Merged on CPU after initial failure.")
            except Exception as e_cpu_merge:
                status_messages.append(f"❌ CPU merge also failed: {e_cpu_merge}")
                yield None, "\n".join(status_messages)
                return

        progress(0.95, desc="Merging complete.")
        yield None, "\n".join(status_messages)

        if merged_wav.numel() == 0:
            status_messages.append("❌ Error: Merged audio is empty.")
            yield None, "\n".join(status_messages)
            return

        status_messages.append("🔄 Saving audio...")
        yield None, "\n".join(status_messages)

        output_dir = os.path.join(APP_DATA_DIR, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        output_filename = f"chatterbox_output_{timestamp}.wav"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            sample_rate = getattr(tts_model, "sr", 24000)
            ta.save(output_filepath, merged_wav.cpu(), sample_rate)
            if not (os.path.exists(output_filepath) and os.path.getsize(output_filepath) >= MIN_SAVED_AUDIO_FILE_SIZE_BYTES):
                raise IOError("Saved file missing or too small.")
            generated_audio_final_path = output_filepath
            status_messages.append(f"✅ Audio saved: {output_filepath}")
            current_settings["output_filepath"] = output_filepath
        except Exception as e_save:
            status_messages.append(f"⚠️ Error saving to primary: {e_save}. Trying backup.")
            logger.warning("Primary save failed. Trying backup.", exc_info=True)
            backup_output_filepath = os.path.join(os.getcwd(), f"chatterbox_output_backup_{timestamp}.wav")
            try:
                ta.save(backup_output_filepath, merged_wav.cpu(), sample_rate)
                if not (os.path.exists(backup_output_filepath) and os.path.getsize(backup_output_filepath) >= MIN_SAVED_AUDIO_FILE_SIZE_BYTES):
                    raise IOError("Backup saved file missing or too small.")
                generated_audio_final_path = backup_output_filepath
                status_messages.append(f"✅ Audio saved to backup: {backup_output_filepath}")
            except Exception as e_backup_save:
                status_messages.append(f"❌ Error saving to backup: {e_backup_save}")
                yield None, "\n".join(status_messages)
                return

        progress(0.98, desc="Saving complete.")
        duration_secs = merged_wav.shape[1] / sample_rate
        final_msg = f"🎉 Success! Audio duration: {duration_secs:.2f}s. Output: {os.path.basename(generated_audio_final_path)}"
        if failed_chunks_count > 0:
            final_msg += f" (with {failed_chunks_count} silent chunks)"
        status_messages.append(final_msg)
        print(f"CONSOLE: {final_msg}")

        # IMPORTANT: update save_settings signature in your codebase to include these extra fields
        save_settings(
            device_choice, cfg_weight_input, exaggeration_input, temperature_input,
            long_text_input, target_chars_input, max_chars_input,
            int(seed_num), prompt_successfully_persisted_this_run,
            model_choice, top_p_input, top_k_input, repetition_penalty_input, norm_loudness_input
        )

        status_messages.append("✅ Done!")
        progress(1.0, desc="Completed!")
        yield generated_audio_final_path, "\n".join(status_messages)
        return

    except Exception as e_outer:
        error_msg = f"❌ Critical error in TTS process: {e_outer}"
        print(f"CONSOLE: {error_msg}\n{traceback.format_exc()}")
        logger.error("Critical TTS error", exc_info=True)
        status_messages.append(error_msg)
        yield None, "\n".join(status_messages)
        return

    finally:
        if tts_model and current_device_loaded == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared.")
            except Exception as e_cache:
                logger.warning(f"Could not clear CUDA cache: {e_cache}")
        logger.info("TTS generation process finished.")



# --- Function to Clear Persistent Prompt ---
def clear_persistent_prompt_action(
        current_audio_prompt_path: Optional[str],  # ADDED: To receive current value of audio_prompt_input
        current_status_text_val: str  # Existing input
):
    messages = []
    logger.info(f"Clear persistent prompt action called. Current UI prompt path: {current_audio_prompt_path}")
    # The current_audio_prompt_path isn't strictly needed for the clearing logic itself,
    # but having it in 'inputs' seems to help gr.File initialize correctly.

    cleared_physically = False
    settings_updated = False
    try:
        if os.path.exists(PERSISTENT_PROMPT_PATH):
            os.remove(PERSISTENT_PROMPT_PATH)
            messages.append(f"✅ Successfully removed persistent prompt file: {PERSISTENT_PROMPT_FILENAME}")
            logger.info(f"User cleared persistent prompt file: {PERSISTENT_PROMPT_PATH}")
            cleared_physically = True
        else:
            messages.append(f"ℹ️ No persistent prompt file found at {PERSISTENT_PROMPT_PATH} to remove.")
            # Consider this a success in terms of the desired state (no persistent file)
            cleared_physically = True

        # Update settings to reflect no persistent prompt
        current_saved_settings = load_settings()
        if current_saved_settings.get("has_persistent_prompt", False) or current_saved_settings.get("audio_prompt_path") is not None:
            save_settings(
                device=current_saved_settings["device"],
                cfg=current_saved_settings["cfg_weight"],
                exg=current_saved_settings["exaggeration"],
                temp=current_saved_settings["temperature"],
                long_text=current_saved_settings["long_text"],
                target_chars=current_saved_settings["target_chars_per_chunk"],
                max_chars=current_saved_settings["max_chars_per_chunk"],
                seed_value=current_saved_settings["seed"],
                prompt_successfully_persisted_this_run=False,
                model_choice=current_saved_settings.get("model_choice", "turbo"),
                top_p=current_saved_settings.get("top_p", 0.95),
                top_k=current_saved_settings.get("top_k", 1000),
                repetition_penalty=current_saved_settings.get("repetition_penalty", 1.2),
                norm_loudness=current_saved_settings.get("norm_loudness", True),
            )

            messages.append("✅ Settings updated to reflect no active persistent prompt.")
            settings_updated = True
        else:
            messages.append("ℹ️ Settings already indicate no active persistent prompt.")
            settings_updated = True  # State is already as desired

        if cleared_physically and settings_updated:
            # Return update for the audio prompt input (to clear it) and the status text
            return gr.update(value=None), "\n".join(messages)
        else:
            # Should ideally always be true if logic is correct, but as a fallback
            return gr.update(value=None), "\n".join(messages)  # Still attempt to clear UI field

    except Exception as e:
        logger.error(f"Error clearing persistent prompt: {e}", exc_info=True)
        messages.append(f"❌ Error clearing persistent prompt: {e}")
        # Return current (or no change) for audio_prompt_input, and error for status
        return gr.update(), "\n".join(messages)  # gr.update() with no args means no change


# --- Gradio Interface Definition ---
css = """
footer {display: none !important;}

/* --- KEY FIXES FOR FULL-WIDTH LAYOUT --- */
:root {
    /* Allows the inner content container to grow */
    --layout-container-width-max: 100% !important;
    /* Removes the max-width limit from the main Blocks container */
    --max-width: none !important;
}

.gr-input-label {font-weight: bold;}

/* --- CSS for .file-input-container-fixed-height (for audio_prompt_input VERTICAL stability) --- */
/* THIS SECTION REMAINS THE SAME - IT'S FOR HEIGHT */
.file-input-container-fixed-height {
    min-height: 100px;
    height: 100px;
    width: 100%;       /* Takes full width of its parent PROPORTIONAL column */
    display: flex;
    flex-direction: column;
    border: 1px solid #E0E0E0;
    padding: 8px;
    box-sizing: border-box;
    overflow: hidden;
}
.file-input-container-fixed-height > .styler {
    display: flex; flex-direction: column; flex-grow: 1; width: 100%; min-height: 0; overflow:hidden;
}
.file-input-container-fixed-height > .styler > .block {
    flex-grow: 1; display: flex; flex-direction: column; width: 100%; min-height: 0; overflow: hidden;
}
.file-input-container-fixed-height > .styler > .block > label {
    flex-shrink: 0; margin-bottom: 8px; text-align: left; width: 100%; box-sizing: border-box;
}
.file-input-container-fixed-height .file-preview-holder {
    flex-grow: 1; display: flex; flex-direction: column; justify-content: center;
    align-items: flex-start; width: 100%; min-height: 0; overflow-y: auto;
}
.file-input-container-fixed-height .file-preview-holder td.filename {
    max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block;
}
.file-input-container-fixed-height > .styler > .block > button {
    flex-grow: 1; height: auto; min-height: 100px; width: 100%;
    display: flex; align-items: center; justify-content: center;
    border: 2px dashed #A0A0A0; box-sizing: border-box; padding: 10px; text-align: center;
}


/* --- Styling for Output Row to have equal height columns (Optional but good) --- */
/* If you added elem_id="main_output_row" to the gr.Row for outputs: */
#main_output_row { /* Keep this if you want stretched equal height for output columns */
    display: flex;
    align-items: stretch;
}
#main_output_row > div[class*="gradio-column"] > div.block,
#main_output_row > div[class*="gradio-column"] > div.form {
    flex-grow: 1; display: flex; flex-direction: column; height: 100%;
}
#main_output_row div.gradio-audio {
    flex-grow: 1; display: flex; flex-direction: column; min-height: 150px;
}
#main_output_row div.gradio-audio .empty {
    flex-grow: 1; display: flex; align-items: center; justify-content: center;
}

/* --- General rule to help components conform to column widths --- */
/* This tells direct .block or .form children of any Gradio column to take 100% width */
div[class*="gradio-column"] > .block,
div[class*="gradio-column"] > .form {
    width: 100% !important;
    box-sizing: border-box; /* Include padding/border in the 100% width */
}
/* And textareas within those forms */
div[class*="gradio-column"] > .form textarea {
    width: 100% !important;
    box-sizing: border-box;
}
"""
def ui_on_model_change(model_choice: str):
    is_turbo = (model_choice == "turbo")
    return (
        gr.update(visible=not is_turbo),  # cfg_weight_slider
        gr.update(visible=not is_turbo),  # exaggeration_slider
        gr.update(visible=is_turbo),      # top_p_slider
        gr.update(visible=is_turbo),      # top_k_slider
        gr.update(visible=is_turbo),      # repetition_penalty_slider
        gr.update(visible=is_turbo),      # norm_loudness_checkbox
    )

try:
    initial_settings = load_settings()
    current_settings = {
        "audio_prompt_file_path": initial_settings.get("audio_prompt_path"),
        "cfg_weight_input": initial_settings.get("cfg_weight"),
        "exaggeration_input": initial_settings.get("exaggeration"),
        "temperature_input": initial_settings.get("temperature"),
        "target_chars_input": initial_settings.get("target_chars_per_chunk"),
        "max_chars_input": initial_settings.get("max_chars_per_chunk"),
        "device_choice": initial_settings.get("device"),
        "seed_num": initial_settings.get("seed"),
        "output_filepath": None,
        "model_choice": initial_settings.get("model_choice", "turbo"),
        "top_p_input": initial_settings.get("top_p", 0.95),
        "top_k_input": initial_settings.get("top_k", 1000),
        "repetition_penalty_input": initial_settings.get("repetition_penalty", 1.2),
        "norm_loudness_input": initial_settings.get("norm_loudness", True),
    }
except Exception as e:
    logger.critical(f"FATAL: Could not load initial settings: {e}. Using hardcoded defaults.", exc_info=True)
    initial_settings = {
        "device": "cpu", "cfg_weight": 0.2, "exaggeration": 0.8, "temperature": 0.8,
        "long_text": "Welcome! Settings failed to load.",
        "target_chars_per_chunk": 100, "max_chars_per_chunk": 200, "seed": 0,
        "audio_prompt_path": None
    }

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), fill_width=True) as demo:
    gr.Markdown("<h1 align='center'>ChatterBoxViz Long Text Synthesizer</h1>"
                "Synthesize long text. Upload text, a WAV voice prompt, adjust parameters, and generate!")

    # This is the main row whose columns you want to fix
    with gr.Row(elem_id="main_content_row"):
        with gr.Column(scale=3):  # Input Column Left
            # ... (text_input, audio_prompt_input, clear_prompt_button are the same)
            text_input = gr.Textbox(
                label="📜 Long Text Input",
                placeholder="Paste long text here...",
                lines=22,
                value=initial_settings["long_text"]
            )
            with gr.Group(elem_classes="file-input-container-fixed-height"):
                _initial_audio_prompt_ui = initial_settings.get("audio_prompt_path")
                if _initial_audio_prompt_ui and not os.path.exists(_initial_audio_prompt_ui):
                    _initial_audio_prompt_ui = None
                audio_prompt_input = gr.File(
                    label="🎤 Voice Prompt WAV File (Max 100MB, 0.5-30s)",  # ...
                    value=_initial_audio_prompt_ui
                )
            clear_prompt_button = gr.Button("🧹 Clear Last-Used Voice Prompt", variant="stop", size="sm")

        with gr.Column(scale=2):  # Input Column Right

            # ### NEW ###: Settings Management Section
            with gr.Accordion("⚙️ Settings Profile Management", open=True):
                with gr.Row():
                    settings_dropdown = gr.Dropdown(
                        label="Load Profile",
                        choices=get_settings_profiles(),
                        scale=3
                    )
                with gr.Row():
                    profile_name_textbox = gr.Textbox(
                        label="Profile Name",
                        placeholder="Enter name to save or rename...",
                        scale=3
                    )
                    save_profile_button = gr.Button("💾 Save", variant="primary", scale=1)
                    delete_profile_button = gr.Button("🗑️ Delete", variant="stop", scale=1)

            gr.Markdown("### ⚙️ Generation Parameters")
            device_input = gr.Dropdown(
                # ... (rest of the right column is the same)
                label="🧠 Processing Device", choices=["cuda", "cpu"],
                value=initial_settings["device"], info="Select 'cuda' for GPU or 'cpu'."
            )
            model_dropdown = gr.Dropdown(
                label="🗣️ TTS Model",
                choices=[("ChatterboxTTS (base)", "base"), ("ChatterboxTurboTTS", "turbo")],
                value=initial_settings.get("model_choice", "turbo"),
                info="Base supports CFG/Exaggeration. Turbo supports Top-P/Top-K/Repetition/Norm Loudness."
            )

            _initial_model = initial_settings.get("model_choice", "turbo")
            is_base = (_initial_model == "base")
            is_turbo = (_initial_model == "turbo")
            # Base params
            cfg_weight_slider = gr.Slider(
                minimum=0.0, maximum=2.0,
                value=initial_settings.get("cfg_weight", 0.2),
                step=0.01,
                label="🧭 CFG Weight",
                visible=is_base
            )
            exaggeration_slider = gr.Slider(
                minimum=0.0, maximum=2.0,
                value=initial_settings.get("exaggeration", 0.8),
                step=0.01,
                label="🎭 Exaggeration",
                visible=is_base
            )

            # Turbo params
            top_p_slider = gr.Slider(
                minimum=0.0, maximum=1.0,
                value=initial_settings.get("top_p", 0.95),
                step=0.01,
                label="🔊 Top P",
                visible=is_turbo
            )
            top_k_slider = gr.Slider(
                minimum=0, maximum=1000,
                value=initial_settings.get("top_k", 1000),
                step=10,
                label="🎭 Top K",
                visible=is_turbo
            )
            repetition_penalty_slider = gr.Slider(
                minimum=1.0, maximum=2.0,
                value=initial_settings.get("repetition_penalty", 1.2),
                step=0.01,
                label="🔁 Repetition Penalty",
                visible=is_turbo
            )
            norm_loudness_checkbox = gr.Checkbox(
                value=bool(initial_settings.get("norm_loudness", True)),
                label="🔈 Normalize Loudness",
                visible=is_turbo
            )

            temperature_slider = gr.Slider(
                minimum=0.05, maximum=2.0,
                value=initial_settings.get("temperature", 0.8),
                step=0.05,
                label="🔥 Temperature"
            )

            seed_slider = gr.Slider(
                minimum=0, maximum=2 ** 32 - 1, value=initial_settings["seed"],
                step=1, label="🌱 Seed", info="Set a specific seed for reproducibility. 0 means random."
            )
            gr.Markdown("### ✂️ Chunking Parameters")
            target_chars_slider = gr.Slider(
                minimum=MIN_CHUNK_SIZE, maximum=500, value=initial_settings["target_chars_per_chunk"], step=10,
                label="🎯 Target Chars/Chunk", info=f"Approx. target characters per audio chunk (min {MIN_CHUNK_SIZE})."
            )
            max_chars_slider = gr.Slider(
                minimum=MIN_CHUNK_SIZE + 10, maximum=1000, value=initial_settings["max_chars_per_chunk"], step=10,
                label="🛑 Max Chars/Chunk", info="Hard character limit per chunk (must be > Target)."
            )

    with gr.Row():
        submit_button = gr.Button("🚀 Generate Speech", variant="primary", scale=1)

    # ... (rest of the UI layout is the same)
    with gr.Row(elem_id="main_content_row"):
        with gr.Column(scale=3):
            audio_output = gr.Audio(label="🎧 Generated Speech Output", type="filepath")
        with gr.Column(scale=2):
            status_output = gr.Textbox(
                label="📊 Log / Status Updates",
                lines=15,
                interactive=False,
                max_lines=20,
                autoscroll=True
            )

    # ### NEW ###: List of components to be controlled by settings profiles
    # This makes the event handler definitions cleaner
    profile_controlled_components = [
        text_input, audio_prompt_input,

        # base
        cfg_weight_slider, exaggeration_slider,

        # turbo
        top_p_slider, top_k_slider, repetition_penalty_slider, norm_loudness_checkbox,

        # shared
        temperature_slider,
        target_chars_slider, max_chars_slider,
        device_input, seed_slider,
        model_dropdown,

        # name field (last)
        profile_name_textbox
    ]

    # Event listener for submit button (no changes here)
    # --- Dynamic show/hide on model change ---
    model_dropdown.change(
        fn=ui_on_model_change,
        inputs=[model_dropdown],
        outputs=[
            cfg_weight_slider, exaggeration_slider,
            top_p_slider, top_k_slider, repetition_penalty_slider, norm_loudness_checkbox
        ]
    )

    submit_button.click(
        fn=process_text_to_speech,
        inputs=[
            text_input, audio_prompt_input,

            # base params
            cfg_weight_slider, exaggeration_slider,

            # turbo params
            top_p_slider, top_k_slider, repetition_penalty_slider, norm_loudness_checkbox,

            # shared
            temperature_slider,
            target_chars_slider, max_chars_slider,
            device_input, seed_slider,
            model_dropdown,
        ],
        outputs=[audio_output, status_output],
        show_progress_on=audio_output,
        show_progress="full"
    )

    clear_prompt_button.click(
        fn=clear_persistent_prompt_action,
        inputs=[audio_prompt_input, status_output],
        outputs=[audio_prompt_input, status_output]
    )

    settings_dropdown.change(
        fn=load_profile,
        inputs=[settings_dropdown],
        outputs=profile_controlled_components
    )

    save_profile_button.click(
        fn=save_profile,
        inputs=[
            profile_name_textbox,
            text_input, audio_prompt_input,

            cfg_weight_slider, exaggeration_slider,

            top_p_slider, top_k_slider, repetition_penalty_slider, norm_loudness_checkbox,

            temperature_slider,
            target_chars_slider, max_chars_slider,
            device_input, seed_slider,
            model_dropdown,
        ],
        outputs=[settings_dropdown]
    )

    delete_profile_button.click(
        fn=delete_profile,
        inputs=[settings_dropdown],
        outputs=[settings_dropdown, profile_name_textbox]
    )

    gr.Markdown("---")
    gr.Markdown(f"💡 **Note:** Last-session state is in `{APP_DATA_DIR}`. Saved profiles are in `{SETTINGS_DIR}`. Outputs in `{os.path.join(APP_DATA_DIR, 'outputs')}`.")

    # --- Live current_settings sync ---
    audio_prompt_input.change(update_setting("audio_prompt_file_path"), inputs=audio_prompt_input)
    device_input.change(update_setting("device_choice"), inputs=device_input)
    seed_slider.change(update_setting("seed_num"), inputs=seed_slider)

    model_dropdown.change(update_setting("model_choice"), inputs=model_dropdown)

    # base
    cfg_weight_slider.change(update_setting("cfg_weight_input"), inputs=cfg_weight_slider)
    exaggeration_slider.change(update_setting("exaggeration_input"), inputs=exaggeration_slider)

    # turbo
    top_p_slider.change(update_setting("top_p_input"), inputs=top_p_slider)
    top_k_slider.change(update_setting("top_k_input"), inputs=top_k_slider)
    repetition_penalty_slider.change(update_setting("repetition_penalty_input"), inputs=repetition_penalty_slider)
    norm_loudness_checkbox.change(update_setting("norm_loudness_input"), inputs=norm_loudness_checkbox)

    # shared
    temperature_slider.change(update_setting("temperature_input"), inputs=temperature_slider)
    target_chars_slider.change(update_setting("target_chars_input"), inputs=target_chars_slider)
    max_chars_slider.change(update_setting("max_chars_input"), inputs=max_chars_slider)

if __name__ == "__main__":
    try:
        ensure_app_data_dir()  # Ensure it's created at startup
        logger.info(f"Application data directory '{APP_DATA_DIR}' ensured.")
    except Exception as e:
        logger.critical(f"Could not initialize application data directory '{APP_DATA_DIR}': {e}. App may not function correctly.", exc_info=True)
        # The app might still run if settings load uses hardcoded defaults and output goes to script dir.
    threading.Thread(target=run_fastapi, daemon=True).start()
    print(f"CONSOLE: Initial settings: Device={initial_settings['device']}, Seed={initial_settings['seed']}, Prompt={initial_settings['audio_prompt_path']}")
    demo.queue().launch(debug=True, share=False)
