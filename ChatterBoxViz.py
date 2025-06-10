import random
from typing import Optional, Tuple

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
from huggingface_hub import hf_hub_download
from chatterbox.tts import ChatterboxTTS
if not os.path.exists("outputs"):
    os.makedirs("outputs")
# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
PUNCTUATION_SPLIT_STRONGLY = ".!?"
PUNCTUATION_SPLIT_WEAKLY = ",;:"
APP_DATA_DIR = "app_data"
SETTINGS_FILE = os.path.join(APP_DATA_DIR, "settings.json")
PERSISTENT_PROMPT_FILENAME = "last_used_prompt.wav"
PERSISTENT_PROMPT_PATH = os.path.join(APP_DATA_DIR, PERSISTENT_PROMPT_FILENAME)

# Audio validation constants
MAX_AUDIO_DURATION_SECONDS = 30
MIN_AUDIO_DURATION_SECONDS = 0.5
SUPPORTED_SAMPLE_RATES = [16000, 22050, 24000, 44100, 48000]
MAX_PROMPT_AUDIO_FILE_SIZE_MB = 100 # Added for clarity
MIN_SAVED_AUDIO_FILE_SIZE_BYTES = 1024 # For output verification

MAX_TEXT_LENGTH = 50000
MIN_CHUNK_SIZE = 10
MAX_OUTPUT_FILE_SIZE_MB = 500 # Will be used to warn user


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
        raise # Re-raise the exception
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
    if not (0.0 <= cfg_weight <= 2.0):
        return False, f"CFG weight must be between 0.0 and 2.0 (got {cfg_weight})"

    if not (0.0 <= exaggeration <= 2.0):
        return False, f"Exaggeration must be between 0.0 and 2.0 (got {exaggeration})"

    if not (0.1 <= temperature <= 1.5):
        return False, f"Temperature must be between 0.1 and 1.5 (got {temperature})"

    if not (MIN_CHUNK_SIZE <= target_chars <= 500): # Max target chars can be adjusted
        return False, f"Target chars must be between {MIN_CHUNK_SIZE} and 500 (got {target_chars})"

    if not (target_chars < max_chars <= 1000): # Max max_chars can be adjusted
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
            check_path = '.' # Check current working directory if APP_DATA_DIR is not valid yet
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
        if not (0 <= seed <= 2**32 - 1): # Common range for seeds
            raise ValueError(f"Seed must be between 0 and 2**32 - 1, got {seed}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Global seed set to: {seed}")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        # Not raising gr.Error as it might be non-critical for some operations,
        # but it's important for reproducibility. The calling function should handle UI.
        raise


def ensure_app_data_dir():
    """Create app data directory with error handling"""
    try:
        app_data_path = Path(APP_DATA_DIR)
        app_data_path.mkdir(parents=True, exist_ok=True) # parents=True if APP_DATA_DIR can be nested

        # Test write permissions
        test_file = app_data_path / '.write_test'
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"No write permission to {APP_DATA_DIR}: {e}")

        logger.info(f"App data directory ready: {APP_DATA_DIR}")
    except Exception as e:
        logger.error(f"Error setting up app data directory: {e}")
        raise gr.Error(f"Cannot create/access application data directory '{APP_DATA_DIR}': {e}. Please check permissions.")


def save_settings(device, cfg, exg, temp, long_text, target_chars, max_chars, seed_value,
                  prompt_successfully_persisted_this_run):
    """Save settings only if they have changed, with atomic write and conditional backup."""
    temp_settings_file_path = None
    made_change = False  # Flag to track if a save actually happened

    try:
        ensure_app_data_dir()

        new_settings = {
            "device": device,
            "cfg_weight": float(cfg),
            "exaggeration": float(exg),
            "temperature": float(temp),
            "long_text": str(long_text)[:50000],  # Limit stored text length
            "target_chars_per_chunk": int(target_chars),
            "max_chars_per_chunk": int(max_chars),
            "seed": int(seed_value),
            "has_persistent_prompt": prompt_successfully_persisted_this_run and os.path.exists(PERSISTENT_PROMPT_PATH),
            "last_saved_timestamp": time.time()  # Use a more descriptive key
        }

        existing_settings = None
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f_old:
                    existing_settings = json.load(f_old)
                    # Normalize by re-dumping to compare consistently, or compare key by key
                    # For simplicity, direct dictionary comparison often works if types are consistent.
                    # However, float precision or order of keys can be an issue.
                    # A robust comparison might involve checking each key or converting to a canonical JSON string.

                    # Let's remove the 'last_saved_timestamp' for comparison, as it will always change.
                    existing_settings_for_comparison = existing_settings.copy()
                    existing_settings_for_comparison.pop("last_saved_timestamp", None)
                    new_settings_for_comparison = new_settings.copy()
                    new_settings_for_comparison.pop("last_saved_timestamp", None)

                    if existing_settings_for_comparison == new_settings_for_comparison:
                        logger.info(f"Settings have not changed. Skipping save to {SETTINGS_FILE}.")
                        return  # Exit if settings are identical (excluding timestamp)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode existing settings file '{SETTINGS_FILE}' for comparison. Will overwrite.")
                existing_settings = None  # Treat as if no valid existing settings
            except Exception as e_read:
                logger.warning(f"Error reading existing settings file '{SETTINGS_FILE}' for comparison: {e_read}. Will proceed to save/overwrite.")
                existing_settings = None

        logger.info(f"Settings have changed or no existing valid settings. Proceeding to save to {SETTINGS_FILE}.")
        made_change = True

        # Write to a temporary file in the same directory for atomic replace
        # Use 'delete=False' with NamedTemporaryFile to manage renaming/deletion manually
        # Or stick to mkstemp for more control as you had.
        fd, temp_settings_file_path = tempfile.mkstemp(suffix=".json", prefix="settings_tmp_", dir=APP_DATA_DIR)
        with os.fdopen(fd, 'w', encoding='utf-8') as f_temp:
            json.dump(new_settings, f_temp, indent=4, ensure_ascii=False)
        # fdopen closes the fd, so no need to os.close(fd)

        # Create backup of existing settings *only if we are actually overwriting different settings*
        if os.path.exists(SETTINGS_FILE):  # Check again before backup/move
            backup_file = f"{SETTINGS_FILE}.{int(time.time())}.backup"
            try:
                shutil.copy2(SETTINGS_FILE, backup_file)  # copy2 preserves metadata
                logger.info(f"Created settings backup: {backup_file}")
            except Exception as e_backup:
                logger.warning(f"Could not create settings backup for '{SETTINGS_FILE}': {e_backup}")

        # Atomically replace the old settings file with the new one
        shutil.move(temp_settings_file_path, SETTINGS_FILE)
        temp_settings_file_path = None  # Mark as moved, so finally block doesn't try to delete
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

        # Optional: Prune old backups if too many exist
        cleanup_old_backups(APP_DATA_DIR, SETTINGS_FILE, keep_latest_n=10)


# Optional: Function to clean up old backups
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
    """Load settings with comprehensive error handling and validation"""
    defaults = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cfg_weight": 0.2,
        "exaggeration": 0.8,
        "temperature": 0.8,
        "long_text": "",
        "target_chars_per_chunk": 100,
        "max_chars_per_chunk": 200,
        "seed": 0, # 0 means random
        "audio_prompt_path": None
    }

    try:
        ensure_app_data_dir() # Ensures APP_DATA_DIR exists

        if not os.path.exists(SETTINGS_FILE):
            logger.info(f"No settings file found at {SETTINGS_FILE}, using defaults.")
            return defaults

        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Validate and apply loaded settings
        loaded_values = {}
        for key, default_value in defaults.items():
            if key in settings:
                try:
                    value = settings[key]
                    if key in ["cfg_weight", "exaggeration", "temperature"]:
                        loaded_values[key] = float(value)
                    elif key in ["target_chars_per_chunk", "max_chars_per_chunk", "seed"]:
                        loaded_values[key] = int(value)
                    elif key == "device":
                        if value not in ["cuda", "cpu"]:
                            logger.warning(f"Invalid device '{value}' in settings, using default '{default_value}'.")
                            loaded_values[key] = default_value
                        elif value == "cuda" and not torch.cuda.is_available():
                            logger.warning("CUDA specified in settings but not available, switching to CPU.")
                            loaded_values[key] = "cpu"
                        else:
                            loaded_values[key] = value
                    elif key == "long_text":
                        loaded_values[key] = str(value) # Ensure it's a string
                    # audio_prompt_path is handled separately below
                    elif key != "audio_prompt_path":
                         loaded_values[key] = value # For other potential settings
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for '{key}' in settings ('{settings[key]}'), using default '{default_value}'. Error: {e}")
                    loaded_values[key] = default_value
            else:
                loaded_values[key] = default_value # Key not in settings, use default

        # Handle persistent prompt
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
                loaded_values["audio_prompt_path"] = None # Ensure it's None if removed
        else:
            loaded_values["audio_prompt_path"] = None # Default if no persistent prompt flag or file

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
        if os.path.exists(SETTINGS_FILE): # Check again before renaming
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
def _find_intelligent_split_point(text_segment: str, target_chars: int, max_chars: int) -> int:
    """Find optimal text split point with better error handling"""
    try:
        segment_len = len(text_segment)
        if segment_len <= max_chars:
            return segment_len

        # Define search range: try to split near target_chars, but within a window towards max_chars
        # Search backwards from a point slightly beyond target_chars, up to max_chars
        # Ideal search end: min(segment_len -1, max_chars -1)
        # Ideal search start: max(0, target_chars - some_leniency)
        search_end_idx = min(segment_len - 1, max_chars - 1)
        search_start_idx = max(0, target_chars - (max_chars - target_chars) // 2) # Start searching from a bit before target if possible
        search_start_idx = max(0, min(search_start_idx, search_end_idx -1)) # Ensure start_idx is valid and before end_idx

        # Search for strong punctuation (e.g., ".!?")
        for i in range(search_end_idx, search_start_idx - 1, -1):
            if text_segment[i] in PUNCTUATION_SPLIT_STRONGLY:
                # Ensure split is not mid-word if punctuation is followed by char
                if i + 1 < segment_len and text_segment[i+1].isspace():
                    return i + 1 # Include the punctuation, split after it
                elif i == segment_len -1: # Punctuation at the very end
                    return i + 1
                # If punctuation is followed by non-space, it might be part of word (e.g. "e.g."), prefer other splits

        # Search for weak punctuation (e.g., ",;:")
        for i in range(search_end_idx, search_start_idx - 1, -1):
            if text_segment[i] in PUNCTUATION_SPLIT_WEAKLY:
                if i + 1 < segment_len and text_segment[i+1].isspace():
                    return i + 1
                elif i == segment_len -1:
                    return i + 1

        # Search for whitespace (split between words)
        # Widen search for whitespace if no punctuation found
        whitespace_search_end_idx = min(segment_len -1, max_chars -1)
        whitespace_search_start_idx = max(0, MIN_CHUNK_SIZE -1) # Don't split too early for whitespace

        for i in range(whitespace_search_end_idx, whitespace_search_start_idx - 1, -1):
            if text_segment[i].isspace():
                return i + 1 # Split after the space

        # Fallback: hard split at max_chars if no better point found
        logger.warning(f"Could not find an ideal split point for segment of length {segment_len}, target {target_chars}, max {max_chars}. Using hard split at {max_chars}.")
        return min(segment_len, max_chars)

    except Exception as e:
        logger.error(f"Error in _find_intelligent_split_point: {e}. Falling back to max_chars.")
        return min(len(text_segment), max_chars)


def intelligent_chunk_text(long_text: str, target_chars: int, max_chars: int) -> list[str]:
    """Chunk text intelligently with comprehensive error handling"""
    try:
        if not long_text or not long_text.strip():
            return []

        long_text = re.sub(r'\s+', ' ', long_text).strip() # Normalize whitespace

        if not (MIN_CHUNK_SIZE <= target_chars < max_chars): # Basic sanity check
            logger.error(f"Invalid chunking parameters: target={target_chars}, max={max_chars}. Using fallback.")
            # Fallback to simple splitting or return error indicator if preferred
            return [long_text[i:i+max_chars] for i in range(0, len(long_text), max_chars)]


        sentences = sent_tokenize(long_text)
        if not sentences: # Should not happen if long_text is not empty
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

            if sentence_length > max_chars: # Sentence itself is too long
                if current_buffer: # Flush buffer before handling long sentence
                    chunks.append(" ".join(current_buffer).strip())
                    current_buffer = []
                    current_buffer_length = 0

                # Split the oversized sentence
                temp_segment_start_idx = 0
                while temp_segment_start_idx < sentence_length:
                    remaining_sentence_part = sentence[temp_segment_start_idx:]
                    if not remaining_sentence_part.strip(): break # Avoid empty trailing parts

                    if len(remaining_sentence_part) <= max_chars:
                        chunks.append(remaining_sentence_part.strip())
                        break
                    else:
                        split_at = _find_intelligent_split_point(remaining_sentence_part, target_chars, max_chars)
                        chunk_to_add = remaining_sentence_part[:split_at].strip()
                        if chunk_to_add: # Ensure non-empty chunk
                            chunks.append(chunk_to_add)
                        temp_segment_start_idx += split_at
                        # Skip any leading whitespace for the next part of the split sentence
                        while temp_segment_start_idx < sentence_length and sentence[temp_segment_start_idx].isspace():
                            temp_segment_start_idx += 1
            else: # Sentence fits or can be combined
                # If adding this sentence exceeds max_chars for the buffer
                if current_buffer_length + (1 if current_buffer else 0) + sentence_length > max_chars:
                    if current_buffer: # Flush current buffer
                        chunks.append(" ".join(current_buffer).strip())
                    current_buffer = [sentence] # Start new buffer with current sentence
                    current_buffer_length = sentence_length
                else: # Add sentence to buffer
                    current_buffer.append(sentence)
                    current_buffer_length += (1 if len(current_buffer) > 1 else 0) + sentence_length

                # If buffer is full enough (>= target_chars)
                if current_buffer_length >= target_chars:
                    chunks.append(" ".join(current_buffer).strip())
                    current_buffer = []
                    current_buffer_length = 0

        if current_buffer: # Flush any remaining sentences in buffer
            chunks.append(" ".join(current_buffer).strip())

        # Final filter for chunk size and non-empty
        valid_chunks = [chunk for chunk in chunks if MIN_CHUNK_SIZE <= len(chunk.strip()) <= max_chars]

        if not valid_chunks and long_text: # If all filtering removed everything, but there was text
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
            return [long_text[i:i+max_chars] for i in range(0, len(long_text), max_chars) if long_text[i:i+max_chars].strip()]
        except Exception as fallback_error:
            logger.error(f"Fallback chunking also failed: {fallback_error}")
            return []


# --- Model Management ---
#def ensure_model_loaded(device_choice: str, progress_fn): # progress_fn is the gr.Progress object
def ensure_model_loaded(device_choice: str):
    """Load TTS model with comprehensive error handling"""
    global tts_model, current_device_loaded

    try:
        if tts_model is not None and current_device_loaded == device_choice:
            logger.info(f"Model already loaded on {current_device_loaded}")
            # We don't need to call progress_fn here if model is already loaded,
            # as the main function will update status.
            return f"Model already loaded on {current_device_loaded}"

        # Call progress_fn without desc
        #progress_fn(0) # Indicates start of this sub-task
        logger.info(f"Attempting to load ChatterboxTTS model on device: {device_choice}")

        if device_choice == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA selected, but not available. Please select CPU or check CUDA installation.")
            try:
                torch.cuda.empty_cache()
                # Test allocation
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                logger.info("CUDA device seems responsive.")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    mem_info = torch.cuda.memory_stats() if torch.cuda.is_available() else {}
                    logger.error(f"CUDA OOM during test allocation: {e}. Memory stats: {mem_info}")
                    raise RuntimeError(f"Insufficient GPU memory to load model. Details: {e}")
                else:
                    logger.error(f"CUDA error during test allocation: {e}")
                    raise RuntimeError(f"GPU error, cannot load model. Details: {e}")
        elif device_choice != "cpu":
            raise ValueError(f"Invalid device choice: {device_choice}. Must be 'cuda' or 'cpu'.")


        # Suppress potential verbose C++ extension loading messages from Chatterbox or its deps
        with suppress_stdout_stderr():
            tts_model_candidate = ChatterboxTTS.from_pretrained(device=device_choice)
            #tts_model_candidate.conds.= tts_model_candidate.Conditionals(t3_cond, s3gen_ref_dict)


        tts_model = tts_model_candidate # Assign only after successful load
        current_device_loaded = device_choice
        # Call progress_fn without desc
        #progress_fn(1.0) # Indicates completion of this sub-task
        logger.info(f"Model loaded successfully on {current_device_loaded}")
        return f"Model loaded successfully on {current_device_loaded}"

    except Exception as e:
        # Detailed logging for different error types
        error_msg_lower = str(e).lower()
        specific_error = ""
        if "out of memory" in error_msg_lower:
            specific_error = "Out of memory. Try CPU, a smaller model, or free up memory."
        elif any(kw in error_msg_lower for kw in ["download", "connection", "url", "http"]):
            specific_error = "Network error during model download. Check internet connection and Hugging Face Hub status."
        elif "permission" in error_msg_lower or "access is denied" in error_msg_lower:
            specific_error = "File permission error. Check permissions for model cache directory (usually ~/.cache/huggingface/)."
        elif "safetensors_rust" in error_msg_lower or ".so" in error_msg_lower or "libc+" in error_msg_lower:
            specific_error = f"Error with compiled components (e.g. Safetensors). Ensure all dependencies are correctly installed for your OS/Python. Original error: {e}"

        full_error_message = f"Failed to load model: {specific_error if specific_error else e}"
        logger.error(f"{full_error_message}\n{traceback.format_exc()}")

        # Cleanup
        tts_model = None
        current_device_loaded = None
        if device_choice == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise gr.Error(full_error_message)


"""def simple_progress_test(progress=gr.Progress(track_tqdm=False)):
    print("Simple progress test started...")  # Server-side log
    status_updates = []
    dummy_audio_path = os.path.join("outputs", "dummy_audio.wav")  # Define path for dummy audio

    for i in range(101):  # 0 to 100
        time.sleep(0.02)  # Shorter sleep for faster testing
        progress(i / 100)  # Update progress from 0.0 to 1.0
        current_status = f"Progress: {i}%"
        # print(current_status) # Optional server-side log of progress

        # Yield status less frequently to avoid overwhelming the UI update mechanism
        # especially if that's also a bottleneck
        if i % 5 == 0 or i == 100:
            status_updates.append(current_status)
            yield None, "\n".join(status_updates)

    # Create a dummy audio file for the output component
    try:
        import torch
        import torchaudio as ta
        sample_rate = 22050
        audio_data = torch.zeros(1, sample_rate * 1)  # 1 second of silence
        ta.save(dummy_audio_path, audio_data, sample_rate)
        print(f"Dummy audio saved to {dummy_audio_path}")
    except Exception as e:
        print(f"Could not create dummy audio file: {e}")
        dummy_audio_path = None  # Fallback if torch/torchaudio not available or error

    status_updates.append("Simple test done!")
    print("Simple progress test finished.")
    yield dummy_audio_path, "\n".join(status_updates)


    with gr.Blocks() as demo:  # Simplest Blocks setup
        gr.Markdown("Simple Progress Bar Test")
        with gr.Row():
            # Input needed for .click() if inputs are specified, even if not used by fn
            # dummy_input = gr.Textbox(label="Dummy Input", value="click me")
            submit_button = gr.Button("Run Simple Test")
        with gr.Row():
            audio_output = gr.Audio(label="Test Audio Output")
            status_output = gr.Textbox(label="Test Status")
    
        submit_button.click(
            fn=simple_progress_test,
            inputs=[],  # No inputs for this test function
            outputs=[audio_output, status_output],
            show_progress="full"  # This is the key
        )"""

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


def _perform_tts_generation_loop_old(long_text_input, target_chars_input, max_chars_input,
                                 current_prompt_path_for_generation, cfg_weight_input,
                                 exaggeration_input, temperature_input, status_messages, progress):
    """Handles text chunking and the core TTS generation loop."""
    global tts_model  # Needs access to the global tts_model
    all_wavs = []
    failed_chunks_count = 0

    status_messages.append("🔄 Chunking text...")
    # yield None, "\n".join(status_messages) # yield from main
    try:
        chunks = intelligent_chunk_text(long_text_input, target_chars_input, max_chars_input)
        if not chunks:
            status_messages.append("❌ Error: Text chunking resulted in no processable chunks.")
            return None, failed_chunks_count  # Indicate error
        status_messages.append(f"Text divided into {len(chunks)} chunks.")
        progress(0.32)
        # yield None, "\n".join(status_messages) # yield from main

        total_chunks = len(chunks)
        loop_progress_start = 0.32
        loop_progress_range = 0.58

        for i, chunk_text in enumerate(chunks):
            current_chunk_progress_val = loop_progress_start + (loop_progress_range * (i / total_chunks))
            progress(current_chunk_progress_val)
            chunk_info_log = f"Synthesizing chunk {i + 1}/{total_chunks} (len {len(chunk_text)})..."
            # For immediate UI update of this specific message, the main function must yield
            # This helper can append to status_messages, and main function yields the whole list.
            # Or, the main function can construct this message and yield.
            # For now, let main function handle yielding of combined status_messages.
            print(f"CONSOLE: Chunk {i + 1}/{total_chunks} ('{chunk_text[:25].replace(chr(10), ' ')}...')")
            status_messages.append(chunk_info_log)  # Add to main list

            chunk_generated_successfully = False
            for attempt in range(1, 3):  # Retry loop
                try:
                    with suppress_stdout_stderr():
                        if current_prompt_path_for_generation is None:  # Inbuilt voice
                            wav_chunk = tts_model.generate(chunk_text, cfg_weight=cfg_weight_input, exaggeration=exaggeration_input, temperature=temperature_input)
                        else:  # User prompt
                            wav_chunk = tts_model.generate(
                                chunk_text, audio_prompt_path=current_prompt_path_for_generation,
                                cfg_weight=cfg_weight_input, exaggeration=exaggeration_input, temperature=temperature_input
                            )
                    all_wavs.append(wav_chunk)
                    chunk_generated_successfully = True
                    status_messages.append(f"✅ Chunk {i + 1}/{total_chunks} synthesized.")
                    break
                except Exception as e_chunk:
                    logger.warning(f"Attempt {attempt} for chunk {i + 1} failed.", exc_info=True if attempt == 2 else False)
                    if attempt == 2:
                        error_msg_chunk = f"❌ Error on chunk {i + 1}: {str(e_chunk)[:100]}"
                        status_messages.append(error_msg_chunk)
                        status_messages.append(f"⏭️ Skipping chunk {i + 1}. Adding 0.5s silence.")
                        failed_chunks_count += 1
                        silence_device = getattr(tts_model, 'device', "cpu")
                        silence_sr = getattr(tts_model, 'sr', 24000)
                        all_wavs.append(torch.zeros((1, int(silence_sr * 0.5)), device=silence_device))
                    else:
                        time.sleep(0.2)
            # yield None, "\n".join(status_messages) # Yield from main after each chunk status is appended

        progress(loop_progress_start + loop_progress_range)  # Loop finished (0.90)
        if not all_wavs or all(w.numel() == 0 for w in all_wavs):
            status_messages.append("❌ Error: No audio data generated after processing all chunks.")
            return None, failed_chunks_count
        if failed_chunks_count > 0:
            status_messages.append(f"⚠️ Gen completed with {failed_chunks_count}/{total_chunks} failed chunks.")
        return all_wavs, failed_chunks_count

    except Exception as e_gen_loop:
        status_messages.append(f"❌ Error during TTS generation loop: {e_gen_loop}")
        logger.error("TTS generation loop error", exc_info=True)
        return None, failed_chunks_count  # Or a specific error indicator


def _perform_tts_generation_loop(
        long_text_input: str, target_chars_input: int, max_chars_input: int,
        current_prompt_path_for_generation: Optional[str], cfg_weight_input: float,
        exaggeration_input: float, temperature_input: float,
        status_messages: list,  # Takes the main list of messages
        progress: gr.Progress
):
    """
    Handles text chunking and TTS generation loop.
    This is now a GENERATOR that yields the complete, updated status log.
    It returns its final result: (all_wavs_list, failed_chunks_count)
    """
    global tts_model

    # --- Chunking ---
    status_messages.append("🔄 Chunking text...")
    yield None, "\n".join(status_messages)  # Yield tuple (audio, status)
    try:
        chunks = intelligent_chunk_text(long_text_input, target_chars_input, max_chars_input)
        if not chunks:
            status_messages.append("❌ Error: Text chunking resulted in no processable chunks.")
            yield None, "\n".join(status_messages)
            return None, 0  # Return error indicator

        status_messages.append(f"Text divided into {len(chunks)} chunks.")
        progress(0.32, desc="Chunking complete.")
        yield None, "\n".join(status_messages)
    except Exception as e_chunk_text:
        status_messages.append(f"❌ Error during text chunking: {e_chunk_text}")
        logger.error("Text chunking failed", exc_info=True)
        yield None, "\n".join(status_messages)
        return None, 0

    # --- Generation Loop ---
    all_wavs = []
    failed_chunks_count = 0
    total_chunks = len(chunks)
    loop_progress_start = 0.32
    loop_progress_range = 0.58

    for i, chunk_text in enumerate(chunks):
        progress(loop_progress_start + (loop_progress_range * (i / total_chunks)), desc=f"Synthesizing chunk {i + 1}/{total_chunks}")

        # We can add a temporary message for this yield cycle without adding it permanently
        # to the main log, which keeps the final log cleaner.
        temp_status_for_yield = status_messages[:] + [f"Synthesizing chunk {i + 1}/{total_chunks} (len {len(chunk_text)})..."]
        yield None, "\n".join(temp_status_for_yield)

        print(f"CONSOLE: Chunk {i + 1}/{total_chunks} ('{chunk_text[:25].replace(chr(10), ' ')}...')")

        chunk_generated_successfully = False
        for attempt in range(1, 3):
            try:
                with suppress_stdout_stderr():
                    if current_prompt_path_for_generation is None:
                        wav_chunk = tts_model.generate(chunk_text, cfg_weight=cfg_weight_input, exaggeration=exaggeration_input, temperature=temperature_input)
                    else:
                        wav_chunk = tts_model.generate(
                            chunk_text, audio_prompt_path=current_prompt_path_for_generation,
                            cfg_weight=cfg_weight_input, exaggeration=exaggeration_input, temperature=temperature_input
                        )
                all_wavs.append(wav_chunk)
                chunk_generated_successfully = True
                status_messages.append(f"✅ Chunk {i + 1}/{total_chunks} synthesized.")  # Add permanent success message
                break
            except Exception as e_chunk:
                logger.warning(f"Attempt {attempt} for chunk {i + 1} failed.", exc_info=True if attempt == 2 else False)
                if attempt == 2:
                    status_messages.append(f"❌ Error on chunk {i + 1}: {str(e_chunk)[:100]}")
                    status_messages.append(f"⏭️ Skipping chunk {i + 1}. Adding 0.5s silence.")
                    failed_chunks_count += 1
                    silence_device = getattr(tts_model, 'device', "cpu");
                    silence_sr = getattr(tts_model, 'sr', 24000)
                    all_wavs.append(torch.zeros((1, int(silence_sr * 0.5)), device=silence_device))
                else:
                    time.sleep(0.2)

        # After each chunk is fully processed (succeeded or failed), yield the updated permanent log
        yield None, "\n".join(status_messages)

    progress(0.90, desc="All chunks processed.")

    # --- Final Checks for this stage ---
    if not all_wavs or all(w.numel() == 0 for w in all_wavs):
        status_messages.append("❌ Error: No audio data generated after processing all chunks.")
        yield None, "\n".join(status_messages)
        return None, failed_chunks_count

    if failed_chunks_count > 0:
        status_messages.append(f"⚠️ Gen completed with {failed_chunks_count}/{total_chunks} failed chunks.")
        yield None, "\n".join(status_messages)

    # Return the final data from this helper generator
    return all_wavs, failed_chunks_count


# --- Main Processing Function (Refactored) ---
def process_text_to_speech(
        long_text_input: str, audio_prompt_file_path: Optional[str],  # Can be None
        cfg_weight_input: float, exaggeration_input: float,
        temperature_input: float, target_chars_input: int, max_chars_input: int,
        device_choice: str, seed_num: int,
        progress: gr.Progress = gr.Progress(track_tqdm=False)
):
    status_messages = []
    generated_audio_final_path = None
    global tts_model  # Allow modification by ensure_model_loaded

    try:
        # Stage 0: App Data Dir
        progress(0, desc="Initializing...")  # Initial progress description
        try:
            ensure_app_data_dir()
            status_messages.append(f"✅ App data dir ready.")
        except Exception as e:  # Catch PermissionError or other setup errors
            status_messages.append(f"❌ Critical Error setting up app data: {str(e)}")
            logger.error("Critical Error setting up app data directory", exc_info=True)
            yield None, "\n".join(status_messages);
            return
        yield None, "\n".join(status_messages)

        # Stage 1: Initial Validations
        if not _validate_initial_inputs(long_text_input, cfg_weight_input, exaggeration_input, temperature_input,
                                        target_chars_input, max_chars_input, status_messages, progress):
            yield None, "\n".join(status_messages);
            return  # _validate_initial_inputs appends error
        yield None, "\n".join(status_messages)

        # Stage 2: Model Loading
        progress(0.05, desc="Loading model...")  # Update progress description
        status_messages.append(f"🔄 Loading TTS model on {device_choice}...")
        yield None, "\n".join(status_messages)
        try:
            model_load_status = ensure_model_loaded(device_choice)  # No progress obj passed here
            status_messages.append(f"✅ {model_load_status}")
        except gr.Error as e_gr:
            status_messages.append(f"❌ Model loading failed: {str(e_gr)}")
            yield None, "\n".join(status_messages);
            return
        except Exception as e_model_load:  # Other unexpected errors
            status_messages.append(f"❌ Unexpected model loading error: {str(e_model_load)}")
            logger.error("Unexpected model loading error", exc_info=True)
            yield None, "\n".join(status_messages);
            return
        progress(0.20, desc="Model loaded.")
        yield None, "\n".join(status_messages)

        # Stage 3: Audio Prompt Handling
        current_prompt_path_for_generation, prompt_successfully_persisted_this_run = _handle_audio_prompt(
            audio_prompt_file_path, status_messages, progress
        )
        if current_prompt_path_for_generation is False:  # _handle_audio_prompt signals error
            yield None, "\n".join(status_messages);
            return
        yield None, "\n".join(status_messages)  # Show messages from _handle_audio_prompt

        # Handle inbuilt voice if no prompt path was resolved and model is loaded
        if current_prompt_path_for_generation is None and tts_model:
            status_messages.append("✅ Setting up inbuilt voice conditioning...")
            yield None, "\n".join(status_messages)
            try:
                REPO_ID = "ResembleAI/chatterbox";
                conds_filename = "conds.pt"
                with suppress_stdout_stderr():
                    local_path_conds = hf_hub_download(repo_id=REPO_ID, filename=conds_filename)
                load_device = current_device_loaded if current_device_loaded else device_choice
                if hasattr(tts_model, 'conds') and hasattr(tts_model.conds, 'load'):
                    with suppress_stdout_stderr():
                        tts_model.conds = tts_model.conds.load(local_path_conds, load_device)
                    status_messages.append(f"✅ Inbuilt voice conditioning loaded for '{load_device}'.")
                else:
                    status_messages.append(f"⚠️ Model has no 'conds.load' for inbuilt voice.")
            except Exception as e_inbuilt:
                status_messages.append(f"❌ Error setting up inbuilt voice: {str(e_inbuilt)[:100]}...")
                logger.error("Error setting up inbuilt voice", exc_info=True)
            yield None, "\n".join(status_messages)

        # Stage 4: Setup Generation Parameters & Seed
        _setup_generation_parameters_and_seed(status_messages, seed_num, cfg_weight_input, exaggeration_input, temperature_input,
                                              target_chars_input, max_chars_input, current_prompt_path_for_generation, progress)
        yield None, "\n".join(status_messages)  # Show messages from _setup_generation_parameters_and_seed

        # Stage 5: Main Generation Loop
        progress(0.30, desc="Starting TTS Synthesis...")
        """all_wavs, failed_chunks_count = _perform_tts_generation_loop(
            long_text_input, target_chars_input, max_chars_input, current_prompt_path_for_generation,
            cfg_weight_input, exaggeration_input, temperature_input, status_messages, progress
        )"""
        all_wavs, failed_chunks_count = yield from _perform_tts_generation_loop(
            long_text_input, target_chars_input, max_chars_input,
            current_prompt_path_for_generation, cfg_weight_input,
            exaggeration_input, temperature_input,
            status_messages,  # Pass the list to be appended to
            progress
        )
        if all_wavs is None:
            # Error occurred in the loop, status is already updated by the last yield.
            return  # End the main generator

        # _perform_tts_generation_loop appends its own status messages.
        # We yield after each chunk inside the main loop now.

        # Loop for chunks (inside _perform_tts_generation_loop)
        # For now, let's assume _perform_tts_generation_loop handles its internal progress updates and status messages
        # The main function will just yield the accumulated status_messages from it.

        # Simulate the loop by yielding status from the helper
        # In a real scenario, _perform_tts_generation_loop would be a generator itself or update status_messages directly
        # For this refactor, we'll make _perform_tts_generation_loop append to status_messages and update progress.
        # The main function will yield the status_messages list.

        # Corrected flow: _perform_tts_generation_loop does the work and updates status_messages.
        # The main loop needs to yield these messages.
        # Let's adjust _perform_tts_generation_loop to be a generator or call yield here.
        # For simplicity now, assume _perform_tts_generation_loop has updated status_messages.
        # We'll refine this if _perform_tts_generation_loop needs to yield.

        # This part is tricky without making _perform_tts_generation_loop a generator itself that yields status.
        # For now, let's assume status_messages are updated by it and we yield periodically.
        # A better refactor would make helper functions generators too if they have long sub-steps.

        # Let's simplify: The main generation loop within process_text_to_speech
        # (The following is moved from the original _perform_tts_generation_loop for direct yield)

        """status_messages.append("🔄 Chunking text for generation...")
        yield None, "\n".join(status_messages)
        chunks = intelligent_chunk_text(long_text_input, target_chars_input, max_chars_input)
        if not chunks:
            status_messages.append("❌ Error: Text chunking resulted in no processable chunks.")
            yield None, "\n".join(status_messages);
            return
        status_messages.append(f"Text divided into {len(chunks)} chunks.")
        progress(0.32, desc="Chunking complete.")
        yield None, "\n".join(status_messages)

        all_wavs_gen = []  # Renamed to avoid conflict with potential return from helper
        total_chunks_gen = len(chunks)
        failed_chunks_count_gen = 0
        loop_progress_start = 0.32
        loop_progress_range = 0.58

        for i, chunk_text in enumerate(chunks):
            current_chunk_progress_val = loop_progress_start + (loop_progress_range * (i / total_chunks_gen))
            progress(current_chunk_progress_val, desc=f"Synthesizing chunk {i + 1}/{total_chunks_gen}")

            chunk_info_log = f"Synthesizing chunk {i + 1}/{total_chunks_gen} (len {len(chunk_text)} chars)..."
            # We need to yield the *current full log*
            # Create a temporary list for this yield including the new chunk_info_log
            current_yield_messages = status_messages[:] + [chunk_info_log]
            yield None, "\n".join(current_yield_messages)
            print(f"CONSOLE: Chunk {i + 1}/{total_chunks_gen} ('{chunk_text[:25].replace(chr(10), ' ')}...')")

            # Actual generation
            chunk_generated_successfully = False
            # Retry loop (kept from your original)
            for attempt in range(1, 3):
                try:
                    with suppress_stdout_stderr():
                        if current_prompt_path_for_generation is None:
                            wav_chunk = tts_model.generate(chunk_text, cfg_weight=cfg_weight_input, exaggeration=exaggeration_input, temperature=temperature_input)
                        else:
                            wav_chunk = tts_model.generate(
                                chunk_text, audio_prompt_path=current_prompt_path_for_generation,
                                cfg_weight=cfg_weight_input, exaggeration=exaggeration_input, temperature=temperature_input
                            )
                    all_wavs_gen.append(wav_chunk)
                    chunk_generated_successfully = True
                    status_messages.append(f"✅ Chunk {i + 1}/{total_chunks_gen} synthesized.")  # Add to permanent log
                    break
                except Exception as e_chunk:
                    logger.warning(f"Attempt {attempt} for chunk {i + 1} failed.", exc_info=True if attempt == 2 else False)
                    if attempt == 2:
                        error_msg_chunk = f"❌ Error on chunk {i + 1}: {str(e_chunk)[:100]}"
                        status_messages.append(error_msg_chunk)  # Add to permanent log
                        status_messages.append(f"⏭️ Skipping chunk {i + 1}. Adding 0.5s silence.")
                        failed_chunks_count_gen += 1
                        silence_device = getattr(tts_model, 'device', "cpu");
                        silence_sr = getattr(tts_model, 'sr', 24000)
                        all_wavs_gen.append(torch.zeros((1, int(silence_sr * 0.5)), device=silence_device))
                    else:
                        time.sleep(0.2)
            # Yield the permanent log AFTER processing the chunk (success or failure)
            yield None, "\n".join(status_messages)

        progress(0.90, desc="All chunks processed.")  # Loop finished
        yield None, "\n".join(status_messages)

        if not all_wavs_gen or all(w.numel() == 0 for w in all_wavs_gen):
            status_messages.append("❌ Error: No audio data generated after processing all chunks.")
            yield None, "\n".join(status_messages);
            return

        if failed_chunks_count_gen > 0:
            status_messages.append(f"⚠️ Gen completed with {failed_chunks_count_gen}/{total_chunks_gen} failed chunks.")"""

        # End of main generation loop section that was refactored back in

        # Stage 6: Merging and Saving
        progress(0.90, desc="Merging audio...")
        status_messages.append("🔊 Merging audio chunks...")
        yield None, "\n".join(status_messages)
        # ... (Your existing merge and save logic - ensure it updates status_messages and handles errors) ...
        # For brevity, assuming merge_and_save would be a helper or this logic is here:
        try:
            merge_device = getattr(tts_model, 'device', 'cpu')
            processed_wavs = [wav.to(merge_device) for wav in all_wavs]
            merged_wav = torch.cat(processed_wavs, dim=1)
        except RuntimeError as e_merge:  # OOM etc.
            status_messages.append(f"❌ Error merging audio: {str(e_merge)}. Try CPU merge.")
            yield None, "\n".join(status_messages)
            try:  # CPU Merge Fallback
                cpu_wavs = [wav.cpu() for wav in all_wavs]
                merged_wav = torch.cat(cpu_wavs, dim=1)
                status_messages.append("✅ Merged on CPU after initial failure.")
            except Exception as e_cpu_merge:
                status_messages.append(f"❌ CPU merge also failed: {e_cpu_merge}")
                yield None, "\n".join(status_messages);
                return
        progress(0.95, desc="Merging complete.")
        yield None, "\n".join(status_messages)

        if merged_wav.numel() == 0:
            status_messages.append("❌ Error: Merged audio is empty.")
            yield None, "\n".join(status_messages);
            return

        status_messages.append("🔄 Saving audio...")
        yield None, "\n".join(status_messages)
        output_dir = os.path.join(APP_DATA_DIR, "outputs");
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time());
        output_filename = f"chatterbox_output_{timestamp}.wav"
        output_filepath = os.path.join(output_dir, output_filename)
        try:
            sample_rate = getattr(tts_model, 'sr', 24000)
            ta.save(output_filepath, merged_wav.cpu(), sample_rate)
            if not (os.path.exists(output_filepath) and os.path.getsize(output_filepath) >= MIN_SAVED_AUDIO_FILE_SIZE_BYTES):
                raise IOError("Saved file missing or too small.")
            generated_audio_final_path = output_filepath
            status_messages.append(f"✅ Audio saved: {output_filepath}")
        except Exception as e_save:  # Handle primary save error, try backup
            status_messages.append(f"⚠️ Error saving to primary: {e_save}. Trying backup.")
            logger.warning(f"Primary save failed. Trying backup.", exc_info=True)
            backup_output_filepath = os.path.join(os.getcwd(), f"chatterbox_output_backup_{timestamp}.wav")
            try:
                ta.save(backup_output_filepath, merged_wav.cpu(), sample_rate)
                if not (os.path.exists(backup_output_filepath) and os.path.getsize(backup_output_filepath) >= MIN_SAVED_AUDIO_FILE_SIZE_BYTES):
                    raise IOError("Backup saved file missing or too small.")
                generated_audio_final_path = backup_output_filepath
                status_messages.append(f"✅ Audio saved to backup: {backup_output_filepath}")
            except Exception as e_backup_save:
                status_messages.append(f"❌ Error saving to backup: {e_backup_save}")
                yield None, "\n".join(status_messages);
                return

        progress(0.98, desc="Saving complete.")
        duration_secs = merged_wav.shape[1] / sample_rate
        final_msg = f"🎉 Success! Audio duration: {duration_secs:.2f}s. Output: {os.path.basename(generated_audio_final_path)}"
        if failed_chunks_count > 0: final_msg += f" (with {failed_chunks_count} silent chunks)"
        status_messages.append(final_msg)
        print(f"CONSOLE: {final_msg}")

        save_settings(device_choice, cfg_weight_input, exaggeration_input, temperature_input,
                      long_text_input, target_chars_input, max_chars_input,
                      int(seed_num), prompt_successfully_persisted_this_run)

        status_messages.append("✅ Done!")
        progress(1.0, desc="Completed!")
        # Final yield that becomes the return value for the generator
        yield generated_audio_final_path, "\n".join(status_messages)
        return  # Explicit return to signal end of generator

    except Exception as e_outer:
        error_msg = f"❌ Critical error in TTS process: {e_outer}"
        print(f"CONSOLE: {error_msg}\n{traceback.format_exc()}");
        logger.error("Critical TTS error", exc_info=True)
        status_messages.append(error_msg)
        yield None, "\n".join(status_messages);
        return  # Yield final error status
    finally:
        if tts_model and current_device_loaded == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache(); logger.info("CUDA cache cleared.")
            except Exception as e_cache:
                logger.warning(f"Could not clear CUDA cache: {e_cache}")
        logger.info("TTS generation process finished.")



# --- Function to Clear Persistent Prompt ---
def clear_persistent_prompt_action(
    current_audio_prompt_path: Optional[str], # ADDED: To receive current value of audio_prompt_input
    current_status_text_val: str # Existing input
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
            save_settings( # This will now save with has_persistent_prompt = False
                device=current_saved_settings["device"], cfg=current_saved_settings["cfg_weight"],
                exg=current_saved_settings["exaggeration"], temp=current_saved_settings["temperature"],
                long_text=current_saved_settings["long_text"], target_chars=current_saved_settings["target_chars_per_chunk"],
                max_chars=current_saved_settings["max_chars_per_chunk"], seed_value=current_saved_settings["seed"],
                prompt_successfully_persisted_this_run=False # THIS IS THE KEY CHANGE for settings
            )
            messages.append("✅ Settings updated to reflect no active persistent prompt.")
            settings_updated = True
        else:
            messages.append("ℹ️ Settings already indicate no active persistent prompt.")
            settings_updated = True # State is already as desired

        if cleared_physically and settings_updated:
            # Return update for the audio prompt input (to clear it) and the status text
            return gr.update(value=None), "\n".join(messages)
        else:
            # Should ideally always be true if logic is correct, but as a fallback
            return gr.update(value=None), "\n".join(messages) # Still attempt to clear UI field

    except Exception as e:
        logger.error(f"Error clearing persistent prompt: {e}", exc_info=True)
        messages.append(f"❌ Error clearing persistent prompt: {e}")
        # Return current (or no change) for audio_prompt_input, and error for status
        return gr.update(), "\n".join(messages) # gr.update() with no args means no change

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

try:
    initial_settings = load_settings()
except Exception as e:
    logger.critical(f"FATAL: Could not load initial settings: {e}. Using hardcoded defaults.", exc_info=True)
    initial_settings = {
        "device": "cpu", "cfg_weight": 0.2, "exaggeration": 0.8, "temperature": 0.8,
        "long_text": "Welcome! Settings failed to load.",
        "target_chars_per_chunk": 100, "max_chars_per_chunk": 200, "seed": 0,
        "audio_prompt_path": None
    }

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky),fill_width=True) as demo:
    gr.Markdown("<h1 align='center'>ChatterboxTTS Long Text Synthesizer</h1>"
                "Synthesize long text. Upload text, a WAV voice prompt, adjust parameters, and generate!")

    # This is the main row whose columns you want to fix
    with gr.Row(elem_id="main_content_row"):
        with gr.Column(scale=3): # Input Column Left
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
                    label="🎤 Voice Prompt WAV File (Max 100MB, 0.5-30s)", #...
                    value=_initial_audio_prompt_ui
                )
            clear_prompt_button = gr.Button("🧹 Clear Saved Voice Prompt", variant="stop", size="sm")

        with gr.Column(scale=2): # Input Column Right
            gr.Markdown("### ⚙️ Generation Parameters")
            device_input = gr.Dropdown(
                label="🧠 Processing Device", choices=["cuda", "cpu"],
                value=initial_settings["device"], info="Select 'cuda' for GPU or 'cpu'."
            )
            cfg_weight_slider = gr.Slider(#...
                value=initial_settings["cfg_weight"])
            exaggeration_slider = gr.Slider(#...
                value=initial_settings["exaggeration"])
            temperature_slider = gr.Slider(#...
                value=initial_settings["temperature"])
            seed_slider = gr.Slider(#...
                value=initial_settings["seed"]
            )
            gr.Markdown("### ✂️ Chunking Parameters")
            target_chars_slider = gr.Slider(#...
                value=initial_settings["target_chars_per_chunk"])
            max_chars_slider = gr.Slider(#...
                value=initial_settings["max_chars_per_chunk"])

    with gr.Row():
        submit_button = gr.Button("🚀 Generate Speech", variant="primary", scale=1)

    with gr.Row(elem_id="main_content_row"):
        with gr.Column(scale=3):
            # Define audio_output directly where it should appear in the layout
            audio_output = gr.Audio(label="🎧 Generated Speech Output", type="filepath")
        with gr.Column(scale=2):
            # Define status_output directly where it should appear in the layout
            status_output = gr.Textbox(
                label="📊 Log / Status Updates",
                lines=15,
                interactive=False,
                max_lines=20,
                autoscroll=True
            )

    # Event listener for submit button
    submit_button.click(
        fn=process_text_to_speech,
        inputs=[
            text_input, audio_prompt_input,
            cfg_weight_slider, exaggeration_slider, temperature_slider,
            target_chars_slider, max_chars_slider,
            device_input, seed_slider
        ],
        outputs=[audio_output, status_output],
        show_progress_on=audio_output,  # Link spinner to audio_output
        show_progress="full"  # 'full' shows progress bar and percentage updates
    )

    # Event listener for the clear button
    clear_prompt_button.click(
        fn=clear_persistent_prompt_action,
        # Ensure audio_prompt_input is an input for interactivity
        inputs=[audio_prompt_input, status_output],
        outputs=[audio_prompt_input, status_output]  # Update audio input field and status
    )

    gr.Markdown("---")
    gr.Markdown(f"💡 **Note:** Settings in `{APP_DATA_DIR}`. Outputs in `{os.path.join(APP_DATA_DIR, 'outputs')}`.")

if __name__ == "__main__":
    try:
        ensure_app_data_dir()  # Ensure it's created at startup
        logger.info(f"Application data directory '{APP_DATA_DIR}' ensured.")
    except Exception as e:
        logger.critical(f"Could not initialize application data directory '{APP_DATA_DIR}': {e}. App may not function correctly.", exc_info=True)
        # The app might still run if settings load uses hardcoded defaults and output goes to script dir.

    print(f"CONSOLE: Initial settings: Device={initial_settings['device']}, Seed={initial_settings['seed']}, Prompt={initial_settings['audio_prompt_path']}")
    demo.queue().launch(debug=True, share=False)