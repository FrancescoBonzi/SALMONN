# prepare_afthink.py
"""
Prepare NVIDIA AF-Think dataset for SALMONN training.
Dataset: https://huggingface.co/datasets/nvidia/AF-Think

AF-Think provides audio reasoning annotations (chain-of-thought) but does NOT
include the actual audio files. Audio must be downloaded from the original
source datasets.

This script:
1. Downloads AF-Think annotations from HuggingFace
2. Downloads audio from source datasets (when available on HuggingFace)
3. Matches annotations with audio files
4. Creates SALMONN-compatible annotation JSON files
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from datasets import load_dataset, Audio
from huggingface_hub import hf_hub_download, list_repo_files

from recipes.afthink.answer_extractor import (
    clean_question,
    extract_choices_from_question,
    format_answer_with_choice,
)


@dataclass
class SourceDatasetConfig:
    """Configuration for a source audio dataset."""
    hf_repo: Optional[str]  # HuggingFace repo (None if manual download required)
    hf_name: Optional[str]  # Dataset config name (if any)
    hf_split: Optional[str]  # Split to use
    audio_column: str  # Column containing audio
    id_column: Optional[str]  # Column to match with AF-Think IDs
    id_transform: Optional[Callable[[Any], str]]  # Transform ID to match AF-Think format
    manual_url: Optional[str]  # URL for manual download if not on HF
    notes: str  # Additional notes


# Mapping from AF-Think splits to their source datasets
# Many datasets require manual download or have complex setups
SOURCE_DATASETS: Dict[str, SourceDatasetConfig] = {
    # === Datasets available on HuggingFace ===
    "urbansound8k": SourceDatasetConfig(
        hf_repo="danavery/urbansound8K",
        hf_name=None,
        hf_split="train",
        audio_column="audio",
        id_column="slice_file_name",
        id_transform=lambda x: x.replace(".wav", ""),
        manual_url=None,
        notes="Urban sound classification dataset",
    ),
    "esc50": SourceDatasetConfig(
        hf_repo="ashraq/esc50",
        hf_name=None,
        hf_split="train",
        audio_column="audio",
        id_column="filename",
        id_transform=lambda x: x.replace(".wav", ""),
        manual_url=None,
        notes="Environmental Sound Classification",
    ),
    "af_cot_train_esc50": SourceDatasetConfig(
        hf_repo="ashraq/esc50",
        hf_name=None,
        hf_split="train",
        audio_column="audio",
        id_column="filename",
        id_transform=lambda x: x.replace(".wav", ""),
        manual_url=None,
        notes="ESC-50 for CoT training",
    ),
    "meld": SourceDatasetConfig(
        hf_repo="declare-lab/MELD",
        hf_name=None,
        hf_split="train",
        audio_column="audio",
        id_column="Utterance_ID",
        id_transform=None,
        manual_url=None,
        notes="Multimodal EmotionLines Dataset",
    ),
    "af_cot_train_clotho_v2": SourceDatasetConfig(
        hf_repo="d0rj/clotho-dataset",
        hf_name=None,
        hf_split="development",
        audio_column="audio",
        id_column="file_name",
        id_transform=lambda x: x.replace(".wav", ""),
        manual_url=None,
        notes="Clotho audio captioning dataset",
    ),
    "af_cot_train_gtzan": SourceDatasetConfig(
        hf_repo="marsyas/gtzan",
        hf_name=None,
        hf_split="train",
        audio_column="audio",
        id_column="file",
        id_transform=lambda x: Path(x).stem if x else None,
        manual_url=None,
        notes="GTZAN music genre dataset",
    ),
    
    # === Datasets requiring manual download or special handling ===
    "audioset": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://research.google.com/audioset/download.html",
        notes="Requires manual download from Google. Large-scale audio event dataset.",
    ),
    "audioset_sl": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://research.google.com/audioset/download.html",
        notes="AudioSet strong labels subset",
    ),
    "af_cot_train_audioset": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://research.google.com/audioset/download.html",
        notes="AudioSet for CoT training",
    ),
    "af_cot_train_audioset_sl": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://research.google.com/audioset/download.html",
        notes="AudioSet strong labels for CoT training",
    ),
    "musiccaps": SourceDatasetConfig(
        hf_repo="google/MusicCaps",
        hf_name=None,
        hf_split="train",
        audio_column=None,  # MusicCaps only has YouTube IDs, audio must be downloaded
        id_column="ytid",
        id_transform=None,
        manual_url="https://www.kaggle.com/datasets/googleai/musiccaps",
        notes="Audio must be downloaded from YouTube using ytid. See google/MusicCaps.",
    ),
    "vggsound": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://www.robots.ox.ac.uk/~vgg/data/vggsound/",
        notes="Requires manual download from VGG",
    ),
    "af_cot_train_fsd50k": SourceDatasetConfig(
        hf_repo="Fhrozen/FSD50k",
        hf_name=None,
        hf_split="train",
        audio_column="audio",
        id_column="filename",
        id_transform=lambda x: x.replace(".wav", ""),
        manual_url="https://zenodo.org/record/4060432",
        notes="Freesound Dataset 50k",
    ),
    "af_cot_train_fma": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://github.com/mdeff/fma",
        notes="Free Music Archive - requires manual download",
    ),
    "af_cot_train_freesound": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://freesound.org/",
        notes="Freesound - requires API access or manual download",
    ),
    "af_cot_train_bbc_sound_effects": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://sound-effects.bbcrewind.co.uk/",
        notes="BBC Sound Effects - requires manual download",
    ),
    "af_cot_train_chime_home": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://archive.org/details/chime-home",
        notes="CHiME-Home dataset",
    ),
    "af_cot_train_clotho_aqa": SourceDatasetConfig(
        hf_repo="d0rj/clotho-dataset",
        hf_name=None,
        hf_split="development",
        audio_column="audio",
        id_column="file_name",
        id_transform=lambda x: x.replace(".wav", ""),
        manual_url=None,
        notes="Clotho AQA subset",
    ),
    "af_cot_train_cochlscene": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://github.com/cochl/CochlScene",
        notes="CochlScene acoustic scene dataset",
    ),
    "msdfreesound": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://freesound.org/",
        notes="MSD-Freesound subset",
    ),
    "wavtext5k": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url=None,
        notes="WavText5K dataset",
    ),
    "tut_urbanswitchboard": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://webpages.tuni.fi/arg/dcase2016/task-acoustic-scene-classification",
        notes="TUT Urban Acoustic Scenes",
    ),
    "fisher": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://catalog.ldc.upenn.edu/LDC2004T19",
        notes="Fisher corpus - requires LDC license",
    ),
    "nissaf_cot_train_audioset": SourceDatasetConfig(
        hf_repo=None,
        hf_name=None,
        hf_split=None,
        audio_column="audio",
        id_column=None,
        id_transform=None,
        manual_url="https://research.google.com/audioset/download.html",
        notes="NISSA AudioSet subset",
    ),
}

# All available AF-Think splits
AFTHINK_SPLITS = list(SOURCE_DATASETS.keys())

# Splits with HuggingFace audio available (can be auto-downloaded)
AUTO_DOWNLOAD_SPLITS = [
    name for name, cfg in SOURCE_DATASETS.items() 
    if cfg.hf_repo is not None and cfg.audio_column is not None
]

# Splits requiring manual audio download
MANUAL_DOWNLOAD_SPLITS = [
    name for name, cfg in SOURCE_DATASETS.items()
    if cfg.hf_repo is None or cfg.audio_column is None
]

# Default train/eval splits
TRAIN_SPLITS = [s for s in AFTHINK_SPLITS if "train" in s or "cot" in s]
EVAL_SPLITS = ["urbansound8k", "esc50", "audioset", "musiccaps"]


def parse_conversations(conversations: Any) -> Dict[str, str]:
    """
    Parse the conversation format from AF-Think into question/answer pairs.
    Handles both list of dicts and JSON string formats.
    """
    question = ""
    answer = ""
    
    # Handle case where conversations is a JSON string
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except json.JSONDecodeError:
            return {"question": "", "answer": conversations}
    
    # Handle case where it's already a list
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict):
                if turn.get("from") == "human":
                    question = turn.get("value", "").replace("<sound>", "").strip()
                    question = " ".join(question.split())
                    # Remove boilerplate instructions
                    question = clean_question(question)
                elif turn.get("from") == "gpt":
                    answer = turn.get("value", "").strip()
    
    # Extract and append the correct answer choice (for multiple choice questions)
    if question and answer:
        answer = format_answer_with_choice(question, answer)
    
    return {"question": question, "answer": answer}


def load_afthink_from_jsonl(split: str, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load AF-Think data directly from JSONL files on HuggingFace.
    This bypasses the broken dataset loading mechanism.
    
    Args:
        split: The split name (e.g., 'urbansound8k')
        cache_dir: Directory to cache downloaded files
        
    Returns:
        List of sample dictionaries
    """
    repo_id = "nvidia/AF-Think"
    
    # Try to find the JSONL file for this split
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
        
        # Look for matching JSONL file
        jsonl_files = [f for f in files if f.endswith('.jsonl') or f.endswith('.json')]
        
        # Find file matching the split
        matching_file = None
        for f in jsonl_files:
            if split in f.lower():
                matching_file = f
                break
        
        if matching_file is None:
            # Try the data directory structure
            possible_paths = [
                f"data/{split}.jsonl",
                f"data/afthink/{split}.jsonl",
                f"{split}.jsonl",
                f"data/{split}.json",
            ]
            for path in possible_paths:
                if path in files:
                    matching_file = path
                    break
        
        if matching_file is None:
            print(f"    Could not find JSONL file for split: {split}")
            print(f"    Available files: {[f for f in files if 'json' in f.lower()][:10]}")
            return []
        
        # Download the file
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=matching_file,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        
        # Read JSONL file
        samples = []
        with open(local_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        print(f"    Loaded {len(samples)} samples from {matching_file}")
        return samples
        
    except Exception as e:
        print(f"    Error loading JSONL: {e}")
        return []


def load_afthink_split(split: str) -> List[Dict[str, Any]]:
    """
    Load AF-Think annotations for a split.
    Tries multiple methods:
    1. Standard dataset loading
    2. Streaming mode
    3. Direct JSONL download
    
    Args:
        split: The split name
        
    Returns:
        List of sample dictionaries
    """
    # Method 1: Try standard loading (without trust_remote_code)
    try:
        print(f"    Trying standard load...")
        ds = load_dataset(
            "nvidia/AF-Think",
            name="afthink",
            split=split,
        )
        return [dict(sample) for sample in ds]
    except Exception as e:
        print(f"    Standard load failed: {e}")
    
    # Method 2: Try streaming mode
    try:
        print(f"    Trying streaming mode...")
        ds = load_dataset(
            "nvidia/AF-Think",
            name="afthink",
            split=split,
            streaming=True,
        )
        samples = []
        for sample in ds:
            samples.append(dict(sample))
            if len(samples) >= 100000:  # Safety limit
                break
        return samples
    except Exception as e:
        print(f"    Streaming failed: {e}")
    
    # Method 3: Try loading from raw JSONL files
    print(f"    Trying direct JSONL download...")
    return load_afthink_from_jsonl(split)


def download_source_audio(
    split: str,
    audio_dir: str,
    required_audio_ids: Optional[set] = None,
) -> Dict[str, str]:
    """
    Download audio from source dataset and return mapping of ID -> audio path.
    
    Uses huggingface_hub to download raw audio files directly, avoiding
    the datasets library's automatic audio decoding which requires torchcodec.
    
    Args:
        split: The AF-Think split name
        audio_dir: Directory to save audio files
        required_audio_ids: Set of audio IDs to download (from AF-Think annotations)
    
    Returns:
        Dictionary mapping sample IDs to local audio file paths
    """
    import librosa
    import soundfile as sf
    import numpy as np
    from huggingface_hub import snapshot_download, hf_hub_download
    import io
    import pyarrow.parquet as pq
    
    config = SOURCE_DATASETS.get(split)
    if config is None:
        print(f"  Unknown split: {split}")
        return {}
    
    if config.hf_repo is None:
        print(f"  {split}: No HuggingFace repo available")
        print(f"    Manual download required: {config.manual_url}")
        print(f"    Notes: {config.notes}")
        return {}
    
    if config.audio_column is None:
        print(f"  {split}: Audio column not available in HF dataset")
        print(f"    Notes: {config.notes}")
        return {}
    
    split_audio_dir = os.path.join(audio_dir, split)
    os.makedirs(split_audio_dir, exist_ok=True)
    
    print(f"  Downloading from {config.hf_repo}...")
    
    id_to_path = {}
    
    try:
        # Method 1: Try to download parquet files directly and extract audio
        print(f"    Downloading dataset files...")
        
        # List files in the repo
        files = list_repo_files(config.hf_repo, repo_type="dataset")
        
        # Find parquet or audio files
        parquet_files = [f for f in files if f.endswith('.parquet')]
        audio_files = [f for f in files if any(f.endswith(ext) for ext in ['.wav', '.flac', '.mp3', '.ogg'])]
        
        if audio_files:
            # Dataset stores audio as separate files - download them directly
            print(f"    Found {len(audio_files)} audio files in repo")
            
            # Filter to only needed files
            if required_audio_ids:
                # Match audio files to required IDs
                needed_files = []
                for audio_file in audio_files:
                    filename = os.path.basename(audio_file)
                    sample_id = os.path.splitext(filename)[0]
                    if sample_id in required_audio_ids:
                        needed_files.append(audio_file)
                print(f"    Need to download {len(needed_files)} of them")
                audio_files = needed_files
            
            for i, audio_file in enumerate(audio_files):
                try:
                    # Download the audio file
                    local_path = hf_hub_download(
                        repo_id=config.hf_repo,
                        filename=audio_file,
                        repo_type="dataset",
                    )
                    
                    # Get the filename as ID
                    filename = os.path.basename(audio_file)
                    sample_id = os.path.splitext(filename)[0]
                    
                    # Load and resample to 16kHz
                    audio_array, sr = librosa.load(local_path, sr=16000, mono=True)
                    
                    # Save to output dir
                    audio_path = os.path.join(split_audio_dir, f"{sample_id}.wav")
                    sf.write(audio_path, audio_array, 16000)
                    id_to_path[sample_id] = audio_path
                    
                    if (i + 1) % 100 == 0:
                        print(f"    Downloaded {i + 1} audio files...")
                        
                except Exception as e:
                    if i < 5:
                        print(f"    Error downloading {audio_file}: {e}")
                    continue
        
        elif parquet_files:
            # Audio embedded in parquet - extract it
            print(f"    Found {len(parquet_files)} parquet files, extracting audio...")
            
            if required_audio_ids:
                print(f"    Looking for {len(required_audio_ids)} specific audio files...")
            
            count = 0
            found_ids = set()
            
            for pq_file in parquet_files:
                # If we have a required set and found all of them, stop
                if required_audio_ids and found_ids >= required_audio_ids:
                    break
                
                try:
                    local_path = hf_hub_download(
                        repo_id=config.hf_repo,
                        filename=pq_file,
                        repo_type="dataset",
                    )
                    
                    # Read parquet
                    table = pq.read_table(local_path)
                    df = table.to_pandas()
                    
                    # Find audio column
                    audio_col = config.audio_column
                    if audio_col not in df.columns:
                        # Try to find it
                        for col in df.columns:
                            if 'audio' in col.lower():
                                audio_col = col
                                break
                    
                    if audio_col not in df.columns:
                        print(f"    No audio column found in {pq_file}")
                        continue
                    
                    # Find ID column  
                    id_col = config.id_column
                    if id_col and id_col not in df.columns:
                        id_col = None
                    
                    for idx, row in df.iterrows():
                        # Get sample ID first to check if we need this sample
                        if id_col:
                            sample_id = row[id_col]
                            if config.id_transform:
                                sample_id = config.id_transform(sample_id)
                        else:
                            sample_id = str(idx)
                        
                        if sample_id is None:
                            sample_id = str(idx)
                        
                        sample_id = str(sample_id)
                        
                        # Skip if we don't need this ID
                        if required_audio_ids and sample_id not in required_audio_ids:
                            continue
                        
                        # Skip if already processed
                        if sample_id in found_ids:
                            continue
                        
                        # Get audio data
                        audio_data = row[audio_col]
                        
                        try:
                            audio_array = None
                            sample_rate = 16000
                            
                            if isinstance(audio_data, dict):
                                if 'bytes' in audio_data and audio_data['bytes']:
                                    # Decode audio bytes
                                    audio_bytes = audio_data['bytes']
                                    with io.BytesIO(audio_bytes) as f:
                                        audio_array, sample_rate = sf.read(f)
                                elif 'path' in audio_data and audio_data['path']:
                                    # Load from path
                                    src_path = audio_data['path']
                                    if os.path.exists(src_path):
                                        audio_array, sample_rate = librosa.load(src_path, sr=16000, mono=True)
                            elif isinstance(audio_data, bytes):
                                with io.BytesIO(audio_data) as f:
                                    audio_array, sample_rate = sf.read(f)
                            
                            if audio_array is None or len(audio_array) == 0:
                                continue
                            
                            # Ensure numpy array
                            audio_array = np.array(audio_array)
                            
                            # Ensure mono
                            if len(audio_array.shape) > 1:
                                audio_array = audio_array.mean(axis=-1)
                            
                            # Resample to 16kHz
                            if sample_rate != 16000:
                                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                            
                            audio_path = os.path.join(split_audio_dir, f"{sample_id}.wav")
                            sf.write(audio_path, audio_array, 16000)
                            id_to_path[sample_id] = audio_path
                            found_ids.add(sample_id)
                            count += 1
                            
                            if count % 50 == 0:
                                print(f"    Extracted {count} audio files...")
                                
                        except Exception as e:
                            if count < 5:
                                print(f"    Error extracting audio for {sample_id}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"    Error processing {pq_file}: {e}")
                    continue
            
            if required_audio_ids:
                missing = required_audio_ids - found_ids
                if missing:
                    print(f"    Warning: Could not find {len(missing)} audio files in source dataset")
        
        else:
            print(f"    No parquet or audio files found in {config.hf_repo}")
            print(f"    Available files: {files[:20]}")
        
        print(f"    Saved {len(id_to_path)} audio files to {split_audio_dir}")
        return id_to_path
        
    except Exception as e:
        print(f"    Error downloading source dataset: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_id_from_sound(sound_filename: str, split: str) -> str:
    """
    Extract the sample ID from AF-Think sound filename.
    The format varies by source dataset.
    """
    # Remove file extension
    base = os.path.splitext(sound_filename)[0]
    return base


def prepare_afthink_split(
    split: str,
    audio_dir: str,
    max_samples: Optional[int] = None,
    download_audio: bool = False,
    existing_audio_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare a single AF-Think split.
    
    Args:
        split: AF-Think split name
        audio_dir: Directory for downloaded audio
        max_samples: Maximum samples to process
        download_audio: Whether to download audio from source
        existing_audio_dir: Directory with pre-downloaded audio files
    
    Returns:
        List of annotation dictionaries
    """
    print(f"\nProcessing split: {split}")
    
    # Step 1: Load AF-Think annotations FIRST to know which audio files we need
    print(f"  Loading AF-Think annotations...")
    afthink_samples = load_afthink_split(split)
    
    if not afthink_samples:
        print(f"  Could not load AF-Think split {split}")
        return []
    
    # Limit samples if specified
    if max_samples:
        afthink_samples = afthink_samples[:max_samples]
    
    # Step 2: Collect required audio file IDs from AF-Think annotations
    required_audio_ids = set()
    for sample in afthink_samples:
        sound_file = sample.get("sound", "")
        if sound_file:
            file_id = extract_id_from_sound(sound_file, split)
            required_audio_ids.add(file_id)
    
    print(f"  Need {len(required_audio_ids)} unique audio files")
    
    # Step 3: Get audio file mapping
    id_to_audio_path = {}
    
    if download_audio:
        # Download ONLY the audio files we need from source dataset
        id_to_audio_path = download_source_audio(split, audio_dir, required_audio_ids=required_audio_ids)
    elif existing_audio_dir:
        # Use existing audio directory - scan for files
        split_audio_dir = os.path.join(existing_audio_dir, split)
        if os.path.exists(split_audio_dir):
            for f in os.listdir(split_audio_dir):
                if f.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    file_id = os.path.splitext(f)[0]
                    id_to_audio_path[file_id] = os.path.join(split_audio_dir, f)
            print(f"  Found {len(id_to_audio_path)} existing audio files")
    
    # Step 4: Create annotations
    annotations = []
    missing_audio = 0
    
    for i, sample in enumerate(afthink_samples):
        sample_id = sample.get("id", f"{split}_{i}")
        sound_file = sample.get("sound", "")
        conversations = sample.get("conversations", [])
        
        # Parse conversations
        qa = parse_conversations(conversations)
        
        # Find audio path
        file_id = extract_id_from_sound(sound_file, split)
        
        if file_id in id_to_audio_path:
            audio_path = id_to_audio_path[file_id]
        else:
            # Fallback: construct expected path
            audio_path = os.path.join(audio_dir, split, sound_file)
            if not os.path.exists(audio_path):
                missing_audio += 1
                if missing_audio <= 3:
                    print(f"    Warning: Audio not found for {file_id}")
        
        annotation = {
            "id": sample_id,
            "path": audio_path,
            "sound_file": sound_file,
            "question": qa["question"],
            "answer": qa["answer"],
            "task": "audio_reasoning",
            "split": split,
        }
        
        annotations.append(annotation)
    
    if missing_audio > 3:
        print(f"    ... and {missing_audio - 3} more missing audio files")
    
    print(f"  Created {len(annotations)} annotations ({missing_audio} missing audio)")
    return annotations


def prepare_afthink(
    output_dir: str,
    audio_dir: str,
    splits: Optional[List[str]] = None,
    max_samples_per_split: Optional[int] = None,
    download_audio: bool = False,
    existing_audio_dir: Optional[str] = None,
):
    """
    Prepare AF-Think dataset for SALMONN training.
    
    Args:
        output_dir: Directory to save annotation JSON files
        audio_dir: Directory for audio files
        splits: List of splits to process (default: auto-downloadable splits)
        max_samples_per_split: Maximum samples per split
        download_audio: Whether to download audio from source datasets
        existing_audio_dir: Directory with pre-downloaded audio
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    if splits is None:
        if download_audio:
            splits = AUTO_DOWNLOAD_SPLITS
            print(f"Auto-download mode: processing {len(splits)} splits with HF audio")
        else:
            splits = AFTHINK_SPLITS
    
    all_annotations = []
    split_stats = {}
    
    for split in splits:
        annotations = prepare_afthink_split(
            split=split,
            audio_dir=audio_dir,
            max_samples=max_samples_per_split,
            download_audio=download_audio,
            existing_audio_dir=existing_audio_dir,
        )
        
        all_annotations.extend(annotations)
        split_stats[split] = len(annotations)
    
    # Save all annotations
    output_file = os.path.join(output_dir, "afthink_all.json")
    with open(output_file, "w") as f:
        json.dump({"annotation": all_annotations}, f, indent=2)
    print(f"\nSaved {len(all_annotations)} total samples to {output_file}")
    
    # Create train/eval splits
    train_annotations = [a for a in all_annotations if a["split"] in TRAIN_SPLITS]
    eval_annotations = [a for a in all_annotations if a["split"] in EVAL_SPLITS]
    
    if train_annotations:
        train_file = os.path.join(output_dir, "train_afthink.json")
        with open(train_file, "w") as f:
            json.dump({"annotation": train_annotations}, f, indent=2)
        print(f"Saved {len(train_annotations)} training samples")
    
    if eval_annotations:
        eval_file = os.path.join(output_dir, "eval_afthink.json")
        with open(eval_file, "w") as f:
            json.dump({"annotation": eval_annotations}, f, indent=2)
        print(f"Saved {len(eval_annotations)} evaluation samples")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for split, count in split_stats.items():
        status = "✓" if SOURCE_DATASETS[split].hf_repo else "✗ (manual)"
        print(f"  {status} {split}: {count} samples")
    print(f"\nTotal: {len(all_annotations)} samples")
    
    return all_annotations


def print_dataset_info():
    """Print information about source datasets and download requirements."""
    print("=" * 70)
    print("AF-Think Source Dataset Information")
    print("=" * 70)
    
    print("\n📦 AUTO-DOWNLOADABLE (HuggingFace):")
    print("-" * 70)
    for name in AUTO_DOWNLOAD_SPLITS:
        cfg = SOURCE_DATASETS[name]
        print(f"  • {name}")
        print(f"    HF: {cfg.hf_repo}")
        print(f"    Notes: {cfg.notes}")
    
    print("\n⚠️  MANUAL DOWNLOAD REQUIRED:")
    print("-" * 70)
    for name in MANUAL_DOWNLOAD_SPLITS:
        cfg = SOURCE_DATASETS[name]
        print(f"  • {name}")
        if cfg.manual_url:
            print(f"    URL: {cfg.manual_url}")
        print(f"    Notes: {cfg.notes}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare AF-Think dataset for SALMONN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset info and download requirements
  python prepare_afthink.py --info
  
  # Download available datasets and create annotations
  python prepare_afthink.py --download-audio
  
  # Process specific splits only
  python prepare_afthink.py --splits urbansound8k esc50 --download-audio
  
  # Use existing audio directory
  python prepare_afthink.py --existing-audio-dir /path/to/audio
  
  # Debug mode with limited samples
  python prepare_afthink.py --splits urbansound8k --max-samples 100 --download-audio
        """,
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save annotation files",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory for audio files",
    )
    parser.add_argument(
        "--existing-audio-dir",
        type=str,
        default=None,
        help="Directory with pre-downloaded audio files",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Specific splits to process",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (for debugging)",
    )
    parser.add_argument(
        "--download-audio",
        action="store_true",
        help="Download audio from source HuggingFace datasets",
    )
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info()
        exit(0)
    
    # Get the SALMONN project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    
    output_dir = args.output_dir or str(data_dir / "afthink")
    audio_dir = args.audio_dir or str(data_dir / "afthink" / "audio")
    
    prepare_afthink(
        output_dir=output_dir,
        audio_dir=audio_dir,
        splits=args.splits,
        max_samples_per_split=args.max_samples,
        download_audio=args.download_audio,
        existing_audio_dir=args.existing_audio_dir,
    )
