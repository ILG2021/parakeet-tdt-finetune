import os
import json
import argparse
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def create_manifest_local_audiofolder(data_dir, output_manifest, lang, text_col='sentence', metadata_name='metadata.csv', convert_to_wav=False):
    """
    Generate a NeMo manifest from a local AudioFolder.
    Assumes a CSV file exists in data_dir with columns [file_name, text_col].
    If convert_to_wav is True, it will resample and save audio as 16kHz mono WAV.
    """
    metadata_path = os.path.join(data_dir, metadata_name)
    if not os.path.exists(metadata_path):
        print(f"Metadata {metadata_name} not found in {data_dir}, searching for any CSV/JSONL...")
        for f in os.listdir(data_dir):
            if f.endswith('.csv') or f.endswith('.jsonl'):
                metadata_path = os.path.join(data_dir, f)
                break
    
    if not os.path.exists(metadata_path):
        print(f"Error: Could not find metadata file in {data_dir}")
        return

    print(f"Reading metadata from {metadata_path}...")
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
    else:
        df = pd.read_json(metadata_path, lines=True)

    # Setup conversion directory if needed
    wav_dir = os.path.join(data_dir, "standardized_wavs")
    if convert_to_wav:
        os.makedirs(wav_dir, exist_ok=True)
        print(f"Audio conversion enabled. Standardized WAVs will be saved to {wav_dir}")

    manifest = []
    print(f"Processing {len(df)} entries...")
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = row.get('file_name') or row.get('file')
        transcript = row.get(text_col) or row.get('text') or row.get('sentence')
        
        if not rel_path or pd.isna(transcript):
            continue
            
        abs_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(abs_path):
            abs_path = os.path.join(os.path.dirname(data_dir), rel_path)
            if not os.path.exists(abs_path):
                continue
        
        final_audio_path = abs_path
        duration = 0
        
        try:
            # Check if conversion is actually needed
            is_wav = abs_path.lower().endswith('.wav')
            should_convert = convert_to_wav
            
            if is_wav and convert_to_wav:
                # Check WAV properties to see if we can skip conversion
                info = sf.info(abs_path)
                if info.channels == 1 and info.samplerate == 16000:
                    should_convert = False # Already 16k mono WAV, skip
            
            if should_convert:
                # Load and resample to 16kHz mono
                y, sr = librosa.load(abs_path, sr=16000, mono=True)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Create a unique filename for the WAV
                file_id = os.path.splitext(os.path.basename(rel_path))[0]
                wav_path = os.path.join(wav_dir, f"{file_id}_{i}.wav")
                
                # Save as WAV
                if not os.path.exists(wav_path):
                    sf.write(wav_path, y, sr)
                
                final_audio_path = wav_path
            else:
                # Just get duration from original or already-valid wav
                duration = librosa.get_duration(path=abs_path)
        except Exception as e:
            print(f"Error processing {abs_path}: {e}")
            continue
            
        manifest.append({
            "audio_filepath": os.path.abspath(final_audio_path),
            "duration": float(duration),
            "text": str(transcript),
            "lang": lang
        })
        
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\nManifest created at {output_manifest} with {len(manifest)} entries.")
    if convert_to_wav:
        print(f"Converted audio files are in: {os.path.abspath(wav_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NeMo manifest from local AudioFolder")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory of the dataset")
    parser.add_argument("--output_manifest", type=str, required=True, 
                        help="Path to save the NeMo manifest.json")
    parser.add_argument("--text_col", type=str, default="sentence", 
                        help="Column name for transcription in metadata")
    parser.add_argument("--metadata_name", type=str, default="metadata.csv",
                        help="Name of the metadata file")
    parser.add_argument("--lang", type=str, default="zh", help="Language code")
    parser.add_argument("--convert_to_wav", action="store_true", 
                        help="Convert all audio to 16kHz mono WAV (Highly recommended for MP3/Stereo sources)")
    
    args = parser.parse_args()

    create_manifest_local_audiofolder(
        data_dir=args.data_dir,
        output_manifest=args.output_manifest,
        lang=args.lang,
        text_col=args.text_col,
        metadata_name=args.metadata_name,
        convert_to_wav=args.convert_to_wav
    )
