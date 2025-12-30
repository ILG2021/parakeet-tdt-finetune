import os
import json
import argparse
import librosa
import pandas as pd
from tqdm import tqdm

def create_manifest_local_audiofolder(data_dir, output_manifest, lang, text_col='sentence', metadata_name='metadata.csv'):
    """
    Generate a NeMo manifest from a local AudioFolder.
    Assumes a CSV file exists in data_dir with columns [file_name, text_col].
    """
    metadata_path = os.path.join(data_dir, metadata_name)
    if not os.path.exists(metadata_path):
        # Try finding any csv/jsonl if metadata.csv doesn't exist
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

    manifest = []
    print(f"Processing {len(df)} entries...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = row.get('file_name') or row.get('file')
        transcript = row.get(text_col) or row.get('text') or row.get('sentence')
        
        if not rel_path or pd.isna(transcript):
            continue
            
        # Construct absolute path
        abs_path = os.path.join(data_dir, rel_path)
        if not os.path.exists(abs_path):
            # Try removing possible leading slashes or 'test/' if it's already in the folder
            # This handles cases where metadata has 'test/audio.mp3' but we are already in the 'test' folder
            abs_path = os.path.join(os.path.dirname(data_dir), rel_path)
            if not os.path.exists(abs_path):
                continue
        
        # Get duration
        try:
            duration = librosa.get_duration(path=abs_path)
        except Exception as e:
            print(f"Error getting duration for {abs_path}: {e}")
            continue
            
        manifest.append({
            "audio_filepath": os.path.abspath(abs_path),
            "duration": float(duration),
            "text": str(transcript),
            "lang": lang
        })
        
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\nManifest created at {output_manifest} with {len(manifest)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NeMo manifest from local AudioFolder")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Root directory of the dataset (where metadata.csv is located)")
    parser.add_argument("--output_manifest", type=str, required=True, 
                        help="Path to save the NeMo manifest.json")
    parser.add_argument("--text_col", type=str, default="sentence", 
                        help="Column name for transcription in metadata")
    parser.add_argument("--metadata_name", type=str, default="metadata.csv",
                        help="Name of the metadata file")
    parser.add_argument("--lang", type=str, default="zh", help="Language code")
    
    args = parser.parse_args()

    create_manifest_local_audiofolder(
        data_dir=args.data_dir,
        output_manifest=args.output_manifest,
        lang=args.lang,
        text_col=args.text_col,
        metadata_name=args.metadata_name
    )
