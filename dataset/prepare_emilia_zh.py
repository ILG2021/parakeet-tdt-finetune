import os
import json
import argparse
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset

def create_manifest_emilia(dataset_name, output_dir, manifest_path, lang, cache_dir):
    """
    Generate a NeMo manifest from a HuggingFace dataset in Emilia-YODAS format.
    Saves audio files locally to output_dir.
    """
    print(f"Loading Emilia dataset {dataset_name} from HuggingFace (cache_dir={cache_dir})...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)
    
    split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    data = dataset[split]
    
    manifest = []
    print(f"Processing {len(data)} entries...")
    
    for i, row in enumerate(tqdm(data)):
        # Emilia-YODAS specific format
        info = row.get('json', {})
        audio_data = row.get('mp3') 
        
        if not info or not audio_data:
            continue
            
        file_id = info.get('_id', f"sample_{i}")
        transcript = info.get('text', "")
        duration = info.get('duration', 0)
        
        # Save audio locally
        audio_filename = f"{file_id}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        if not os.path.exists(audio_path):
            try:
                sf.write(audio_path, audio_data['array'], audio_data['sampling_rate'])
            except Exception as e:
                print(f"Error saving audio for {file_id}: {e}")
                continue
        
        manifest.append({
            "audio_filepath": os.path.abspath(audio_path),
            "duration": float(duration),
            "text": str(transcript),
            "lang": lang
        })
        
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\nManifest created at {manifest_path} with {len(manifest)} entries.")
    print(f"Audio files saved in {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NeMo manifest from Emilia-YODAS HuggingFace dataset")
    parser.add_argument("--hf_dataset", type=str, required=True, default="Vyvo-Research/Emilia-YODAS-ZH",
                        help="HuggingFace dataset name (e.g., Vyvo-Research/Emilia-YODAS-ZH)")
    parser.add_argument("--output_manifest", type=str, required=True, 
                        help="Path to save the NeMo manifest.json")
    parser.add_argument("--output_dir", type=str, default="data/emilia_wavs", 
                        help="Local directory to store extracted audio files")
    parser.add_argument("--hf_cache_dir", type=str, default="./hf_cache", 
                        help="Directory for HuggingFace dataset cache")
    parser.add_argument("--lang", type=str, default="zh", help="Language code")
    
    args = parser.parse_args()

    create_manifest_emilia(
        dataset_name=args.hf_dataset,
        output_dir=args.output_dir,
        manifest_path=args.output_manifest,
        lang=args.lang,
        cache_dir=args.hf_cache_dir
    )
