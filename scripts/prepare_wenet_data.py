import os
import json
import argparse
import soundfile as sf
import torch
from tqdm import tqdm
from datasets import load_dataset

def create_manifest_wenet(dataset_name, output_dir, manifest_path, lang, cache_dir):
    """
    Generate a NeMo manifest from a HuggingFace dataset in WenetSpeech format.
    Saves audio files locally to output_dir.
    """
    print(f"Loading WenetSpeech dataset {dataset_name} from HuggingFace (cache_dir={cache_dir})...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(dataset_name,  split='train[:10%]', cache_dir=cache_dir, trust_remote_code=True)
    
    # Identify the split
    # split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    data = dataset

    # Initialize FunASR punctuation model if requested
    punctuator = None
    try:
        from funasr import AutoModel
        print("Initializing FunASR punctuation model (ct-punc)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # FunASR will download and cache the model automatically
        punctuator = AutoModel(model="ct-punc", device=device)
    except ImportError:
        print("Error: 'funasr' not installed. Please run 'pip install funasr' or disable punctuation.")

    manifest = []
    print(f"Processing {len(data)} entries...")
    
    for i, row in enumerate(tqdm(data)):
        audio_data = row.get('audio')
        transcript = row.get('text', "")
        
        if not audio_data:
            continue
            
        # FunASR Punctuation restoration
        if punctuator and transcript:
            try:
                # FunASR generate returns a list of results with processed text
                res = punctuator.generate(input=transcript)
                if res and len(res) > 0:
                    transcript = res[0].get('text', transcript)
            except Exception as e:
                print(f"Punctuation error for item {i}: {e}")
        
        orig_path = row.get('audio_path', f"sample_{i}")
        file_id = os.path.splitext(os.path.basename(orig_path))[0]
        
        audio_filename = f"{file_id}.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        
        if not os.path.exists(audio_path):
            try:
                sf.write(audio_path, audio_data['array'], audio_data['sampling_rate'])
            except Exception as e:
                print(f"Error saving audio for {file_id}: {e}")
                continue
        
        duration = len(audio_data['array']) / audio_data['sampling_rate']
        
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
    parser = argparse.ArgumentParser(description="Prepare NeMo manifest from WenetSpeech HuggingFace dataset")
    parser.add_argument("--hf_dataset", type=str,
                        default="zxsddcs/wenetspeech_S",
                        help="HuggingFace dataset name (e.g., wenet-e2e/WenetSpeech)")
    parser.add_argument("--output_manifest", type=str, default="wenet.json",
                        help="Path to save the NeMo manifest.json")
    parser.add_argument("--output_dir", type=str, default="data/wenet_wavs", 
                        help="Local directory to store extracted audio files")
    parser.add_argument("--hf_cache_dir", type=str, default="./hf_cache", 
                        help="Directory for HuggingFace dataset cache")
    parser.add_argument("--lang", type=str, default="zh", help="Language code")
    
    args = parser.parse_args()

    create_manifest_wenet(
        dataset_name=args.hf_dataset,
        output_dir=args.output_dir,
        manifest_path=args.output_manifest,
        lang=args.lang,
        cache_dir=args.hf_cache_dir
    )
