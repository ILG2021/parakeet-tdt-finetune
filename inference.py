import nemo.collections.asr as nemo_asr
import argparse
import os

def run_inference(model_path, audio_files):
    # Determine if it's a .nemo file or a pre-trained name
    if model_path.endswith(".nemo"):
        print(f"Restoring model from {model_path}...")
        model = nemo_asr.models.ASRModel.restore_from(model_path)
    else:
        print(f"Loading pre-trained model: {model_path}...")
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path)

    # Move to GPU if available
    import torch
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    print(f"Transcribing {len(audio_files)} files...")
    # transcribe() supports both list of file paths and single file
    results = model.transcribe(audio_files, batch_size=4)
    
    # Parakeet-TDT transcribes usually return a list of objects or strings depending on version
    # In recent NeMo, it returns a list of Transcription objects if requested, or strings by default
    
    for i, file in enumerate(audio_files):
        # Handle different return types
        text = results[i]
        if hasattr(text, 'text'):
            text = text.text
            
        print(f"\nFile: {file}")
        print(f"Transcription: {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Parakeet-TDT v3")
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-0.6b-v3", help=".nemo file or pre-trained name")
    parser.add_argument("--audio", type=str, nargs='+', required=True, help="Path(s) to audio file(s)")
    
    args = parser.parse_args()
    run_inference(args.model, args.audio)
