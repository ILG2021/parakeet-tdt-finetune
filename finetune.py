import torch
import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, DictConfig
import argparse
import os
import json

def extract_char_list(manifest_paths):
    """Extract all unique characters from the manifest files to build a Chinese vocabulary."""
    chars = set()
    paths = manifest_paths.split(',')
    print(f"Extracting characters from {len(paths)} manifest(s)...")
    for path in paths:
        with open(path.strip(), 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                chars.update(list(item['text']))
    # Filter out common non-Chinese characters if needed, or keep all
    # Sorted for consistency
    char_list = sorted(list(chars))
    print(f"Total unique characters found: {len(char_list)}")
    return char_list

def train_char_tokenizer(manifest_paths, output_dir):
    """Trains a character-level SentencePiece tokenizer on the fly."""
    import sentencepiece as spm
    print(f"Training character-level tokenizer in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, 'corpus.txt')
    
    # Collect all text for training
    chars = set()
    with open(text_path, 'w', encoding='utf-8') as f_out:
        for m in manifest_paths.split(','):
            with open(m.strip(), 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    text = json.loads(line)['text']
                    f_out.write(text + '\n')
                    chars.update(list(text))
    
    vocab_size = len(chars) + 32  # Add buffer for control tokens (<unk>, <s>, </s>, etc.)
    
    spm.SentencePieceTrainer.train(
        input=text_path,
        model_prefix=os.path.join(output_dir, 'spm'), # Temporary prefix
        vocab_size=vocab_size,
        model_type='char',
        character_coverage=1.0,
    )
    
    # NeMo expects specifically named files in the directory: 'tokenizer.model' and 'vocab.txt'
    os.rename(os.path.join(output_dir, 'spm.model'), os.path.join(output_dir, 'tokenizer.model'))
    os.rename(os.path.join(output_dir, 'spm.vocab'), os.path.join(output_dir, 'vocab.txt'))
    
    print(f"Tokenizer training complete. Files saved in {output_dir}")

def main(args):
    # 1. Load the pre-trained Parakeet-TDT-0.6b-v3 model
    print("Loading pre-trained model: nvidia/parakeet-tdt-0.6b-v3")
    # TDT models are sensitive to config, so we load carefully
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

    # [OPTIMIZATION] Enable Gradient Checkpointing
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'set_gradient_checkpointing'):
        print("Enabling Gradient Checkpointing for Encoder...")
        model.encoder.set_gradient_checkpointing(True)
    model.set_gradient_checkpointing(True)

    # 2. Chinese Vocabulary Replacement (MANDATORY for English base model)
    print("Preparing for Chinese vocabulary replacement...")
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        print(f"Using provided tokenizer from {args.tokenizer_path}")
        model.change_vocabulary(new_tokenizer_dir=args.tokenizer_path, new_tokenizer_type="bpe")
    else:
        print("No tokenizer provided. Generating character-level SP tokenizer from manifests...")
        # For EncDecRNNTBPEModel, we must provide a tokenizer directory. 
        # We train a quick character-level SentencePiece model to act as a char-tokenizer.
        temp_tokenizer_dir = "chinese_tokenizer_chars"
        train_char_tokenizer(args.train_manifest, temp_tokenizer_dir)
        model.change_vocabulary(new_tokenizer_dir=temp_tokenizer_dir, new_tokenizer_type="bpe")
    
    # 3. Setup training data
    # NeMo supports multiple manifests. We use a more robust setup for TDT.
    print(f"Setting up training data (multiple manifests: {args.train_manifest})")
    
    # TDT requires specific data config to handle its duration-based targets
    model.setup_training_data(train_data_config={
        'manifest_filepath': args.train_manifest,
        'sample_rate': 16000,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'trim_silence': False,
        'max_duration': 30.0,  # [USER REQUEST] No duration limit
        'min_duration': 1.0,
        'use_start_end_at_placeholder': True,
        # [OPTIMIZATION] Bucketing helps managing memory with variant lengths
        'bucketing_strategy': 'synced_randomized',
        'bucketing_batch_size': args.batch_size,
    })

    # 4. Setup validation data
    if args.val_manifest:
        print(f"Setting up validation data from {args.val_manifest}")
        model.setup_validation_data(val_data_config={
            'manifest_filepath': args.val_manifest,
            'sample_rate': 16000,
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': args.num_workers // 2 if args.num_workers > 1 else 1,
        })

    # 5. Setup Optimization
    # For fine-tuning, we use a smaller learning rate
    optim_config = {
        'name': 'adamw',
        'lr': args.lr,
        'betas': [0.9, 0.98],
        'weight_decay': 1e-3,
        'sched': {
            'name': 'CosineAnnealing',
            'warmup_steps': 500,
            'min_lr': 1e-8,
        }
    }
    model.setup_optimization(optim_config=OmegaConf.create(optim_config))

    # 6. Initialize Trainer
    # bf16-mixed is highly recommended for Parakeet (Ampere GPUs and later)
    precision = 'bf16-mixed' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else '16-mixed'
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='parakeet-tdt-zh-{epoch:02d}-{val_wer:.2f}',
        save_top_k=3,
        monitor='val_wer' if args.val_manifest else 'train_loss',
        mode='min',
        save_last=True
    )

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator='gpu', 
        max_epochs=args.epochs,
        precision=precision,
        accumulate_grad_batches=args.grad_acc,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true" if args.gpus > 1 else "auto"
    )

    # 7. Start Fine-tuning
    print("Starting fine-tuning...")
    trainer.fit(model)

    # 8. Save the final model
    print(f"Saving fine-tuned model to {args.save_path}")
    model.save_to(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Parakeet-TDT v3 for Chinese ASR")
    parser.add_argument("--train_manifest", type=str, required=True, help="Manifest paths, comma separated")
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to a pre-trained SP model (optional)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--grad_acc", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader num_workers")
    parser.add_argument("--lr", type=float, default=7.5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="parakeet_tdt_zh_5090.nemo")
    
    args = parser.parse_args()
    main(args)
