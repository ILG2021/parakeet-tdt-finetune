import torch
import pytorch_lightning as pl
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, dictconfig
import argparse

def main(args):
    # 1. Load the pre-trained Parakeet-TDT-0.6b-v3 model
    print("Loading pre-trained model: nvidia/parakeet-tdt-0.6b-v3")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

    # 2. Setup training data
    print(f"Setting up training data from {args.train_manifest}")
    model.setup_training_data(train_data_config={
        'manifest_filepath': args.train_manifest,
        'sample_rate': 16000,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
    })

    # 3. Setup validation data (Optional)
    if args.val_manifest:
        print(f"Setting up validation data from {args.val_manifest}")
        model.setup_validation_data(val_data_config={
            'manifest_filepath': args.val_manifest,
            'sample_rate': 16000,
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 4,
        })
    else:
        print("No validation manifest provided. Skipping validation steps.")

    # 4. (Optional) Setup SpecAugment - keeping original settings is usually fine
    # But we can update it if needed via model.spec_augmentation

    # 5. Setup Optimization
    # Based on official NeMo TDT specs: betas=[0.9, 0.98], weight_decay=1e-3
    optim_config = {
        'name': 'adamw',
        'lr': args.lr,
        'betas': [0.9, 0.98], # Official TDT spec
        'weight_decay': 1e-3,
        'sched': {
            'name': 'CosineAnnealing',
            'warmup_steps': 200, # Small warmup for fine-tuning
            'min_lr': 1e-9,
        }
    }
    model.setup_optimization(optim_config=OmegaConf.create(optim_config))

    # 6. Initialize Trainer
    # Parakeet-0.6b is large; use precision='bf16-mixed' if you have A100/H100/RTX30+
    trainer = pl.Trainer(
        devices=1, 
        accelerator='gpu', 
        max_epochs=args.epochs,
        precision='bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed',
        accumulate_grad_batches=args.grad_acc, # Gradient accumulation to handle large models
        gradient_clip_val=1.0, # Recommended for Transducers
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    # 7. Start Fine-tuning
    print("Starting fine-tuning...")
    trainer.fit(model)

    # 8. Save the final model
    print(f"Saving fine-tuned model to {args.save_path}")
    model.save_to(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Parakeet-TDT v3")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16) # Smaller default for 0.6b model
    parser.add_argument("--grad_acc", type=int, default=1) # Default grad accumulation
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="parakeet_tdt_finetuned.nemo")
    
    args = parser.parse_args()
    main(args)
