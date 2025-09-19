import argparse
import yaml
import torch
import numpy as np
import random
import os
from pathlib import Path

from . import preprocess
from . import train
from . import evaluate

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Run SOFT-FAR experiments.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true", help="Run a small-scale smoke test.")
    mode.add_argument("--full-experiment", action="store_true", help="Run the full-scale experiment.")
    args = parser.parse_args()

    if args.smoke_test:
        config_path = Path('config/smoke_test.yaml')
    else:
        config_path = Path('config/full_experiment.yaml')

    if not config_path.exists():
        print(f"Configuration file not found at {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'training_args' in config:
            for key in ['learning_rate', 'max_grad_norm']:
                if key in config['training_args']:
                    config['training_args'][key] = float(config['training_args'][key])
        
        if 'soft_far_params' in config:
            for key in ['lambda_fact', 'delta_w2', 'sinkhorn_epsilon']:
                if key in config['soft_far_params']:
                    config['soft_far_params'][key] = float(config['soft_far_params'][key])
                    
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        return

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Initial model and tokenizer loading to be used across seeds
    print(f"Loading base model and tokenizer for {config['model_name']}...")
    base_model, tokenizer = train.get_model_and_tokenizer(
        config['model_name'], 
        fp16=config['training_args'].get('fp16', True)
    )

    print("Loading and preparing data...")
    train_dataloader, eval_dataloader = preprocess.load_and_prepare_data(config, tokenizer)

    for seed in config['seeds']:
        print(f"\n{'='*20} RUNNING EXPERIMENT FOR SEED: {seed} {'='*20}")
        set_seed(seed)
        config['current_seed'] = seed
        
        # We can either reload the model to reset weights or continue fine-tuning from the base.
        # For independent runs, reloading is better.
        print("Reloading model for new seed...")
        model, _ = train.get_model_and_tokenizer(
            config['model_name'],
            fp16=config['training_args'].get('fp16', True)
        )

        print("Starting training...")
        trained_model, train_losses = train.train_model(
            config,
            model,
            tokenizer,
            train_dataloader,
            eval_dataloader
        )

        print("Starting evaluation...")
        evaluate.evaluate_model(
            trained_model,
            tokenizer,
            eval_dataloader,
            config,
            train_losses
        )

        # Clean up GPU memory
        del model
        del trained_model
        torch.cuda.empty_cache()

    print("\nAll experiments completed.")

if __name__ == '__main__':
    main()
