import sys
import os
import json
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix Unsloth + TRL 0.24+ compatibility issue
try:
    import unsloth.trainer
    if not hasattr(unsloth.trainer, 'PADDING_FREE_BLOCKLIST'):
        unsloth.trainer.PADDING_FREE_BLOCKLIST = []
except:
    pass

from trl import SFTTrainer, SFTConfig

# Handle different TRL versions for DataCollatorForCompletionOnlyLM
try:
    from trl import DataCollatorForCompletionOnlyLM
    USE_CUSTOM_COLLATOR = False
except ImportError:
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
        USE_CUSTOM_COLLATOR = False
    except ImportError:
        USE_CUSTOM_COLLATOR = True


# Custom data collator for newer TRL versions that removed DataCollatorForCompletionOnlyLM
class CustomDataCollatorForCompletionOnlyLM:
    """
    Custom collator that masks labels before the response template.
    Only computes loss on tokens after the response template.
    """
    def __init__(self, response_template, tokenizer):
        self.response_template = response_template  # List of token IDs
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # Tokenize if needed
        if isinstance(examples[0], dict) and "text" in examples[0]:
            texts = [ex["text"] for ex in examples]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            )
        else:
            batch = self.tokenizer.pad(examples, return_tensors="pt")

        # Create labels (copy of input_ids)
        labels = batch["input_ids"].clone()

        # Mask everything before response template
        for i, input_ids in enumerate(batch["input_ids"]):
            input_list = input_ids.tolist()
            response_start = None

            # Find where response template starts
            template_len = len(self.response_template)
            for j in range(len(input_list) - template_len + 1):
                if input_list[j:j + template_len] == self.response_template:
                    response_start = j + template_len
                    break

            if response_start is not None:
                # Mask everything before and including the template
                labels[i, :response_start] = -100
            else:
                # If template not found, mask everything (no loss)
                labels[i, :] = -100

        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

from src.config_parser import ConfigParser
from src.utils import seed_everything, setup_logging
from src.dataset import PRMDatasetBuilder
from src.model import PRMModelLoader


def main():
    # 1. Setup
    config = ConfigParser.get_config()
    logger = setup_logging(config['project']['logging_dir'])
    seed_everything(config['project']['seed'])

    logger.info("Starting training pipeline...")
    logger.info(f"Configuration: {config}")

    # Ensure output directories exist
    os.makedirs(config['project']['output_dir'], exist_ok=True)
    os.makedirs(config['project']['logging_dir'], exist_ok=True)

    # 2. Model & Tokenizer
    logger.info(f"Loading model: {config['model']['base_model']}")
    model, tokenizer = PRMModelLoader.load(config)

    # 3. Data Preparation
    logger.info("Preparing dataset...")
    dataset_builder = PRMDatasetBuilder(config, tokenizer)
    train_dataset = dataset_builder.load_and_prepare()

    if len(train_dataset) == 0:
        logger.error("No training samples found! Check dataset configuration.")
        return

    logger.info(f"Training dataset size: {len(train_dataset)}")

    # 4. Collator
    # This effectively makes it a classification task by ignoring the loss
    # on everything except the token following <|verify|>
    response_template = config['training']['response_template']

    # Tokenize the response template to get the token IDs
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    logger.info(f"Response template '{response_template}' -> token IDs: {response_template_ids}")

    # Use custom collator for newer TRL versions, or built-in for older versions
    if USE_CUSTOM_COLLATOR:
        logger.info("Using custom DataCollatorForCompletionOnlyLM (TRL 0.24+ compatibility)")
        collator = CustomDataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer
        )
    else:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer
        )

    # 5. Trainer Config
    sft_args = SFTConfig(
        output_dir=config['project']['output_dir'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        logging_steps=config['training']['logging_steps'],
        num_train_epochs=config['training']['num_train_epochs'],
        max_seq_length=config['model']['max_seq_length'],
        dataset_text_field="text",
        packing=False,
        # Add save configuration
        save_strategy=config['training'].get('save_strategy', 'steps'),
        save_steps=config['training'].get('save_steps', 100),
        logging_dir=config['project']['logging_dir'],
        # Optimization settings
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        # Disable default evaluation (we don't have eval set by default)
        do_eval=False,
        # Report to console
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=sft_args,
    )

    # 6. Train
    logger.info("Starting training loop...")
    trainer.train()

    # 7. Merge & Save (Critical for vLLM Inference)
    final_path = os.path.join(config['project']['output_dir'], "merged_model")
    logger.info(f"Merging and saving model to: {final_path}")
    PRMModelLoader.save_merged(model, tokenizer, final_path)

    # Save config for traceability
    config_save_path = os.path.join(final_path, "training_config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)

    logger.info(f"Training complete. Artifacts saved to {final_path}")


if __name__ == "__main__":
    main()