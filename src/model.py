from typing import Tuple, Dict
import os

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None

class PRMModelLoader:
    """
    Wrapper for loading Qwen models using Unsloth for optimized training.
    """

    @staticmethod
    def load(config: Dict) -> Tuple[object, object]:
        """
        Loads the 4-bit quantized model and tokenizer.

        Args:
            config: Configuration dictionary.

        Returns:
            model, tokenizer
        """
        if not UNSLOTH_AVAILABLE:
            raise ImportError(
                "Unsloth is not installed. Please install it with:\n"
                "  pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'\n"
                "Note: Unsloth requires CUDA. Training is not supported on CPU/MPS."
            )

        max_seq_length = config['model']['max_seq_length']
        load_in_4bit = config['model']['load_in_4bit']

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config['model']['base_model'],
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,  # Auto-detect
        )

        # Configure tokenizer for training
        # Set padding token if not already set (required for batch training)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Required for causal LM training

        # Configure LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config['lora']['r'],
            target_modules=config['lora']['target_modules'],
            lora_alpha=config['lora']['lora_alpha'],
            lora_dropout=config['lora']['lora_dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        return model, tokenizer

    @staticmethod
    def save_merged(model, tokenizer, output_dir: str):
        """
        Merges LoRA adapters and saves in 16-bit for vLLM compatibility.
        """
        print(f"Merging model and saving to {output_dir}...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model.save_pretrained_merged(
            output_dir, 
            tokenizer, 
            save_method="merged_16bit"
        )