import yaml
import argparse
from typing import Dict, Any

class ConfigParser:
    """
    Handles loading of YAML configurations and overriding them via command line arguments.
    """

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """Loads a YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parses command line arguments."""
        parser = argparse.ArgumentParser(description="PRM Training/Inference")
        parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")

        # Allow dynamic overrides (e.g. --training.learning_rate 1e-5)
        # This consumes all remaining args to allow flexible overriding
        return parser.parse_known_args()

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
        """
        Merges command line overrides into the dictionary.
        Args format expected: ['--training.learning_rate', '1e-5']
        """
        it = iter(overrides)
        for key_arg in it:
            if not key_arg.startswith("--"):
                continue

            key_path = key_arg.lstrip("-").split(".")
            try:
                value = next(it)
                # Attempt to convert to number/bool
                if value.lower() == 'true': value = True
                elif value.lower() == 'false': value = False
                else:
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass # Keep as string

                # Update dict
                curr = base_config
                for k in key_path[:-1]:
                    curr = curr.setdefault(k, {})
                curr[key_path[-1]] = value

            except StopIteration:
                break

        return base_config

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Main entry point to get the final configuration."""
        args, overrides = cls.parse_args()
        config = cls.load_yaml(args.config)
        config = cls.merge_configs(config, overrides)
        return config
