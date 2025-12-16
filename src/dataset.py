import random
import re
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class PRMDatasetBuilder:
    """
    Handles data loading, flattening, and formatting for Process Reward Models.
    Specifically designed for Math-Shepherd style datasets (step-wise).

    Math-Shepherd format:
    - 'input': Problem statement + solution steps (without labels)
    - 'label': Problem + steps with +/- labels at end of each line
    - 'task': Source dataset (e.g., "GSM8K")

    Example label format:
    "Problem text Step 1: Calculate 40*3=120 +\nStep 2: Calculate 28*5=140 -\n..."

    Each step line ends with a space and +/- indicating correct/incorrect.
    """

    def __init__(self, config: Dict, tokenizer: PreTrainedTokenizer):
        """
        Args:
            config: Data configuration dictionary.
            tokenizer: Tokenizer to use for formatting.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.verify_token = config['training']['response_template']  # e.g. "<|verify|>"

    def _extract_problem_from_input(self, input_text: str) -> str:
        """
        Extracts just the problem statement from the input field.
        The input contains problem + solution, we want just the problem.
        """
        # Find where the solution starts (usually "Step 1:" or similar)
        step_match = re.search(r'Step\s*1\s*:', input_text)
        if step_match:
            return input_text[:step_match.start()].strip()

        # Fallback: look for "ки" marker
        ki_match = input_text.find('ки')
        if ki_match != -1:
            # Find the last sentence before the first step
            text_before = input_text[:ki_match]
            # Try to find where problem ends
            for marker in ['?', '.']:
                last_marker = text_before.rfind(marker)
                if last_marker != -1:
                    # Check if there's step content after
                    potential_problem = text_before[:last_marker + 1].strip()
                    if len(potential_problem) > 20:  # Reasonable problem length
                        return potential_problem

        # Ultimate fallback: return first portion
        return input_text[:500] if len(input_text) > 500 else input_text

    def _parse_math_shepherd_entry(self, entry: Dict) -> List[Tuple[str, str, int]]:
        """
        Parses a Math-Shepherd entry into (context, step, label) tuples.

        Math-Shepherd 'label' field format:
        "Problem text Step 1: content +\nStep 2: content -\n..."

        Each step ends with a space and +/- label, separated by newlines.

        Returns:
            List of (cumulative_context, step_text, label) tuples
        """
        label_text = entry.get('label', '')

        if not label_text:
            return []

        # Find where Step 1 starts to separate problem from steps
        step1_match = re.search(r'Step\s*1\s*[:\.]', label_text)
        if not step1_match:
            return []

        problem = label_text[:step1_match.start()].strip()
        steps_text = label_text[step1_match.start():]

        if not problem:
            return []

        # Parse steps using regex to find "Step N: content +/-"
        # The pattern captures: step header, content, and label
        step_pattern = r'(Step\s*\d+\s*[:\.])\s*(.+?)\s+([+-])(?=\s*Step\s*\d+|\s*$)'
        matches = re.findall(step_pattern, steps_text, re.DOTALL)

        if not matches:
            # Fallback: split by newlines and look for +/- at end
            lines = steps_text.strip().split('\n')
            matches = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check for " +" or " -" at the end
                match = re.match(r'(.+?)\s+([+-])\s*$', line)
                if match:
                    # Store as (header, content, label) format for consistency
                    matches.append(('', match.group(1).strip(), match.group(2)))

        if not matches:
            return []

        results = []
        current_context = f"Problem: {problem}\n\nSolution:"

        for header, content, label_char in matches:
            step_text = f"{header} {content}".strip() if header else content.strip()

            if not step_text:
                continue

            # Convert label char to int
            label = 1 if label_char == '+' else 0

            results.append((current_context, step_text, label))
            current_context = f"{current_context}\n{step_text}"

        return results

    def load_and_prepare(self) -> Dataset:
        """
        Loads dataset, balances positives/negatives, and formats for SFT.
        """
        print(f"Loading dataset: {self.config['data']['dataset_name']}")

        # Load dataset
        try:
            ds = load_dataset(self.config['data']['dataset_name'], split="train")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Trying alternative loading method...")
            ds = load_dataset(self.config['data']['dataset_name'], split="train", trust_remote_code=True)

        print(f"Dataset loaded with {len(ds)} entries")

        max_samples = self.config['data'].get('max_samples')
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
            print(f"Limited to {max_samples} samples")

        processed_data = []

        # Counters for balancing
        pos_count = 0
        neg_count = 0
        skipped_for_balance = 0
        parse_errors = 0

        # Iterate over the dataset with progress bar
        print("Processing dataset entries...")
        for entry in tqdm(ds, desc="Parsing"):
            try:
                parsed_steps = self._parse_math_shepherd_entry(entry)
            except Exception as e:
                parse_errors += 1
                continue

            for context, step, label in parsed_steps:
                # Determine target token
                target = "+" if label == 1 else "-"

                # Balancing Logic - undersample positives to ~50/50
                if self.config['data'].get('balance_positives', True):
                    if target == "+":
                        # Skip if we have too many positives relative to negatives
                        if pos_count > neg_count + 100:
                            skipped_for_balance += 1
                            continue
                        pos_count += 1
                    else:
                        neg_count += 1
                else:
                    if target == "+":
                        pos_count += 1
                    else:
                        neg_count += 1

                # Format: Context + Step + VerifyToken + Target
                # The DataCollatorForCompletionOnlyLM will handle masking
                # so loss is only calculated on tokens AFTER <|verify|>
                full_text = f"{context}\n{step}\n{self.verify_token} {target}"

                # Check sequence length
                tokenized = self.tokenizer.encode(full_text)
                max_len = self.config['model'].get('max_seq_length', 2048)
                if len(tokenized) > max_len:
                    continue  # Skip overly long sequences

                processed_data.append({
                    "text": full_text,
                    "label": label
                })

        print(f"\nProcessing complete!")
        print(f"  Total samples: {len(processed_data)}")
        print(f"  Positive steps: {pos_count}")
        print(f"  Negative steps: {neg_count}")
        if skipped_for_balance > 0:
            print(f"  Skipped for balance: {skipped_for_balance}")
        if parse_errors > 0:
            print(f"  Parse errors: {parse_errors}")

        if len(processed_data) == 0:
            raise ValueError("No valid samples found! Check dataset format and configuration.")

        # Shuffle the dataset
        random.shuffle(processed_data)

        return Dataset.from_list(processed_data)


def test_dataset_parser():
    """
    Test function to verify the dataset parser works correctly.
    Run with: python -c "from src.dataset import test_dataset_parser; test_dataset_parser()"
    """
    print("Testing Math-Shepherd dataset parser...")

    # Load a few samples
    ds = load_dataset("peiyi9979/Math-Shepherd", split="train")

    print(f"\nDataset columns: {ds.column_names}")
    print(f"Dataset size: {len(ds)}")

    # Show first entry
    print("\n" + "=" * 60)
    print("SAMPLE ENTRY")
    print("=" * 60)

    entry = ds[0]
    print(f"\n--- LABEL field (last 300 chars) ---")
    print(repr(entry['label'][-300:]))
    print(f"\n--- TASK ---")
    print(entry.get('task', 'N/A'))

    # Test parsing
    print("\n" + "=" * 60)
    print("PARSED STEPS")
    print("=" * 60)

    # Create a minimal config for testing
    class MockTokenizer:
        def encode(self, text):
            return list(range(len(text) // 4))  # Rough approximation

    config = {
        'training': {'response_template': '<|verify|>'},
        'data': {'balance_positives': False},
        'model': {'max_seq_length': 2048}
    }

    builder = PRMDatasetBuilder(config, MockTokenizer())
    parsed = builder._parse_math_shepherd_entry(entry)

    print(f"\nFound {len(parsed)} steps:")
    for i, (context, step, label) in enumerate(parsed[:3]):
        print(f"\n--- Step {i+1} ---")
        print(f"Label: {'+' if label == 1 else '-'}")
        print(f"Step: {step[:100]}...")
        if i == 0:
            print(f"Context preview: {context[:150]}...")

    # Test a few more entries to ensure robustness
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE ENTRIES")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    for i in range(min(100, len(ds))):
        parsed = builder._parse_math_shepherd_entry(ds[i])
        if parsed:
            success_count += 1
        else:
            fail_count += 1

    print(f"\nParsed {success_count}/100 entries successfully")
    print(f"Failed: {fail_count}/100")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_dataset_parser()
