#!/usr/bin/env python3
"""
Network Configuration Parser AI Training System

This system trains a model to generate regex-based parsers for network configuration lines.
It learns from existing parser patterns and can suggest new parsers for unseen config formats.
"""

import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import os

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigParserDataset(Dataset):
    """Dataset for network configuration parser training."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Create input text combining config lines and argspec
        config_lines = "\n".join(example["config_lines"])
        argspec_str = json.dumps(example["argspec"], indent=2)
        input_text = (
            f"CONFIG:\n{config_lines}\n\nARGSPEC:\n{argspec_str}\n\nGENERATE_PARSER:"
        )

        # Target is the parser structure
        target_parser = json.dumps(example["parser"], indent=2)

        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize targets
        targets = self.tokenizer(
            target_parser,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }


class ConfigParserModel(PreTrainedModel):
    """Custom model for generating network configuration parsers."""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=6,
        )
        self.output_projection = nn.Linear(768, self.encoder.config.vocab_size)

    def get_input_embeddings(self):
        """Return the input embeddings from the encoder."""
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set the input embeddings for the encoder."""
        self.encoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Return the output embeddings (projection layer)."""
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings (projection layer)."""
        self.output_projection = new_embeddings

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings for both encoder and output projection."""
        # Resize encoder embeddings
        self.encoder.resize_token_embeddings(new_num_tokens)

        # Resize output projection
        old_embeddings = self.output_projection
        new_embeddings = nn.Linear(old_embeddings.in_features, new_num_tokens)

        # Copy weights for existing tokens
        old_num_tokens = old_embeddings.out_features
        new_embeddings.weight.data[: min(old_num_tokens, new_num_tokens)] = (
            old_embeddings.weight.data[: min(old_num_tokens, new_num_tokens)]
        )

        self.output_projection = new_embeddings
        return new_embeddings

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=512,
        num_beams=4,
        early_stopping=True,
        pad_token_id=None,
        **kwargs,
    ):
        """Generate method for inference compatibility."""
        # For now, implement a simple greedy decoding
        # In a full implementation, you'd want proper beam search

        batch_size = input_ids.size(0)

        # Encode the input
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Start with input_ids as the generated sequence
        generated = input_ids.clone()

        # Simple greedy generation (can be improved with beam search)
        for _ in range(max_length - input_ids.size(1)):
            # Get decoder outputs
            decoder_outputs = self.decoder(
                tgt=encoder_outputs.last_hidden_state,
                memory=encoder_outputs.last_hidden_state,
            )
            logits = self.output_projection(decoder_outputs)

            # Get next token (greedy)
            next_token = logits[:, -1:, :].argmax(dim=-1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for early stopping (if all sequences have pad_token_id)
            if early_stopping and pad_token_id is not None:
                if (next_token == pad_token_id).all():
                    break

        return generated

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Encode the input (config + argspec)
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        if labels is not None:
            # During training, use teacher forcing
            decoder_outputs = self.decoder(
                tgt=encoder_outputs.last_hidden_state,
                memory=encoder_outputs.last_hidden_state,
            )
            logits = self.output_projection(decoder_outputs)

            # Calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}
        else:
            # During inference
            decoder_outputs = self.decoder(
                tgt=encoder_outputs.last_hidden_state,
                memory=encoder_outputs.last_hidden_state,
            )
            logits = self.output_projection(decoder_outputs)
            return {"logits": logits}


class NetworkConfigParserAI:
    """Main class for training and using the network config parser AI."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

        # Add special tokens for our domain
        special_tokens = {
            "additional_special_tokens": [
                "[CONFIG]",
                "[ARGSPEC]",
                "[PARSER]",
                "[REGEX]",
                "[JINJA]",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

    def prepare_training_data(self, parser_examples: List[Dict]) -> List[Dict]:
        """
        Convert parser examples into training format.

        Expected input format:
        {
            "config_lines": ["bgp additional-paths install receive"],
            "argspec": {...},
            "parser": {...}
        }
        """
        processed_examples = []

        for example in parser_examples:
            processed_examples.append(
                {
                    "config_lines": example["config_lines"],
                    "argspec": example["argspec"],
                    "parser": example["parser"],
                }
            )

        return processed_examples

    def create_example_from_your_format(
        self, config_lines: List[str], argspec: Dict, parser_data: Dict
    ) -> Dict:
        """Convert your existing parser format to training format."""
        return {
            "config_lines": config_lines,
            "argspec": argspec,
            "parser": {
                "name": parser_data["name"],
                "getval": (
                    parser_data["getval"].pattern
                    if hasattr(parser_data["getval"], "pattern")
                    else str(parser_data["getval"])
                ),
                "setval": parser_data["setval"],
                "result": parser_data["result"],
            },
        }

    def train(
        self, training_examples: List[Dict], output_dir: str = "./config_parser_model"
    ):
        """Train the model on parser examples."""

        # Prepare data
        train_data, val_data = train_test_split(
            training_examples, test_size=0.2, random_state=42
        )

        train_dataset = ConfigParserDataset(train_data, self.tokenizer)
        val_dataset = ConfigParserDataset(val_data, self.tokenizer)

        # Initialize model
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model_name)
        self.model = ConfigParserModel(config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Training arguments - completely disable external logging
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            eval_strategy="steps",  # Updated from evaluation_strategy
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Empty list = no external logging (wandb, tensorboard, etc.)
            disable_tqdm=False,  # Keep console progress bars
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def load_model(self, model_path: str):
        """Load a trained model with robust tokenizer handling."""
        try:
            # Try to load the tokenizer from the saved model first
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(
                f"Loaded tokenizer from saved model with {len(self.tokenizer)} tokens"
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer from model path: {e}")
            # Fallback to base tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Ensure special tokens are added (in case tokenizer doesn't have them)
        special_tokens = {
            "additional_special_tokens": [
                "[CONFIG]",
                "[ARGSPEC]",
                "[PARSER]",
                "[REGEX]",
                "[JINJA]",
            ]
        }
        # This will only add tokens that aren't already present
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            logger.info(f"Added {num_added} special tokens to tokenizer")

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path)

        # Try to load the model with more robust error handling
        try:
            self.model = ConfigParserModel.from_pretrained(model_path, config=config)
            logger.info("Successfully loaded model from checkpoint")
        except Exception as e:
            if "size mismatch" in str(e):
                logger.warning(f"Embedding size mismatch detected: {e}")
                logger.info("Attempting to load model with embedding resize...")

                # Create a fresh model and then load state dict manually
                self.model = ConfigParserModel(config)

                # Load the state dict and handle embedding mismatch
                import torch
                from pathlib import Path

                # Check for both safetensors and pytorch formats
                safetensors_path = Path(f"{model_path}/model.safetensors")
                pytorch_path = Path(f"{model_path}/pytorch_model.bin")

                if safetensors_path.exists():
                    logger.info("Loading from safetensors format")
                    from safetensors.torch import load_file

                    checkpoint = load_file(safetensors_path)
                elif pytorch_path.exists():
                    logger.info("Loading from pytorch format")
                    checkpoint = torch.load(pytorch_path, map_location="cpu")
                else:
                    raise FileNotFoundError(f"No model file found at {model_path}")

                # Get current and saved embedding sizes
                current_embed_size = self.model.get_input_embeddings().num_embeddings
                saved_embed_size = checkpoint[
                    "encoder.embeddings.word_embeddings.weight"
                ].shape[0]

                logger.info(
                    f"Current embedding size: {current_embed_size}, Saved: {saved_embed_size}"
                )

                if current_embed_size != saved_embed_size:
                    # Resize current model to match saved model
                    logger.info(
                        f"Resizing model embeddings to match saved model ({saved_embed_size})"
                    )
                    self.model.resize_token_embeddings(saved_embed_size)

                # Now load the state dict
                self.model.load_state_dict(checkpoint, strict=False)
                logger.info("Successfully loaded model with embedding size adjustment")
            else:
                raise e

        self.model.eval()

        # Final verification
        final_embed_size = self.model.get_input_embeddings().num_embeddings
        tokenizer_size = len(self.tokenizer)
        logger.info(
            f"Final model embedding size: {final_embed_size}, Tokenizer size: {tokenizer_size}"
        )

        if final_embed_size != tokenizer_size:
            logger.warning(
                f"Size mismatch remains: model={final_embed_size}, tokenizer={tokenizer_size}"
            )
            logger.info(
                "Model should still work, but may not use all tokenizer vocabulary"
            )

    def generate_parser(self, config_lines: List[str], argspec: Dict) -> Dict:
        """Generate a parser for given config lines and argspec."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        # Prepare input
        config_text = "\n".join(config_lines)
        argspec_str = json.dumps(argspec, indent=2)
        input_text = (
            f"CONFIG:\n{config_text}\n\nARGSPEC:\n{argspec_str}\n\nGENERATE_PARSER:"
        )

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # Parse the generated parser JSON
            parser_json = generated_text.split("GENERATE_PARSER:")[-1].strip()
            if parser_json:
                parser = json.loads(parser_json)
                return parser
            else:
                # If no output after GENERATE_PARSER, use fallback
                raise json.JSONDecodeError("No parser generated", "", 0)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated parser: {e}")
            logger.info("Using fallback rule-based parser generation")

            # Fallback: Generate a simple parser based on the config pattern
            return self._generate_fallback_parser(config_lines, argspec)

    def _generate_fallback_parser(self, config_lines: List[str], argspec: Dict) -> Dict:
        """Generate a simple rule-based parser when AI generation fails."""
        if not config_lines:
            return {"error": "No config lines provided"}

        config_line = config_lines[0]

        # Basic pattern matching for common network commands
        import re

        # Extract command components
        words = config_line.strip().split()
        if not words:
            return {"error": "Empty config line"}

        # Generate basic regex pattern
        pattern_parts = []
        result_parts = {}

        for i, word in enumerate(words):
            if word.isdigit():
                # Numeric value
                var_name = f"value_{i}"
                pattern_parts.append(rf"(?P<{var_name}>\d+)")
                result_parts[var_name] = f"{{{{ {var_name} }}}}"
            elif "-" in word and any(c.isdigit() for c in word):
                # Range like "20-24"
                var_name = f"range_{i}"
                pattern_parts.append(rf"(?P<{var_name}>[\d-]+)")
                result_parts[var_name] = f"{{{{ {var_name} }}}}"
            elif word in ["enable", "disable", "activate", "primary", "secondary"]:
                # Boolean keywords
                var_name = word
                pattern_parts.append(rf"(?P<{var_name}>{word})?")
                result_parts[var_name] = f"{{{{ not not {var_name} }}}}"
            else:
                # Literal word
                pattern_parts.append(re.escape(word))

        # Construct the regex - use raw string format
        getval_pattern = r"\s+" + r"\s+".join(pattern_parts) + r"$"

        # Construct the result structure
        result_dict = {}
        current_dict = result_dict

        # Try to match argspec structure
        for key, value in argspec.items():
            if isinstance(value, dict) and "options" in value:
                current_dict[key] = result_parts
                break

        if not result_dict and result_parts:
            # Simple fallback structure
            main_key = words[0] if words else "config"
            result_dict[main_key] = result_parts

        return {
            "name": f"{words[0]}.generated" if words else "unknown.generated",
            "getval": getval_pattern,
            "setval": config_line,  # Simple setval
            "result": result_dict,
            "note": "Generated by fallback rule-based logic",
        }

    def suggest_parser_improvements(
        self, existing_parser: Dict, config_lines: List[str]
    ) -> List[str]:
        """Suggest improvements to an existing parser."""
        suggestions = []

        # Check if regex covers all config variations
        regex_pattern = existing_parser.get("getval", "")
        for line in config_lines:
            if isinstance(regex_pattern, str):
                if not re.search(regex_pattern, line):
                    suggestions.append(f"Regex doesn't match line: {line}")
            else:
                try:
                    if not regex_pattern.search(line):
                        suggestions.append(f"Regex doesn't match line: {line}")
                except:
                    suggestions.append("Invalid regex pattern")

        # Check for missing optional groups
        optional_keywords = ["activate", "disable", "enable", "all", "best"]
        for keyword in optional_keywords:
            if any(keyword in line for line in config_lines):
                if keyword not in str(regex_pattern):
                    suggestions.append(
                        f"Consider adding optional group for keyword: {keyword}"
                    )

        return suggestions


# Example usage and data preparation functions
def create_sample_training_data():
    """Create sample training data in the expected format."""

    # Your BGP examples converted to training format
    bgp_argspec = {
        "bgp": {
            "type": "dict",
            "options": {
                "additional_paths": {
                    "type": "dict",
                    "options": {
                        "install": {"type": "bool"},
                        "receive": {"type": "bool"},
                        "select": {
                            "type": "dict",
                            "options": {
                                "all": {"type": "bool"},
                                "best": {"type": "int"},
                                "best_external": {"type": "bool"},
                                "group_best": {"type": "bool"},
                            },
                        },
                        "send": {"type": "bool"},
                    },
                },
                "graceful_shutdown": {
                    "type": "dict",
                    "options": {
                        "neighbors": {
                            "type": "dict",
                            "options": {
                                "time": {"type": "int"},
                                "activate": {"type": "bool"},
                            },
                        },
                        "vrfs": {
                            "type": "dict",
                            "options": {
                                "time": {"type": "int"},
                                "activate": {"type": "bool"},
                            },
                        },
                        "community": {"type": "str"},
                        "local_preference": {"type": "int"},
                    },
                },
            },
        }
    }

    training_examples = [
        {
            "config_lines": ["bgp additional-paths install receive"],
            "argspec": bgp_argspec,
            "parser": {
                "name": "bgp.additional_paths",
                "getval": r"\sbgp\sadditional-paths(\s(?P<install>install))?(\s(?P<receive>receive))?(\s(?P<select>select))?(\s(?P<send>send))?$",
                "setval": "bgp additional-paths{{ (' install' ) if bgp.additional_paths.install|d(False) else '' }}{{ (' receive' ) if bgp.additional_paths.receive|d(False) else '' }}{{ (' select' ) if bgp.additional_paths.select|d(False) else '' }}{{ (' send' ) if bgp.additional_paths.send|d(False) else '' }}",
                "result": {
                    "bgp": {
                        "additional_paths": {
                            "install": "{{ not not install }}",
                            "receive": "{{ not not receive }}",
                            "select": "{{ not not select }}",
                            "send": "{{ not not send }}",
                        },
                    },
                },
            },
        },
        {
            "config_lines": [
                "bgp graceful-shutdown all vrfs 31 local-preference 230 community 77"
            ],
            "argspec": bgp_argspec,
            "parser": {
                "name": "bgp.graceful_shutdown.vrfs",
                "getval": r"\sbgp\sgraceful-shutdown\sall\svrfs(\s(?P<time>\d+))?(\s(?P<activate>activate))?(\slocal-preference\s(?P<local_preference>\d+))?(\scommunity\s(?P<community>\S+))?$",
                "setval": "bgp graceful-shutdown all vrfs{{ (' ' + bgp.graceful_shutdown.vrfs.time|string) if bgp.graceful_shutdown.vrfs.time is defined else '' }}{{ (' activate') if bgp.graceful_shutdown.vrfs.activate|d(False) else '' }}{{ (' local-preference ' + bgp.graceful_shutdown.local_preference|string) if bgp.graceful_shutdown.local_preference is defined else '' }}{{ (' community ' + bgp.graceful_shutdown.community|string) if bgp.graceful_shutdown.community is defined else '' }}",
                "result": {
                    "bgp": {
                        "graceful_shutdown": {
                            "vrfs": {
                                "time": "{{ time }}",
                                "activate": "{{ not not activate }}",
                            },
                            "community": "{{ community }}",
                            "local_preference": "{{ local_preference }}",
                        },
                    },
                },
            },
        },
    ]

    return training_examples


def main():
    """Example usage of the NetworkConfigParserAI system."""

    # Initialize the AI system
    parser_ai = NetworkConfigParserAI()

    # Create sample training data
    training_data = create_sample_training_data()

    # Add more training examples here...
    # You would load your existing parser library and convert to this format

    # Train the model
    logger.info("Training the model...")
    parser_ai.train(training_data)

    # Example inference
    logger.info("Testing inference...")
    parser_ai.load_model("./config_parser_model")

    # Test with new config lines
    test_config = ["bgp additional-paths send"]
    test_argspec = {
        "bgp": {
            "type": "dict",
            "options": {
                "additional_paths": {
                    "type": "dict",
                    "options": {"send": {"type": "bool"}},
                }
            },
        }
    }

    suggested_parser = parser_ai.generate_parser(test_config, test_argspec)
    print("Suggested parser:")
    print(json.dumps(suggested_parser, indent=2))


if __name__ == "__main__":
    main()
