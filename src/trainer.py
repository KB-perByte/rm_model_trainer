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
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)
import inspect
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

        labels = targets["input_ids"].squeeze()
        # Mask pad tokens so loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


"""
Note: A custom seq2seq has been replaced with a pretrained encoder-decoder (T5)
for stability and better generation quality.
"""


class NetworkConfigParserAI:
    """Main class for training and using the network config parser AI."""

    def __init__(self, model_name: str = "t5-small"):
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

        # Initialize model (pretrained encoder-decoder)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        # Resize embeddings to include our added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Build TrainingArguments robustly across transformers versions
        ta_params = {
            "output_dir": output_dir,
            "num_train_epochs": 10,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_dir": f"{output_dir}/logs",
            "logging_steps": 100,
            "save_steps": 1000,
            "save_total_limit": 3,
            "report_to": [],
            "disable_tqdm": False,
        }

        sig = inspect.signature(TrainingArguments.__init__)

        def supports(name: str) -> bool:
            return name in sig.parameters

        # Prefer step-based eval/save/logging when supported; keep them consistent
        eval_enabled = False
        if supports("evaluation_strategy"):
            ta_params["evaluation_strategy"] = "steps"
            ta_params["eval_steps"] = 500
            eval_enabled = True
        elif supports("evaluate_during_training"):
            ta_params["evaluate_during_training"] = True
            ta_params["eval_steps"] = 500
            eval_enabled = True

        if supports("logging_strategy"):
            ta_params["logging_strategy"] = "steps" if eval_enabled else "no"
        if supports("save_strategy"):
            ta_params["save_strategy"] = "steps" if eval_enabled else "no"

        if supports("load_best_model_at_end") and eval_enabled:
            ta_params["load_best_model_at_end"] = True
        if supports("metric_for_best_model") and eval_enabled:
            ta_params["metric_for_best_model"] = "eval_loss"
        if supports("greater_is_better") and eval_enabled:
            ta_params["greater_is_better"] = False
        # Optional mixed precision if available
        if supports("fp16"):
            ta_params["fp16"] = False

        training_args = TrainingArguments(**ta_params)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, model=self.model
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
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
            # Prefer tokenizer from saved model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(
                f"Loaded tokenizer from saved model with {len(self.tokenizer)} tokens"
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer from model path: {e}")
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

        # Load model (T5) and handle potential embedding size differences
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            logger.info("Successfully loaded model from checkpoint")
        except Exception as e:
            logger.warning(
                f"Model load warning: {e}. Loading base and adjusting embeddings if needed."
            )
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

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
                max_new_tokens=256,
                num_beams=6,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Try to parse a JSON object after the instruction token
        parser_json = generated_text.split("GENERATE_PARSER:")[-1].strip()

        parser = self._parse_and_validate_json(parser_json)
        if parser is None:
            # Attempt to salvage a JSON block from the full text (best-effort)
            parser = self._parse_and_validate_json(
                self._extract_json_block(generated_text)
            )

        if parser is None:
            logger.error(
                "Failed to parse any valid parser JSON from model output. Falling back."
            )
            return self._generate_fallback_parser(config_lines, argspec)

        # Validate and repair regex + structure
        repaired = self._validate_and_repair_parser(parser, config_lines, argspec)
        if repaired is None:
            logger.error("Validation/repair failed. Using fallback parser.")
            return self._generate_fallback_parser(config_lines, argspec)

        return repaired

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

    def _extract_json_block(self, text: str) -> str:
        """Best-effort extraction of a JSON object from free-form text."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return ""

    def _parse_and_validate_json(self, s: str) -> Optional[Dict]:
        """Parse JSON string into a dict; return None if invalid/empty."""
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception as e:
            logger.debug(f"JSON parse error: {e}")
            return None
        if not isinstance(obj, dict):
            logger.debug("Parsed JSON is not an object")
            return None
        return obj

    def _validate_and_repair_parser(
        self, parser: Dict, config_lines: List[str], argspec: Dict
    ) -> Optional[Dict]:
        """Ensure required fields exist; repair common regex issues; return None if irreparable."""
        parser = dict(parser)  # shallow copy

        # Ensure keys exist
        if "getval" not in parser or not isinstance(parser.get("getval"), str):
            logger.debug("Missing or invalid 'getval' in parser JSON")
            return None
        if "result" not in parser or not isinstance(parser.get("result"), dict):
            logger.debug("Missing or invalid 'result' in parser JSON")
            return None
        if "name" not in parser or not isinstance(parser.get("name"), str):
            # Synthesize a name from the first word as a fallback
            first_word = (
                config_lines[0].strip().split()[0]
                if config_lines and config_lines[0].strip()
                else "config"
            )
            parser["name"] = f"{first_word}.generated"

        pattern = parser.get("getval", "")
        fixed_pattern = self._repair_regex(pattern)

        # Try compile original; if fails, try repaired
        try:
            re.compile(pattern)
            compiled_ok = True
        except Exception:
            compiled_ok = False

        if not compiled_ok:
            try:
                re.compile(fixed_pattern)
                parser["getval"] = fixed_pattern
                compiled_ok = True
            except Exception as e:
                logger.debug(f"Regex still invalid after repair: {e}")
                return None

        return parser

    def _repair_regex(self, pattern: str) -> str:
        """Attempt simple repairs for common regex issues: unbalanced parens, stray code fences."""
        p = pattern.strip()
        # Strip code fences/backticks if present
        if p.startswith("```") and p.endswith("```"):
            p = p.strip("`")
        # Balance parentheses by adding missing closers
        open_paren = p.count("(")
        close_paren = p.count(")")
        if open_paren > close_paren:
            p = p + (")" * (open_paren - close_paren))
        elif close_paren > open_paren:
            # remove extra trailing closers conservatively
            extra = close_paren - open_paren
            i = len(p) - 1
            while extra > 0 and i >= 0:
                if p[i] == ")":
                    p = p[:i] + p[i + 1 :]
                    extra -= 1
                i -= 1
        # Remove non-printable characters
        p = "".join(ch for ch in p if ch.isprintable())
        return p

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
