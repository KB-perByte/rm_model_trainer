#!/usr/bin/env python3
"""
Data preparation utilities for the Network Configuration Parser AI

This module provides utilities to help convert existing parser libraries
into the training format expected by the AI model.
"""

import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Union
from trainer import NetworkConfigParserAI


class ParserDataConverter:
    """Utility class to convert various parser formats to AI training format."""

    def __init__(self):
        self.parser_ai = NetworkConfigParserAI()

    def load_parsers_from_file(self, file_path: str) -> List[Dict]:
        """
        Load parser definitions from JSON or YAML file.

        Expected file format:
        {
            "parsers": [
                {
                    "config_lines": ["bgp neighbor 1.2.3.4"],
                    "argspec": {...},
                    "parser": {...}
                }
            ]
        }
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Parser file not found: {file_path}")

        with open(path, "r") as f:
            if path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return data.get("parsers", [])

    def convert_ansible_rm_parsers(
        self, rm_parsers: List[Dict], argspec: Dict
    ) -> List[Dict]:
        """
        Convert Ansible Resource Module parser format to AI training format.

        Args:
            rm_parsers: List of parsers in RM format
            argspec: The argspec for these parsers

        Returns:
            List of training examples
        """
        training_examples = []

        for parser in rm_parsers:
            # Extract config lines that this parser should match
            # This is a simplified example - you might need more sophisticated
            # config line generation based on your parser patterns
            config_lines = self._generate_config_lines_from_parser(parser)

            # Convert to training format
            training_example = self.parser_ai.create_example_from_your_format(
                config_lines=config_lines, argspec=argspec, parser_data=parser
            )

            training_examples.append(training_example)

        return training_examples

    def _generate_config_lines_from_parser(self, parser: Dict) -> List[str]:
        """
        Generate example config lines from a parser definition.

        This is a heuristic approach - you might want to customize this
        based on your specific parser patterns.
        """
        config_lines = []

        # Try to extract base command from parser name
        parser_name = parser.get("name", "")

        # Simple heuristics based on common patterns
        if "bgp" in parser_name.lower():
            if "additional_paths" in parser_name:
                config_lines = ["bgp additional-paths install receive"]
            elif "graceful_shutdown" in parser_name:
                config_lines = ["bgp graceful-shutdown all vrfs 30 activate"]
            elif "neighbor" in parser_name:
                config_lines = ["bgp neighbor 192.168.1.1 remote-as 65001"]
        elif "interface" in parser_name.lower():
            config_lines = ["interface ethernet 1/1 description uplink"]
        elif "vlan" in parser_name.lower():
            config_lines = ["vlan 100 name production"]
        else:
            # Fallback: use parser name as base
            config_lines = [parser_name.replace(".", " ")]

        return config_lines

    def validate_training_data(self, training_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate training data format and provide statistics.

        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {
                "total_examples": len(training_data),
                "unique_parsers": set(),
                "config_families": set(),
                "avg_config_length": 0,
            },
        }

        total_config_length = 0

        for i, example in enumerate(training_data):
            # Check required fields
            required_fields = ["config_lines", "argspec", "parser"]
            for field in required_fields:
                if field not in example:
                    results["errors"].append(
                        f"Example {i}: Missing required field '{field}'"
                    )
                    results["valid"] = False

            if "parser" in example:
                parser = example["parser"]

                # Check parser structure
                parser_required = ["name", "getval", "setval", "result"]
                for field in parser_required:
                    if field not in parser:
                        results["errors"].append(
                            f"Example {i}: Parser missing '{field}'"
                        )
                        results["valid"] = False

                # Collect statistics
                if "name" in parser:
                    results["stats"]["unique_parsers"].add(parser["name"])

                    # Extract config family (first part of parser name)
                    family = parser["name"].split(".")[0]
                    results["stats"]["config_families"].add(family)

            if "config_lines" in example:
                config_lines = example["config_lines"]
                if isinstance(config_lines, list):
                    total_config_length += sum(len(line) for line in config_lines)
                else:
                    results["warnings"].append(
                        f"Example {i}: config_lines should be a list"
                    )

        # Calculate averages
        if len(training_data) > 0:
            results["stats"]["avg_config_length"] = total_config_length / len(
                training_data
            )

        # Convert sets to lists for JSON serialization
        results["stats"]["unique_parsers"] = list(results["stats"]["unique_parsers"])
        results["stats"]["config_families"] = list(results["stats"]["config_families"])

        return results

    def augment_training_data(
        self, training_data: List[Dict], augmentation_factor: int = 2
    ) -> List[Dict]:
        """
        Augment training data by creating variations of existing examples.

        Args:
            training_data: Original training examples
            augmentation_factor: How many variations to create per example

        Returns:
            Augmented training data
        """
        augmented_data = training_data.copy()

        for example in training_data:
            for i in range(augmentation_factor):
                augmented_example = self._create_variation(example, i)
                if augmented_example:
                    augmented_data.append(augmented_example)

        return augmented_data

    def _create_variation(self, example: Dict, variation_num: int) -> Dict:
        """
        Create a variation of a training example.

        This could involve:
        - Reordering optional parameters
        - Using different parameter values
        - Adding/removing optional components
        """
        # This is a simplified implementation
        # In practice, you'd want more sophisticated variation strategies

        config_lines = example["config_lines"]
        if not config_lines:
            return None

        # Simple variation: add some common optional parameters
        base_config = config_lines[0]

        variations = []
        if "bgp" in base_config:
            variations = [
                f"{base_config} activate",
                f"{base_config} send-community",
                f"{base_config} next-hop-self",
            ]
        elif "interface" in base_config:
            variations = [
                f"{base_config} shutdown",
                f"{base_config} no shutdown",
                f"{base_config} mtu 1500",
            ]

        if variation_num < len(variations):
            varied_example = example.copy()
            varied_example["config_lines"] = [variations[variation_num]]
            return varied_example

        return None

    def save_training_data(self, training_data: List[Dict], output_path: str):
        """Save training data to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "training_data": training_data,
                    "metadata": {
                        "total_examples": len(training_data),
                        "created_by": "ParserDataConverter",
                        "format_version": "1.0",
                    },
                },
                f,
                indent=2,
            )

        print(f"✅ Saved {len(training_data)} training examples to {output_path}")


def main():
    """Example usage of the data preparation utilities."""

    print("📊 Data Preparation Utilities Example")
    print("=" * 50)

    converter = ParserDataConverter()

    # Example: Create some sample training data
    sample_data = [
        {
            "config_lines": ["bgp additional-paths install"],
            "argspec": {
                "bgp": {
                    "type": "dict",
                    "options": {
                        "additional_paths": {
                            "type": "dict",
                            "options": {"install": {"type": "bool"}},
                        }
                    },
                }
            },
            "parser": {
                "name": "bgp.additional_paths",
                "getval": r"bgp additional-paths (?P<install>install)?",
                "setval": "bgp additional-paths{{ ' install' if bgp.additional_paths.install else '' }}",
                "result": {
                    "bgp": {"additional_paths": {"install": "{{ not not install }}"}}
                },
            },
        }
    ]

    # Validate the data
    print("🔍 Validating training data...")
    validation_results = converter.validate_training_data(sample_data)

    print(f"Valid: {validation_results['valid']}")
    print(f"Total examples: {validation_results['stats']['total_examples']}")
    print(f"Config families: {validation_results['stats']['config_families']}")

    if validation_results["errors"]:
        print("❌ Errors found:")
        for error in validation_results["errors"]:
            print(f"  - {error}")

    # Augment the data
    print("\n📈 Augmenting training data...")
    augmented_data = converter.augment_training_data(sample_data, augmentation_factor=2)
    print(f"Original examples: {len(sample_data)}")
    print(f"Augmented examples: {len(augmented_data)}")

    # Save the data
    print("\n� Saving training data...")
    converter.save_training_data(augmented_data, "./prepared_training_data.json")

    print("\n✨ Data preparation completed!")


if __name__ == "__main__":
    main()
