#!/usr/bin/env python3
"""
Collection-based Network Configuration Parser AI Trainer

This module is specifically designed to work with Ansible collections structure
and can load argspecs and parser templates from collection paths.
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from .trainer import NetworkConfigParserAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollectionDataLoader:
    """Loads training data from Ansible collection structure."""

    def __init__(self, collection_base_path: str = None):
        """
        Initialize with the base path to collections.

        Args:
            collection_base_path: Path like '/home/sagpaul/Work/AnsibleNetwork/collections'
        """
        if collection_base_path is None:
            # Default relative path from current script location
            current_dir = Path(__file__).parent
            self.collection_base_path = current_dir / "../../../collections"
        else:
            self.collection_base_path = Path(collection_base_path)

        self.collection_base_path = self.collection_base_path.resolve()
        logger.info(f"Collection base path: {self.collection_base_path}")

    def load_argspec_from_path(self, argspec_path: str) -> Dict[str, Any]:
        """
        Load argspec from a directory path containing multiple argspec modules.

        Args:
            argspec_path: Relative path like 'cisco/ios/plugins/module_utils/network/ios/argspec'

        Returns:
            Dictionary containing argspec data from all modules in the directory
        """
        full_path = self.collection_base_path / "ansible_collections" / argspec_path

        if not full_path.exists():
            raise FileNotFoundError(f"Argspec directory not found: {full_path}")

        if not full_path.is_dir():
            raise ValueError(f"Argspec path is not a directory: {full_path}")

        argspec_data = {}

        # Recursively search for argspec files in subdirectories
        for py_file in full_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                # Load the Python module
                spec = importlib.util.spec_from_file_location("argspec_module", py_file)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not create module spec for {py_file}")
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for argspec classes (common pattern in Ansible collections)
                for attr_name in dir(module):
                    if not attr_name.startswith("_"):
                        attr_value = getattr(module, attr_name)

                        # Check if it's a class with argument_spec attribute
                        if hasattr(attr_value, "argument_spec"):
                            module_name = py_file.stem  # filename without extension
                            argspec_data[module_name] = attr_value.argument_spec
                            logger.info(
                                f"Loaded argspec from {py_file.name}: {module_name}"
                            )
                        elif isinstance(attr_value, dict) and "type" in str(attr_value):
                            # Fallback: if it's a dict that looks like argspec
                            module_name = py_file.stem
                            argspec_data[f"{module_name}_{attr_name}"] = attr_value

            except Exception as e:
                logger.warning(f"Failed to load argspec from {py_file}: {e}")

        if not argspec_data:
            logger.warning(f"No argspec data found in {full_path}")

        logger.info(f"Loaded {len(argspec_data)} argspec modules from {full_path}")
        return argspec_data

    def load_parsers_from_rm_templates(
        self, rm_template_path: str
    ) -> List[Dict[str, Any]]:
        """
        Load parser templates from rm_templates directory using text parsing.

        Args:
            rm_template_path: Relative path like 'cisco/ios/plugins/module_utils/network/ios/rm_templates'

        Returns:
            List of parser dictionaries
        """
        full_path = self.collection_base_path / "ansible_collections" / rm_template_path

        if not full_path.exists():
            raise FileNotFoundError(f"RM templates directory not found: {full_path}")

        parsers = []

        # Look for Python files in the rm_templates directory
        for py_file in full_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                # Parse the file as text to extract parser data
                parser_data_list = self._extract_parsers_from_file(py_file)
                for parser_data in parser_data_list:
                    parsers.append(
                        {"source_file": py_file.name, "parser_data": parser_data}
                    )

            except Exception as e:
                logger.warning(f"Failed to load parsers from {py_file}: {e}")

        logger.info(f"Loaded {len(parsers)} parsers from {full_path}")
        return parsers

    def _extract_parsers_from_file(self, py_file: Path) -> List[Dict[str, Any]]:
        """
        Extract parser data from a Python file using text parsing.

        This avoids dependency issues by parsing the file content directly.
        """
        import ast
        import re

        try:
            # Read the file content
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the Python AST
            tree = ast.parse(content)
            parsers = []

            # Look for class definitions with PARSERS attribute
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if this is a template class
                    for item in node.body:
                        if (
                            isinstance(item, ast.Assign)
                            and len(item.targets) == 1
                            and isinstance(item.targets[0], ast.Name)
                            and item.targets[0].id == "PARSERS"
                        ):

                            # Extract parser data from the PARSERS assignment
                            try:
                                # Convert the AST back to a literal value
                                parser_list = ast.literal_eval(item.value)
                                if isinstance(parser_list, list):
                                    for parser_item in parser_list:
                                        if (
                                            isinstance(parser_item, dict)
                                            and "getval" in parser_item
                                        ):
                                            # Clean up the parser data
                                            clean_parser = self._clean_parser_data(
                                                parser_item
                                            )
                                            parsers.append(clean_parser)
                            except (ValueError, TypeError) as e:
                                # If literal_eval fails, try regex extraction as fallback
                                logger.debug(
                                    f"AST parsing failed for {py_file}, trying regex: {e}"
                                )
                                parsers.extend(
                                    self._extract_parsers_with_regex(content)
                                )
                                break

            return parsers

        except Exception as e:
            logger.warning(f"Failed to extract parsers from {py_file}: {e}")
            return []

    def _clean_parser_data(self, parser_dict: Dict) -> Dict:
        """
        Clean parser data to ensure it's suitable for training.
        """
        cleaned = {}

        for key, value in parser_dict.items():
            if key in ["name", "getval", "setval", "result"]:
                if isinstance(value, str):
                    cleaned[key] = value
                elif hasattr(value, "pattern"):  # regex object
                    cleaned[key] = value.pattern
                else:
                    cleaned[key] = str(value)

        return cleaned

    def _extract_parsers_with_regex(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract parser data using regex patterns for parser dictionaries.
        """
        import re

        parsers = []

        # Look for name fields first, then find their context
        name_pattern = r'"name"\s*:\s*"([^"]+)"'
        name_matches = re.finditer(name_pattern, content)

        for name_match in name_matches:
            name = name_match.group(1)
            start_pos = name_match.start()

            # Find the start of this dictionary (look backwards for {)
            dict_start = content.rfind("{", 0, start_pos)
            if dict_start == -1:
                continue

            # Find the end of this dictionary
            brace_count = 0
            dict_end = dict_start
            for i, char in enumerate(content[dict_start:], dict_start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        dict_end = i
                        break

            if dict_end == dict_start:
                continue

            dict_content = content[dict_start : dict_end + 1]

            # Extract getval pattern - look for r""" patterns
            getval_match = re.search(r'r"""([^"]+)"""', dict_content, re.DOTALL)
            if getval_match:
                getval_pattern = getval_match.group(1).strip()

                # Clean up the regex pattern - remove extra whitespace and newlines
                getval_clean = re.sub(r"\s+", " ", getval_pattern).strip()

                parser_data = {
                    "name": name,
                    "getval": getval_clean,
                    "setval": "",
                    "result": {},
                }
                parsers.append(parser_data)

        return parsers

    def create_training_data(
        self, argspec_path: str, rm_template_path: str
    ) -> List[Dict[str, Any]]:
        """
        Create training data by combining argspecs and parser templates.

        Args:
            argspec_path: Path to argspec files
            rm_template_path: Path to rm_template files

        Returns:
            List of training examples
        """
        logger.info(f"Loading argspecs from: {argspec_path}")
        logger.info(f"Loading parsers from: {rm_template_path}")

        # Load argspecs
        argspec_data = self.load_argspec_from_path(argspec_path)

        # Load parsers
        parsers = self.load_parsers_from_rm_templates(rm_template_path)

        # Create training examples
        training_examples = []
        parser_ai = NetworkConfigParserAI()

        for parser_info in parsers:
            parser_data = parser_info["parser_data"]

            try:
                # Generate example config lines from parser
                config_lines = self._generate_config_lines_from_parser(parser_data)

                # Find matching argspec
                matching_argspec = self._find_matching_argspec(
                    parser_data, argspec_data
                )

                if matching_argspec:
                    training_example = parser_ai.create_example_from_your_format(
                        config_lines=config_lines,
                        argspec=matching_argspec,
                        parser_data=parser_data,
                    )

                    training_example["metadata"] = {
                        "source_file": parser_info["source_file"],
                        "parser_name": parser_data.get("name", "unknown"),
                    }

                    training_examples.append(training_example)
                else:
                    logger.warning(
                        f"No matching argspec found for parser: {parser_data.get('name', 'unknown')}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create training example from parser {parser_data.get('name', 'unknown')}: {e}"
                )

        logger.info(f"Created {len(training_examples)} training examples")
        return training_examples

    def _generate_config_lines_from_parser(self, parser_data: Dict) -> List[str]:
        """Generate example configuration lines from parser data."""
        config_lines = []

        parser_name = parser_data.get("name", "")

        # Use parser name to generate realistic config examples
        name_parts = parser_name.split(".")

        if "bgp" in name_parts:
            if "additional_paths" in parser_name:
                config_lines = ["bgp additional-paths install receive"]
            elif "graceful_shutdown" in parser_name:
                config_lines = ["bgp graceful-shutdown all vrfs 30 activate"]
            elif "neighbor" in parser_name:
                config_lines = ["bgp neighbor 192.168.1.1 remote-as 65001"]
            else:
                config_lines = ["bgp router-id 1.1.1.1"]

        elif "interface" in name_parts:
            if "description" in parser_name:
                config_lines = [
                    "interface GigabitEthernet0/1 description uplink-to-core"
                ]
            elif "ip" in parser_name:
                config_lines = [
                    "interface GigabitEthernet0/1 ip address 192.168.1.1 255.255.255.0"
                ]
            else:
                config_lines = ["interface GigabitEthernet0/1"]

        elif "vlan" in name_parts:
            config_lines = ["vlan 100 name production"]

        elif "ospf" in name_parts:
            config_lines = ["router ospf 1 router-id 1.1.1.1"]

        else:
            # Fallback: create basic config from parser name
            config_base = " ".join(name_parts)
            config_lines = [config_base]

        return config_lines

    def _find_matching_argspec(self, parser_data: Dict, argspec_data: Dict) -> Dict:
        """Find the argspec that matches the parser data."""
        parser_name = parser_data.get("name", "")

        # Try to find argspec by matching parser name structure
        name_parts = parser_name.split(".")

        for argspec_name, argspec in argspec_data.items():
            if isinstance(argspec, dict):
                # Check if the first part of parser name matches argspec keys
                if name_parts and name_parts[0] in str(argspec).lower():
                    return argspec

                # Check if argspec name matches parser name parts
                if any(part in argspec_name.lower() for part in name_parts):
                    return argspec

        # Fallback: return the first valid argspec
        for argspec in argspec_data.values():
            if isinstance(argspec, dict) and "type" in str(argspec):
                return argspec

        return {}


class CollectionBasedTrainer:
    """Main trainer class for collection-based training."""

    def __init__(self, collection_base_path: str = None):
        self.loader = CollectionDataLoader(collection_base_path)
        self.parser_ai = NetworkConfigParserAI()

        # Model storage path - always local to the project
        self.model_storage_path = Path(__file__).parent / "trained_models"
        self.model_storage_path.mkdir(exist_ok=True)

        print(f"🏠 Model storage path: {self.model_storage_path.resolve()}")

    def train_from_collections(
        self,
        argspec_paths: List[str],
        rm_template_paths: List[str],
        model_name: str = "collection_parser_model",
    ) -> str:
        """
        Train model using data from multiple collections.

        Args:
            argspec_paths: List of relative paths to argspec directories
            rm_template_paths: List of relative paths to rm_template directories
            model_name: Name for the trained model

        Returns:
            Path to the saved model
        """
        all_training_data = []

        # Ensure we have matching pairs of paths
        for i, (argspec_path, rm_template_path) in enumerate(
            zip(argspec_paths, rm_template_paths)
        ):
            logger.info(f"Processing collection {i+1}/{len(argspec_paths)}")

            try:
                training_data = self.loader.create_training_data(
                    argspec_path, rm_template_path
                )
                all_training_data.extend(training_data)
            except Exception as e:
                logger.error(f"Failed to load data from {argspec_path}: {e}")

        if not all_training_data:
            raise ValueError(
                "No training data was loaded. Check your paths and file formats."
            )

        logger.info(f"Total training examples: {len(all_training_data)}")

        # Create model directory
        model_path = self.model_storage_path / model_name
        model_path.mkdir(exist_ok=True)

        print(f"📂 Training model, will be saved to: {model_path.resolve()}")

        # Train the model
        self.parser_ai.train(all_training_data, output_dir=str(model_path))

        # Save metadata
        metadata = {
            "model_name": model_name,
            "training_examples": len(all_training_data),
            "argspec_paths": argspec_paths,
            "rm_template_paths": rm_template_paths,
            "collection_base_path": str(self.loader.collection_base_path),
        }

        with open(model_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model training completed!")
        print(f"📍 Model saved at: {model_path.resolve()}")

        return str(model_path)

    def load_and_test_model(
        self, model_path: str, test_config: str, test_argspec: Dict
    ) -> Dict:
        """Load a trained model and test it."""
        print(f"🔮 Loading model from: {Path(model_path).resolve()}")

        self.parser_ai.load_model(model_path)

        generated_parser = self.parser_ai.generate_parser([test_config], test_argspec)

        print(f"✨ Generated parser for: {test_config}")
        return generated_parser


def main():
    """Example usage with collection paths."""

    print("🎓 Collection-Based Network Configuration Parser AI")
    print("=" * 60)

    # Initialize trainer
    trainer = CollectionBasedTrainer()

    # Define your collection paths (relative to collections directory)
    argspec_paths = [
        "cisco/ios/plugins/module_utils/network/ios/argspec",
        # Add more argspec paths as needed
    ]

    rm_template_paths = [
        "cisco/ios/plugins/module_utils/network/ios/rm_templates",
        # Add more rm_template paths as needed
    ]

    try:
        # Train the model
        model_path = trainer.train_from_collections(
            argspec_paths=argspec_paths,
            rm_template_paths=rm_template_paths,
            model_name="cisco_ios_parser_model",
        )

        # Test the trained model
        test_config = "bgp additional-paths send"
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

        print("\n" + "=" * 60)
        print("🧪 Testing the trained model...")

        generated_parser = trainer.load_and_test_model(
            model_path, test_config, test_argspec
        )
        print(json.dumps(generated_parser, indent=2))

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Error: {e}")
        print(f"💡 Make sure your collection paths are correct and accessible")


if __name__ == "__main__":
    main()
