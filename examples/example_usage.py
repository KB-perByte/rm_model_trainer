#!/usr/bin/env python3
"""
Example usage script for the Network Configuration Parser AI

This script demonstrates how to:
1. Prepare training data from your existing parsers
2. Train the model
3. Generate new parsers for unseen configurations
"""

import json
import re
from src.trainer import NetworkConfigParserAI, create_sample_training_data


def create_training_data_from_your_parsers():
    """
    Example of how to convert your existing parser format to training data.

    This function shows how to take your existing parser definitions
    and convert them to the format expected by the AI trainer.
    """

    # Example argspec for BGP configurations
    bgp_argspec = {
        "bgp": {
            "type": "dict",
            "options": {
                "additional_paths": {
                    "type": "dict",
                    "options": {
                        "install": {"type": "bool"},
                        "receive": {"type": "bool"},
                        "select": {"type": "bool"},
                        "send": {"type": "bool"},
                    },
                },
                "graceful_shutdown": {
                    "type": "dict",
                    "options": {
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

    # Your existing parser data (from input_info format)
    existing_parsers = [
        {
            "config_lines": ["bgp additional-paths install receive"],
            "parser_data": {
                "name": "bgp.additional_paths",
                "getval": re.compile(
                    r"""
                    \sbgp\sadditional-paths
                    (\s(?P<install>install))?
                    (\s(?P<receive>receive))?
                    (\s(?P<select>select))?
                    (\s(?P<send>send))?
                    $""",
                    re.VERBOSE,
                ),
                "setval": (
                    "bgp additional-paths"
                    "{{ (' install' ) if bgp.additional_paths.install|d(False) else '' }}"
                    "{{ (' receive' ) if bgp.additional_paths.receive|d(False) else '' }}"
                    "{{ (' select' ) if bgp.additional_paths.select|d(False) else '' }}"
                    "{{ (' send' ) if bgp.additional_paths.send|d(False) else '' }}"
                ),
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
            "parser_data": {
                "name": "bgp.graceful_shutdown.vrfs",
                "getval": re.compile(
                    r"""
                    \sbgp\sgraceful-shutdown\sall\svrfs
                    (\s(?P<time>\d+))?
                    (\s(?P<activate>activate))?
                    (\slocal-preference\s(?P<local_preference>\d+))?
                    (\scommunity\s(?P<community>\S+))?
                    $""",
                    re.VERBOSE,
                ),
                "setval": (
                    "bgp graceful-shutdown all vrfs"
                    "{{ (' ' + bgp.graceful_shutdown.vrfs.time|string) if bgp.graceful_shutdown.vrfs.time is defined else '' }}"
                    "{{ (' activate') if bgp.graceful_shutdown.vrfs.activate|d(False) else '' }}"
                    "{{ (' local-preference ' + bgp.graceful_shutdown.local_preference|string) if bgp.graceful_shutdown.local_preference is defined else '' }}"
                    "{{ (' community ' + bgp.graceful_shutdown.community|string) if bgp.graceful_shutdown.community is defined else '' }}"
                ),
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

    # Initialize the AI system
    parser_ai = NetworkConfigParserAI()

    # Convert each parser to training format
    training_examples = []
    for example in existing_parsers:
        training_example = parser_ai.create_example_from_your_format(
            config_lines=example["config_lines"],
            argspec=bgp_argspec,
            parser_data=example["parser_data"],
        )
        training_examples.append(training_example)

    return training_examples


def train_model_example():
    """Example of training the model with your data."""

    print("🚀 Starting model training example...")

    # Initialize the AI system
    parser_ai = NetworkConfigParserAI()

    # Get training data (combine sample data with your converted parsers)
    sample_data = create_sample_training_data()
    your_data = create_training_data_from_your_parsers()

    # Combine all training data
    all_training_data = sample_data + your_data

    print(f"📊 Training with {len(all_training_data)} examples")

    # Train the model
    model_output_dir = "./trained_parser_model"
    parser_ai.train(all_training_data, output_dir=model_output_dir)

    print(f"✅ Model training completed! Saved to: {model_output_dir}")
    return model_output_dir


def generate_parser_example(model_path):
    """Example of generating a new parser with the trained model."""

    print("🎯 Testing parser generation...")

    # Initialize and load the trained model
    parser_ai = NetworkConfigParserAI()
    parser_ai.load_model(model_path)

    # Test configuration that the model hasn't seen before
    test_config = ["bgp additional-paths send select"]
    test_argspec = {
        "bgp": {
            "type": "dict",
            "options": {
                "additional_paths": {
                    "type": "dict",
                    "options": {"send": {"type": "bool"}, "select": {"type": "bool"}},
                }
            },
        }
    }

    print(f"📝 Input config: {test_config[0]}")
    print("📋 Input argspec:")
    print(json.dumps(test_argspec, indent=2))

    # Generate parser
    suggested_parser = parser_ai.generate_parser(test_config, test_argspec)

    print("🔮 Generated parser:")
    print(json.dumps(suggested_parser, indent=2))

    return suggested_parser


def validate_parser_example():
    """Example of validating and improving existing parsers."""

    print("🔍 Testing parser validation...")

    parser_ai = NetworkConfigParserAI()

    # Example existing parser that might need improvement
    existing_parser = {
        "name": "bgp.neighbor",
        "getval": r"bgp neighbor (?P<ip>\d+\.\d+\.\d+\.\d+)",
        "setval": "bgp neighbor {{ bgp.neighbor.ip }}",
        "result": {"bgp": {"neighbor": {"ip": "{{ ip }}"}}},
    }

    # Test configurations that should match
    test_configs = [
        "bgp neighbor 192.168.1.1",
        "bgp neighbor 10.0.0.1 remote-as 65001",  # This won't match the current regex
        "bgp neighbor 172.16.0.1 description peer-router",  # This won't match either
    ]

    suggestions = parser_ai.suggest_parser_improvements(existing_parser, test_configs)

    print("📊 Parser validation results:")
    for suggestion in suggestions:
        print(f"  ⚠️  {suggestion}")

    return suggestions


def main():
    """Main example function demonstrating the complete workflow."""

    print("🎓 Network Configuration Parser AI - Example Usage")
    print("=" * 60)

    try:
        # Step 1: Train the model
        model_path = train_model_example()

        print("\n" + "=" * 60)

        # Step 2: Generate new parsers
        generate_parser_example(model_path)

        print("\n" + "=" * 60)

        # Step 3: Validate existing parsers
        validate_parser_example()

        print("\n✨ Example completed successfully!")
        print(f"📁 Your trained model is saved in: {model_path}")
        print(
            "🚀 You can now use this model to generate parsers for new configurations!"
        )

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("💡 Make sure you have installed all dependencies from requirements.txt")


if __name__ == "__main__":
    main()
