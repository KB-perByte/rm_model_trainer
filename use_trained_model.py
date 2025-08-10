#!/usr/bin/env python3
"""
Example script showing how to use a trained parser model.
"""

from src.collection_trainer import CollectionBasedTrainer
import json


def main():
    print("🤖 Using Trained Parser Model")
    print("=" * 40)

    # Initialize trainer
    trainer = CollectionBasedTrainer()

    # Path to your trained model
    model_path = "./trained_models/multi_vendor_parser_model"

    # Example: Generate parser for new config line
    test_config = "spanning-tree vlan 20-24 root primary diameter 4"
    test_argspec = {
        "spanning_tree ": {
            "type": "dict",
            "options": {
                "vlan_options": {
                    "type": "dict",
                    "options": {
                        "vlan": {"type": "str"},
                        "root": {"type": "bool"},
                        "primary": {"type": "bool"},
                        "diameter": {"type": "int"},
                    },
                }
            },
        }
    }

    print(f"🔍 Input config: {test_config}")
    print(f"📋 Input argspec: {json.dumps(test_argspec, indent=2)}")
    print("\n🎯 Generating parser...")

    try:
        # Check if model exists first
        from pathlib import Path

        if not Path(model_path).exists():
            print(f"❌ Model not found at: {model_path}")
            print(f"")
            print(f"🚀 To train the model first, run:")
            print(f"   python train_from_collections.py")
            print(f"")
            print(f"💡 This will create the trained model that you can then use here.")
            return

        # Load model and generate parser
        generated_parser = trainer.load_and_test_model(
            model_path, test_config, test_argspec
        )

        print(f"✨ Generated Parser:")
        print(json.dumps(generated_parser, indent=2))

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure the model is trained and saved at: {model_path}")
        print(f"")
        print(
            f"🔧 If you're getting embedding size errors, the model may need retraining:"
        )
        print(f"   python train_from_collections.py")


if __name__ == "__main__":
    main()
