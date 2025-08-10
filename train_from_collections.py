#!/usr/bin/env python3
"""
Simple script to train the model using your Ansible collections.

This script provides an easy way to train the parser AI using your collection structure.
"""

import yaml
import argparse
import sys
from pathlib import Path
from src.collection_trainer import CollectionBasedTrainer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train parser AI from Ansible collections"
    )
    parser.add_argument(
        "--config",
        default="collection_config.yaml",
        help="Path to configuration file (default: collection_config.yaml)",
    )
    parser.add_argument("--model-name", help="Override model name from config")
    parser.add_argument("--collections-path", help="Override collections base path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually training",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please create a collection_config.yaml file or specify --config")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.collections_path:
        config["collection_base_path"] = args.collections_path
    if args.model_name:
        config["model"]["name"] = args.model_name

    print("🎯 Collection-Based Parser AI Training")
    print("=" * 50)

    # Print configuration
    collections_base = Path(config["collection_base_path"]).resolve()
    model_storage = Path(config["model"]["storage_path"]).resolve()

    print(f"📁 Collections base path: {collections_base}")
    print(f"🏠 Model storage path: {model_storage}")
    print(f"🤖 Model name: {config['model']['name']}")

    # Get enabled collections
    enabled_collections = [c for c in config["collections"] if c.get("enabled", True)]
    print(f"📚 Enabled collections: {len(enabled_collections)}")

    for collection in enabled_collections:
        print(f"  - {collection['name']}")
        argspec_full_path = (
            collections_base / "ansible_collections" / collection["argspec_path"]
        )
        rm_template_full_path = (
            collections_base / "ansible_collections" / collection["rm_template_path"]
        )
        print(f"    Argspec: {argspec_full_path}")
        print(f"    Templates: {rm_template_full_path}")

        # Check if paths exist
        if not argspec_full_path.exists():
            print(f"    ⚠️  Argspec path does not exist!")
        if not rm_template_full_path.exists():
            print(f"    ⚠️  RM template path does not exist!")

    if args.dry_run:
        print("\n🔍 Dry run completed. Use --help for more options.")
        return

    # Initialize trainer
    trainer = CollectionBasedTrainer(config["collection_base_path"])

    # Prepare paths
    argspec_paths = [c["argspec_path"] for c in enabled_collections]
    rm_template_paths = [c["rm_template_path"] for c in enabled_collections]

    print(f"\n🚀 Starting training with {len(enabled_collections)} collections...")

    try:
        # Train the model
        model_path = trainer.train_from_collections(
            argspec_paths=argspec_paths,
            rm_template_paths=rm_template_paths,
            model_name=config["model"]["name"],
        )

        print(f"\n✅ Training completed successfully!")
        print(f"📍 Model saved at: {model_path}")
        print(f"\n💡 To use the trained model:")
        print(f"   from src.collection_trainer import CollectionBasedTrainer")
        print(f"   trainer = CollectionBasedTrainer()")
        print(
            f"   trainer.load_and_test_model('{model_path}', 'your config', your_argspec)"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        print(
            f"💡 Check your collection paths and ensure they contain valid parser files"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
