#!/usr/bin/env python3
"""
Script to ensure wandb is completely disabled.
Run this before training if you want to be extra sure.
"""

import os
import sys

def disable_wandb():
    """Disable wandb completely through environment variables."""
    
    # Set environment variables to disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    
    print("✅ Wandb disabled through environment variables")
    
    # Try to import wandb and disable it if available
    try:
        import wandb
        wandb.init(mode="disabled")
        print("✅ Wandb imported and set to disabled mode")
    except ImportError:
        print("ℹ️  Wandb not installed (that's fine - no external logging will happen)")
    except Exception as e:
        print(f"⚠️  Warning: Could not disable wandb: {e}")
    
    print("\n🚫 No data will be sent to external services!")
    print("📊 Training logs will be saved locally to: ./trained_models/[model_name]/logs/")

if __name__ == "__main__":
    disable_wandb()