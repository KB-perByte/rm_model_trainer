"""
Network Configuration Parser AI Training System

A system for training AI models to generate regex-based parsers for network configuration lines.
"""

from .trainer import NetworkConfigParserAI
from .collection_trainer import CollectionBasedTrainer, CollectionDataLoader

__version__ = "1.0.0"
__all__ = ["NetworkConfigParserAI", "CollectionBasedTrainer", "CollectionDataLoader"]