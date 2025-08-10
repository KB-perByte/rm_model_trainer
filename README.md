# Network Configuration Parser AI Trainer

**\*** Cursor has been used to Generate Readmes and structure code.

🤖 An AI-powered system that learns from Ansible collection parsers to generate regex-based configuration parsers for network devices.

## 🎯 What It Does

- **Learns from existing parsers**: Trains on your Ansible collection's resource module templates
- **Generates new parsers**: Creates regex patterns and Jinja2 templates for new configurations
- **Validates patterns**: Suggests improvements for existing parsers
- **Handles multiple vendors**: Works with any Ansible network collection structure

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Activate your ML environment (if using pyenv)
pyenv activate ai-test
```

### 2. Configure Collections

Edit `collection_config.yaml` to point to your Ansible collections:

```yaml
collection_base_path: '/path/to/your/ansible_collections'

collections:
  - name: 'cisco_ios'
    argspec_path: 'cisco/ios/plugins/module_utils/network/ios/argspec'
    rm_template_path: 'cisco/ios/plugins/module_utils/network/ios/rm_templates'
    enabled: true
```

### 3. Train the Model

```bash
# Dry run to verify setup
python train_from_collections.py --dry-run

# Train the model
python train_from_collections.py
```

### 4. Use the Trained Model

```python
from src.collection_trainer import CollectionBasedTrainer

trainer = CollectionBasedTrainer()
parser = trainer.load_and_test_model(
    "./trained_models/multi_vendor_parser_model",
    "bgp additional-paths install receive",
    your_argspec_dict
)
```

## 📁 Project Structure

```
rm_model_trainer/
├── src/                          # Core source code
│   ├── trainer.py               # Main AI trainer class
│   ├── collection_trainer.py   # Collection-specific trainer
│   └── data_prep_utils.py      # Data preparation utilities
├── examples/                    # Usage examples
│   ├── use_trained_model.py    # Simple model usage
│   ├── api_usage_example.py    # Advanced API examples
│   └── example_usage.py        # Detailed examples
├── trained_models/             # Saved models (created after training)
├── collection_config.yaml      # Collection configuration
├── train_from_collections.py   # Main training script
└── requirements.txt            # Python dependencies
```

## 🔧 Configuration

### Collection Configuration (`collection_config.yaml`)

```yaml
# Base path to your Ansible collections
collection_base_path: '/home/user/ansible_collections'

# Model settings
model:
  name: 'multi_vendor_parser_model'
  storage_path: './trained_models'

# Collections to train on
collections:
  - name: 'cisco_ios'
    argspec_path: 'cisco/ios/plugins/module_utils/network/ios/argspec'
    rm_template_path: 'cisco/ios/plugins/module_utils/network/ios/rm_templates'
    enabled: true

  - name: 'cisco_nxos'
    argspec_path: 'cisco/nxos/plugins/module_utils/network/nxos/argspec'
    rm_template_path: 'cisco/nxos/plugins/module_utils/network/nxos/rm_templates'
    enabled: false # Disable for now

# Training parameters
training:
  batch_size: 4
  epochs: 10
  validation_split: 0.2
```

## 💡 Usage Examples

### Basic Usage

```python
from src.collection_trainer import CollectionBasedTrainer

# Initialize trainer
trainer = CollectionBasedTrainer()

# Generate parser for new config
config = "interface GigabitEthernet0/1 ip address 192.168.1.1 255.255.255.0"
argspec = {
    "interface": {
        "type": "dict",
        "options": {
            "name": {"type": "str"},
            "ip_address": {"type": "str"}
        }
    }
}

parser = trainer.load_and_test_model(
    "./trained_models/multi_vendor_parser_model",
    config,
    argspec
)
```

### Advanced API Usage

```python
from src.trainer import NetworkConfigParserAI

# Direct API access
parser_ai = NetworkConfigParserAI()
parser_ai.load_model("./trained_models/multi_vendor_parser_model")

# Generate parser
suggested_parser = parser_ai.generate_parser([config], argspec)

# Get improvement suggestions
suggestions = parser_ai.suggest_parser_improvements(existing_parser, config_lines)
```

## 🛠 Training Process

1. **Data Loading**: Extracts argspecs and parser templates from Ansible collections
2. **Data Preparation**: Creates training examples from existing parsers
3. **Model Training**: Fine-tunes a CodeBERT-based model on your data
4. **Model Saving**: Saves the trained model for future use

### Training Features

- ✅ **No Ansible Dependencies**: Parses collection files directly without importing Ansible
- ✅ **Automatic Path Detection**: Finds argspecs and templates in collection structure
- ✅ **Multiple Collections**: Train on multiple vendor collections simultaneously
- ✅ **Progress Tracking**: Real-time training progress and metrics
- ✅ **Model Versioning**: Saves training metadata and model checkpoints

## 🎯 Use Cases

### 1. **New Device Support**

When adding support for a new network device, generate initial parsers:

```python
config = "spanning-tree vlan 100 priority 4096"
# AI suggests regex patterns and Jinja2 templates
```

### 2. **Parser Validation**

Check if existing parsers handle new configuration variations:

```python
parser_ai.suggest_parser_improvements(existing_parser, new_config_examples)
```

### 3. **Configuration Analysis**

Understand structure of unknown network configurations:

```python
# Feed unknown configs, get structured parsing suggestions
```

## 🔍 Troubleshooting

### Common Issues

**Path Not Found Errors**

- Verify `collection_base_path` in `collection_config.yaml`
- Ensure collections are properly installed
- Check argspec and rm_template paths are correct

**Import Errors**

- Run from project root directory
- Ensure `src/` is in Python path
- Check all dependencies are installed

**Training Failures**

- Verify PyTorch and transformers versions
- Check available GPU/CPU memory
- Reduce batch size if out of memory

### Debugging

```bash
# Verify collection paths
python train_from_collections.py --dry-run

# Check what data is loaded
python -c "from src.collection_trainer import CollectionDataLoader; loader = CollectionDataLoader(); print(loader.load_argspec_from_path('your/path'))"
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20+
- scikit-learn
- pandas
- PyYAML

See `requirements.txt` for complete list.

## 🚫 No External Data Sync - Privacy First

**This system is completely self-contained and sends NO data to external services.**

- ✅ **No wandb** - All training metrics stay local
- ✅ **No tensorboard remote sync** - Only local files
- ✅ **No cloud uploads** - Everything saved locally
- ✅ **Privacy focused** - Your data never leaves your machine

### Local Logging Only

Training logs are saved locally to:

- Console output for real-time progress
- `./trained_models/[model_name]/logs/` for detailed logs
- `./trained_models/[model_name]/training_metadata.json` for training info

### Extra Privacy Assurance

If you want to be extra sure wandb is disabled:

```bash
# Optional: Run this before training for extra assurance
python disable_wandb.py

# Then train normally
python train_from_collections.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and examples
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Built on Hugging Face Transformers
- Uses Microsoft CodeBERT as base model
- Designed for Ansible network collections
