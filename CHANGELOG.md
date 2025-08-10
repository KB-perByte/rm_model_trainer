# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-01-XX

### Added
- Initial release of Network Configuration Parser AI Trainer
- Support for Ansible collection argspec and rm_template parsing
- AI model training using CodeBERT base model
- Collection-based training pipeline
- Example scripts for model usage
- Comprehensive documentation

### Features
- ✅ Load argspecs from Ansible collection directories
- ✅ Extract parser templates from rm_templates without Ansible dependencies
- ✅ Train transformer models for parser generation
- ✅ Generate regex patterns and Jinja2 templates for new configurations
- ✅ Validate and improve existing parsers
- ✅ Support multiple vendor collections simultaneously
- ✅ No external dependencies like wandb required

### Technical
- Built on PyTorch and Hugging Face Transformers
- Uses Microsoft CodeBERT as base model
- Text-based parsing to avoid Ansible import dependencies
- Modular architecture with clean separation of concerns