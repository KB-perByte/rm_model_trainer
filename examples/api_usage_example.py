#!/usr/bin/env python3
"""
Example of direct API usage for the trained parser model.
"""

from src.trainer import NetworkConfigParserAI
import json


def example_direct_usage():
    """Example of using the NetworkConfigParserAI directly."""
    
    # Initialize the AI system
    parser_ai = NetworkConfigParserAI()
    
    # Load your trained model
    model_path = "./trained_models/multi_vendor_parser_model"
    
    try:
        print("🔄 Loading trained model...")
        parser_ai.load_model(model_path)
        
        # Example configuration lines
        test_configs = [
            "bgp additional-paths send",
            "interface GigabitEthernet0/1 ip address 192.168.1.1 255.255.255.0",
            "vlan 100 name production"
        ]
        
        for config_line in test_configs:
            print(f"\n🔍 Analyzing: {config_line}")
            
            # You'd need to provide appropriate argspecs for each
            # This is a simplified example
            argspec = {
                "config": {
                    "type": "dict",
                    "options": {
                        "name": {"type": "str"},
                        "value": {"type": "str"}
                    }
                }
            }
            
            # Generate parser
            suggested_parser = parser_ai.generate_parser([config_line], argspec)
            print(f"📝 Suggested parser: {json.dumps(suggested_parser, indent=2)}")
            
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("💡 Train the model first using: python train_from_collections.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_parser_improvement():
    """Example of getting suggestions for improving existing parsers."""
    
    parser_ai = NetworkConfigParserAI()
    
    # Example existing parser
    existing_parser = {
        "name": "bgp.additional_paths",
        "getval": r"\s+bgp\s+additional-paths\s+(?P<action>\w+)",
        "setval": "bgp additional-paths {{ action }}",
        "result": {"bgp": {"additional_paths": {"action": "{{ action }}"}}}
    }
    
    config_lines = [
        "bgp additional-paths install",
        "bgp additional-paths receive", 
        "bgp additional-paths send"
    ]
    
    print("\n🔧 Getting parser improvement suggestions...")
    suggestions = parser_ai.suggest_parser_improvements(existing_parser, config_lines)
    
    if suggestions:
        print("💡 Suggestions:")
        for suggestion in suggestions:
            print(f"   - {suggestion}")
    else:
        print("✅ Parser looks good!")


if __name__ == "__main__":
    print("🎯 Direct API Usage Examples")
    print("=" * 50)
    
    example_direct_usage()
    example_parser_improvement()