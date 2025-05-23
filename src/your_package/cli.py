"""
Command-line interface for the your_package module.

This module provides a simple CLI interface for the package functionality,
demonstrating how to structure command-line tools in Python packages.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .core import YourMainClass, YourSecondaryClass
from .utils import (
    format_data_summary,
    load_config_from_file,
    save_config_to_file,
    utility_function,
    validate_config,
)

# Set up logging
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable debug logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def process_command(
    data: List[str], 
    transform: str = "upper",
    name: str = "cli-processor",
    config_file: Optional[str] = None
) -> None:
    """
    Process data using the main class functionality.
    
    Args:
        data: List of strings to process.
        transform: Transformation to apply.
        name: Name for the processor instance.
        config_file: Optional configuration file path.
    """
    # Load configuration if provided
    config = {}
    if config_file:
        loaded_config = load_config_from_file(config_file)
        if loaded_config:
            config = loaded_config
            logger.info(f"Loaded configuration from {config_file}")
        else:
            logger.error(f"Failed to load configuration from {config_file}")
            return
    
    # Create processor instance
    try:
        processor = YourMainClass(name, config)
        result = processor.process_data(data, transform)
        
        print("Processing Results:")
        print("-" * 40)
        for i, (original, processed) in enumerate(zip(data, result), 1):
            print(f"{i:2d}. {original:15} -> {processed}")
        
        # Show instance info
        print("\nProcessor Info:")
        info = processor.get_info()
        print(f"Name: {info['name']}")
        print(f"Config: {info['config']}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


def extended_process_command(
    data: List[str],
    name: str = "cli-processor",
    config_file: Optional[str] = None
) -> None:
    """
    Process data using extended functionality.
    
    Args:
        data: List of strings to process.
        name: Name for the processor instance.
        config_file: Optional configuration file path.
    """
    # Load configuration if provided
    config = {}
    if config_file:
        loaded_config = load_config_from_file(config_file)
        if loaded_config:
            config = loaded_config
    
    try:
        # Create instances
        main_processor = YourMainClass(name, config)
        extended_processor = YourSecondaryClass(main_processor)
        
        # Process data
        results = extended_processor.extended_process(data)
        
        print("Extended Processing Results:")
        print("=" * 50)
        
        for transform, transformed_data in results.items():
            print(f"\n{transform.upper()} transformation:")
            print("-" * 30)
            for original, processed in zip(data, transformed_data):
                print(f"  {original:15} -> {processed}")
                
    except Exception as e:
        logger.error(f"Extended processing failed: {e}")
        sys.exit(1)


def utility_command(
    value: str,
    operation: str = "stringify"
) -> None:
    """
    Use utility functions from the command line.
    
    Args:
        value: Value to process.
        operation: Operation to perform.
    """
    try:
        # Try to convert to number if possible
        if value.isdigit():
            parsed_value = int(value)
        else:
            try:
                parsed_value = float(value)
            except ValueError:
                parsed_value = value
        
        result = utility_function(parsed_value, operation)
        print(f"Utility result: {result}")
        
    except Exception as e:
        logger.error(f"Utility operation failed: {e}")
        sys.exit(1)


def analyze_command(data: List[str]) -> None:
    """
    Analyze data and show summary.
    
    Args:
        data: List of data items to analyze.
    """
    try:
        # Convert numeric strings to numbers for analysis
        processed_data = []
        for item in data:
            if item.isdigit():
                processed_data.append(int(item))
            else:
                try:
                    processed_data.append(float(item))
                except ValueError:
                    processed_data.append(item)
        
        summary = format_data_summary(processed_data)
        print("Data Analysis:")
        print("=" * 30)
        print(summary)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def config_command(
    action: str,
    config_file: str,
    name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Manage configuration files.
    
    Args:
        action: Action to perform (create, validate, show).
        config_file: Path to configuration file.
        name: Name for the configuration.
        **kwargs: Additional configuration parameters.
    """
    try:
        if action == "create":
            if not name:
                name = "default_config"
            
            config = {"name": name}
            config.update(kwargs)
            
            if save_config_to_file(config, config_file):
                print(f"Configuration saved to {config_file}")
            else:
                print("Failed to save configuration")
                sys.exit(1)
                
        elif action == "validate":
            config = load_config_from_file(config_file)
            if config is None:
                print("Failed to load configuration")
                sys.exit(1)
                
            if validate_config(config):
                print("Configuration is valid")
            else:
                print("Configuration is invalid")
                sys.exit(1)
                
        elif action == "show":
            config = load_config_from_file(config_file)
            if config is None:
                print("Failed to load configuration")
                sys.exit(1)
                
            print("Configuration:")
            print(json.dumps(config, indent=2))
            
        else:
            print(f"Unknown action: {action}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Configuration operation failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="your-package",
        description="CLI interface for your-package-name",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process hello world --transform title
  %(prog)s extended hello world --config config.json
  %(prog)s utility 3.14159 --operation format
  %(prog)s analyze 1 2 3 hello 3.14
  %(prog)s config create config.json --name my_config
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process data with transformations"
    )
    process_parser.add_argument("data", nargs="+", help="Data to process")
    process_parser.add_argument(
        "--transform", 
        choices=["upper", "lower", "title"],
        default="upper",
        help="Transformation to apply"
    )
    process_parser.add_argument("--name", default="cli-processor", help="Processor name")
    process_parser.add_argument("--config", help="Configuration file path")
    
    # Extended process command
    extended_parser = subparsers.add_parser(
        "extended",
        help="Process data with extended functionality"
    )
    extended_parser.add_argument("data", nargs="+", help="Data to process")
    extended_parser.add_argument("--name", default="cli-processor", help="Processor name")
    extended_parser.add_argument("--config", help="Configuration file path")
    
    # Utility command
    utility_parser = subparsers.add_parser(
        "utility",
        help="Use utility functions"
    )
    utility_parser.add_argument("value", help="Value to process")
    utility_parser.add_argument(
        "--operation",
        choices=["stringify", "format"],
        default="stringify",
        help="Operation to perform"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze data and show summary"
    )
    analyze_parser.add_argument("data", nargs="+", help="Data to analyze")
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration files"
    )
    config_parser.add_argument(
        "action",
        choices=["create", "validate", "show"],
        help="Configuration action"
    )
    config_parser.add_argument("config_file", help="Configuration file path")
    config_parser.add_argument("--name", help="Configuration name")
    config_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    config_parser.add_argument("--timeout", type=int, help="Timeout value")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle commands
    if args.command == "process":
        process_command(
            args.data,
            args.transform,
            args.name,
            args.config
        )
    elif args.command == "extended":
        extended_process_command(
            args.data,
            args.name,
            args.config
        )
    elif args.command == "utility":
        utility_command(args.value, args.operation)
    elif args.command == "analyze":
        analyze_command(args.data)
    elif args.command == "config":
        config_kwargs = {}
        if hasattr(args, 'debug') and args.debug:
            config_kwargs['debug'] = args.debug
        if hasattr(args, 'timeout') and args.timeout:
            config_kwargs['timeout'] = args.timeout
            
        config_command(
            args.action,
            args.config_file,
            args.name,
            **config_kwargs
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 