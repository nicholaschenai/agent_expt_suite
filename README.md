# Agent Experiment Suite

A framework for running experiments with AI agents, currently focused on code generation tasks.

## Overview

This suite provides tools and infrastructure for:

- Running experiments with AI agents on coding tasks
- Managing datasets and data pipelines 
- Executing and evaluating generated code safely
- Training and testing agents with various configurations

## Key Components

### Data Tools
- Dataset management for coding tasks (currently supports MBPP Plus)
- Configurable data pipelines with curriculum options
- Support for both PyTorch and Hugging Face datasets

### Environment
- Safe code execution environment with security measures
- Support for public and private test cases
- Configurable test display and timeout settings
- Modular environment registry for different datasets

### Evaluation
- Batch or per-step evaluation options
- Support for parallel evaluation with multiple agents (WIP)
- Checkpoint management and experiment tracking
- Customizable evaluation metrics (WIP)

## Usage

[TODO: Add basic usage examples and setup instructions]

## Supported Datasets

Currently supported datasets:
- MBPP Plus

## Configuration

Key configuration options include:

- Dataset selection and configuration
- Training parameters (iterations, batch size, etc.)
- Evaluation settings (public/private tests, display options)
- Checkpoint and saving preferences
- Debug and development options

See `eval_setup/argparsers.py` for full configuration options.

## Security

The suite implements several security measures for safe code execution:
- Restricted Python environment
- Whitelisted modules and built-ins
- Timeout controls
- Memory and resource limitations

## Contributing

[TODO: Add contribution guidelines]

## License

[TODO: Add license information]
