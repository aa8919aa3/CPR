# CPR Project Structure

This document describes the reorganized directory structure of the CPR (Current-Phase Relation) project.

## Overview

The project has been restructured to follow Python packaging best practices with clear separation of concerns:

```
CPR/
├── 📁 src/                     # Source code
├── 📁 tests/                   # All test files
├── 📁 config/                  # Configuration files
├── 📁 logs/                    # Log files
├── 📁 output/                  # Processing outputs
├── 📁 debug/                   # Debug tools and outputs
├── 📁 data/                    # Input data files
├── 📁 docs/                    # Documentation
├── 📁 examples/                # Example usage
├── 📁 scripts/                 # Utility scripts
└── 📄 Project files            # Setup, requirements, etc.
```

## Directory Details

### 📁 src/
Main source code directory containing the CPR processing modules:
- `cpr/` - Core CPR processing package
  - `main_processor_optimized.py` - Optimized processor with multithreading
  - `config.py` - Configuration management
  - `logger.py` - Logging utilities
  - `josephson_model.py` - Josephson junction models
  - `visualization.py` - Plotting and visualization
  - `analysis_utils.py` - Analysis utilities
  - `memory_manager.py` - Memory management

### 📁 tests/
Organized test suite with clear categories:
- `unit/` - Unit tests for individual components
  - `test_exact_image_size.py` - Image dimension validation
  - `test_image_size.py` - Image size testing
  - `test_optimized.py` - Optimized processor tests
  - `test_process_files.py` - File processing tests
- `integration/` - Integration and system tests
  - `final_integration_test.py` - Complete system integration test
  - `improved_thread_safety_test.py` - Thread safety validation
- `performance/` - Performance and benchmarking tests
  - `final_optimization_test.py` - Performance optimization tests

### 📁 config/
Configuration management:
- `config.json` - Main configuration file with updated paths
- `README.md` - Configuration documentation

### 📁 logs/
Centralized logging:
- `processing.log` - Main processing log file
- `README.md` - Logging documentation

### 📁 output/
Organized output structure:
- `images/` - Generated visualization images (~1699 files)
- `benchmark/` - Benchmark test outputs
- `test/` - Test run outputs
- `data/` - Data analysis results
- `temp/` - Temporary processing files

### 📁 debug/
Debug tools and outputs:
- `scripts/` - Debug scripts
  - `debug_failures.py` - Failure analysis tools
  - `debug_power_spectrum.py` - Power spectrum debugging
- `output/` - Debug output files
- `README.md` - Debug documentation

### 📁 data/
Input data files:
- `Ic/` - Critical current data files (CSV format)

### 📁 docs/
Project documentation:
- `CPR_CONCEPT.md` - Conceptual documentation

### 📁 examples/
Usage examples:
- `example_usage.py` - Example usage script

### 📁 scripts/
Utility scripts:
- `run_analysis.py` - Analysis runner script

## Key Improvements

### 1. **Clean Root Directory**
- Removed scattered test and debug files from root
- Kept only essential project files (setup.py, requirements.txt, etc.)

### 2. **Organized Testing**
- Separated unit, integration, and performance tests
- Added proper `__init__.py` files for Python package structure

### 3. **Centralized Configuration**
- Moved config.json to dedicated config/ directory
- Updated paths in configuration to match new structure

### 4. **Structured Outputs**
- Organized output files by type (images, data, benchmarks)
- Maintained all existing output files in logical locations

### 5. **Debug Organization**
- Separated debug scripts from debug outputs
- Centralized troubleshooting tools

## Migration Summary

The following files were moved during reorganization:

**Tests → tests/**
- `test_*.py` → `tests/unit/`
- `final_integration_test.py` → `tests/integration/`
- `improved_thread_safety_test.py` → `tests/integration/`
- `final_optimization_test.py` → `tests/performance/`

**Debug → debug/**
- `debug_*.py` → `debug/scripts/`
- `debug_output/` → `debug/output/`

**Configuration → config/**
- `config.json` → `config/`

**Logs → logs/**
- `processing.log` → `logs/`

**Outputs → output/**
- `output/` → `output/images/`
- `output_benchmark/` → `output/benchmark/`
- `output_test/` → `output/test/`

**Scripts → examples/ and scripts/**
- `example_usage.py` → `examples/`
- `run_analysis.py` → `scripts/`

## Configuration Updates

Updated `config/config.json` paths:
- `"OUTPUT_FOLDER": "output/images"`
- `"SUMMARY_FILE": "output/data/analysis_summary.csv"`
- `"LOG_FILE": "logs/processing.log"`

## Benefits

1. **Maintainability**: Clear separation of concerns makes code easier to maintain
2. **Scalability**: Organized structure supports project growth
3. **Standards Compliance**: Follows Python packaging best practices
4. **Developer Experience**: Easier to navigate和理解project structure
5. **Testing**: Clear test organization supports better CI/CD practices

## Usage

After reorganization, the project maintains full functionality:
- All optimizations remain intact (threading, numba, FireDucks)
- Image processing continues to generate 1920x1080 images
- Performance improvements are preserved
- All test files work with new structure

The main processor can still be run from the project root:
```bash
python -m src.cpr.main_processor_optimized
```

Tests can be run from their new locations:
```bash
python tests/integration/final_integration_test.py
python tests/performance/final_optimization_test.py
```
