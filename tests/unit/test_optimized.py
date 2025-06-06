#!/usr/bin/env python3
"""
Test script for the optimized CPR (Current-Phase Relation) analysis system
"""
import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor
from cpr.config import config

def test_optimized_analysis():
    """Test the optimized analysis with a small subset of data"""
    print("=== Testing Optimized CPR Analysis ===")
    print(f"Configuration:")
    print(f"  - Figure size: {config.get('FIGURE_SIZE')}")
    print(f"  - DPI: {config.get('DPI_HIGH')}")
    print(f"  - Max workers: {config.get('MAX_WORKERS')}")
    print(f"  - Input folder: {config.get('INPUT_FOLDER')}")
    print(f"  - Output folder: {config.get('OUTPUT_FOLDER')}")
    
    # Initialize the processor
    processor = EnhancedJosephsonProcessor()
    
    # Get a small subset of files (first 10)
    input_folder = Path(config.get('INPUT_FOLDER', 'data/Ic'))
    csv_files = list(input_folder.glob("*.csv"))[:10]  # Test with first 10 files
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"\nTesting with {len(csv_files)} files:")
    for f in csv_files:
        # Get data point count
        try:
            import pandas as pd
            df = pd.read_csv(f)
            data_points = len(df)
            print(f"  - {f.name}: {data_points} data points")
        except Exception as e:
            print(f"  - {f.name}: Error reading file")
    
    # Run the analysis on individual files
    start_time = time.time()
    results = []
    output_folder = config.get('OUTPUT_FOLDER', 'output')
    os.makedirs(output_folder, exist_ok=True)
    
    for csv_file in csv_files:
        result = processor.process_single_file(str(csv_file), output_folder)
        results.append(result)
        # Print data point info
        if result.get('success'):
            print(f"INFO: Processed {result['dataid']}: analysis completed successfully")
        else:
            print(f"ERROR: Failed {result['dataid']}: {result.get('error', 'Unknown error')}")
    end_time = time.time()
    
    # Print results
    print(f"\n=== Analysis Results ===")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Files processed: {len(results)}")
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccess rate: {len(successful)/len(results)*100:.1f}%")
        
        # Show first successful result
        first_success = successful[0]
        print(f"\nSample result from {first_success['dataid']}:")
        print(f"  I_c: {first_success.get('I_c', 'N/A'):.3e}")
        print(f"  f: {first_success.get('f', 'N/A'):.3e}")
        print(f"  φ_0: {first_success.get('phi_0', 'N/A'):.3f}")
        print(f"  T: {first_success.get('T', 'N/A'):.1%}")
        print(f"  R²: {first_success.get('r_squared', 'N/A'):.4f}")
        print(f"  RMSE: {first_success.get('rmse', 'N/A'):.4f}")
    
    if failed:
        print(f"\nSample failures:")
        for i, failure in enumerate(failed[:3]):  # Show first 3 failures
            print(f"  {failure['dataid']}: {failure.get('error', 'Unknown error')}")
    
    # Check output files
    output_folder = Path(config.get('OUTPUT_FOLDER', 'output'))
    if output_folder.exists():
        plot_files = list(output_folder.glob("*.png"))
        if plot_files:
            print(f"\n=== Generated Plots ===")
            print(f"Found {len(plot_files)} PNG files in output folder")
            
            # Check image dimensions of first plot
            try:
                from PIL import Image
                first_plot = plot_files[0]
                with Image.open(first_plot) as img:
                    width, height = img.size
                    print(f"Sample plot '{first_plot.name}': {width}x{height} pixels")
                    if width == 1920 and height == 1080:
                        print("✓ Correct dimensions (1920x1080)")
                    else:
                        print(f"⚠ Expected 1920x1080, got {width}x{height}")
            except ImportError:
                print(f"PIL not available - cannot check image dimensions")
            except Exception as e:
                print(f"Error checking image: {e}")
        else:
            print("No plot files found in output folder")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_optimized_analysis()
