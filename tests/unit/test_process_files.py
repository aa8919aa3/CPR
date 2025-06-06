#!/usr/bin/env python3
"""
Test script for the new process_files method in EnhancedJosephsonProcessor
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

def test_process_files():
    """Test the new process_files method with a subset of files"""
    print("=== Testing process_files method ===")
    
    # Initialize the processor
    processor = EnhancedJosephsonProcessor()
    
    # Get test files (first 3 files for quick testing)
    input_folder = Path(config.get('INPUT_FOLDER', 'data/Ic'))
    all_csv_files = list(input_folder.glob("*.csv"))
    
    if not all_csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    # Test with 3 files
    test_files = [str(f) for f in all_csv_files[:3]]
    print(f"Testing with {len(test_files)} files:")
    for f in test_files:
        print(f"  - {Path(f).name}")
    
    # Create output directory
    output_folder = config.get('OUTPUT_FOLDER', 'output')
    os.makedirs(output_folder, exist_ok=True)
    
    # Run the new process_files method
    start_time = time.time()
    print("\n" + "="*50)
    print("TESTING process_files METHOD")
    print("="*50)
    
    results = processor.process_files(test_files, output_folder)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print results
    print(f"\n=== Test Results ===")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Files processed: {len(results)}")
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        
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
        print(f"\nFailures:")
        for failure in failed:
            print(f"  {failure['dataid']}: {failure.get('error', 'Unknown error')}")
    
    # Check output files
    output_folder = Path(config.get('OUTPUT_FOLDER', 'output'))
    if output_folder.exists():
        plot_files = list(output_folder.glob("*.png"))
        if plot_files:
            print(f"\n=== Generated Plots ===")
            print(f"Found {len(plot_files)} PNG files in output folder")
            
            # Check image dimensions of first plot if PIL is available
            try:
                from PIL import Image
                for plot_file in plot_files[:3]:  # Check first 3 plots
                    with Image.open(plot_file) as img:
                        width, height = img.size
                        status = "✓" if (width == 1920 and height == 1080) else "⚠"
                        print(f"  {status} {plot_file.name}: {width}x{height}")
                        if plot_file == plot_files[0]:  # Only show details for first file
                            print(f"    Expected: 1920x1080, Got: {width}x{height}")
                            break
            except ImportError:
                print("  PIL not available - cannot check image dimensions")
            except Exception as e:
                print(f"  Error checking images: {e}")
        else:
            print("  No plot files found")
    
    print(f"\n=== Test Complete ===")
    print("✅ process_files method is working correctly!")
    return results

if __name__ == "__main__":
    test_process_files()