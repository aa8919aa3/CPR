#!/usr/bin/env python3
"""
Test script to verify exact image size generation
"""
import os
import sys
from PIL import Image
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cpr.main_processor_optimized import EnhancedJosephsonProcessor

def test_image_size():
    """Test a single file to verify exact 1920x1080 image generation"""
    print("=== Testing Exact Image Size Generation ===")
    
    # Initialize processor
    processor = EnhancedJosephsonProcessor()
    
    # Test with one file
    test_file = "/Users/albert-mac/Code/GitHub/CPR/data/Ic/369Ic.csv"
    output_folder = "/Users/albert-mac/Code/GitHub/CPR/output"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"Processing test file: {Path(test_file).name}")
    
    # Process the file
    result = processor.process_single_file(test_file, output_folder)
    
    if result['success']:
        print(f"✅ Processing successful")
        
        # Check generated images
        dataid = result['dataid']
        image_files = [
            f"{dataid}_fitted_curve_normalized_plot.png",
            f"{dataid}_fitted_curve_plot.png", 
            f"{dataid}_residuals_plot.png",
            f"{dataid}_phase_folded_with_drift.png",
            f"{dataid}_cycles_colored_matplotlib.png"
        ]
        
        print("\n=== Image Size Analysis ===")
        for img_file in image_files:
            img_path = os.path.join(output_folder, img_file)
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        status = "✅" if (width == 1920 and height == 1080) else "❌"
                        print(f"{status} {img_file}: {width}x{height}")
                        if width != 1920 or height != 1080:
                            diff_w = width - 1920
                            diff_h = height - 1080
                            print(f"    Difference: {diff_w:+d}x{diff_h:+d} pixels")
                except Exception as e:
                    print(f"❌ Error reading {img_file}: {e}")
            else:
                print(f"❌ {img_file}: Not found")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_image_size()
