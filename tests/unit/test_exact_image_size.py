#!/usr/bin/env python3
"""
Test script to achieve exact 1920x1080 pixel images with matplotlib
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def test_image_size_method(method_name, figsize, dpi, bbox_inches=None, pad_inches=None):
    """Test different methods to achieve exact 1920x1080 pixels"""
    
    # Create test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create plot
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x, y, 'b-', linewidth=2, label='Test Plot')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Test Plot - Method: {method_name}')
    plt.legend()
    plt.grid(True)
    
    # Save with different parameters
    filename = f'test_{method_name}.png'
    if bbox_inches is not None and pad_inches is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
    else:
        plt.savefig(filename, dpi=dpi)
    
    plt.close()
    
    # Check actual size
    if os.path.exists(filename):
        with Image.open(filename) as img:
            width, height = img.size
            print(f"{method_name}: {width}x{height} pixels (figsize={figsize}, dpi={dpi})")
            if width == 1920 and height == 1080:
                print(f"  ✅ EXACT SIZE ACHIEVED!")
            else:
                print(f"  ❌ Expected 1920x1080, got {width}x{height}")
        os.remove(filename)
    else:
        print(f"{method_name}: Failed to create file")

def main():
    print("Testing different methods to achieve exact 1920x1080 pixel images")
    print("="*70)
    
    # Method 1: Standard approach
    test_image_size_method("standard", (19.2, 10.8), 100)
    
    # Method 2: Tight bounding box
    test_image_size_method("tight_bbox", (19.2, 10.8), 100, bbox_inches='tight', pad_inches=0)
    
    # Method 3: Adjusted figure size accounting for margins
    test_image_size_method("adjusted_size", (20.0, 11.25), 100)
    
    # Method 4: Different DPI with adjusted size
    test_image_size_method("dpi_150", (12.8, 7.2), 150)
    
    # Method 5: Calculate exact size based on matplotlib defaults
    # matplotlib typically uses about 4% margins on each side
    margin_factor = 1.08  # 8% total margin
    adjusted_width = 19.2 * margin_factor
    adjusted_height = 10.8 * margin_factor
    test_image_size_method("margin_compensated", (adjusted_width, adjusted_height), 100)
    
    # Method 6: Using subplots_adjust to control margins
    def test_with_subplots_adjust():
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(19.2, 10.8), dpi=100)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.plot(x, y, 'b-', linewidth=2, label='Test Plot')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Test Plot - Subplots Adjust Method')
        plt.legend()
        plt.grid(True)
        
        filename = 'test_subplots_adjust.png'
        plt.savefig(filename, dpi=100)
        plt.close()
        
        if os.path.exists(filename):
            with Image.open(filename) as img:
                width, height = img.size
                print(f"subplots_adjust: {width}x{height} pixels")
                if width == 1920 and height == 1080:
                    print(f"  ✅ EXACT SIZE ACHIEVED!")
                else:
                    print(f"  ❌ Expected 1920x1080, got {width}x{height}")
            os.remove(filename)
    
    test_with_subplots_adjust()
    
    print("="*70)
    print("Testing complete. Look for the method that achieves exact 1920x1080 pixels.")

if __name__ == "__main__":
    main()
