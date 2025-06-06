#!/usr/bin/env python3
"""
CPR - Current-Phase Relation
Main entry point for the Josephson junction analysis suite
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cpr.main_processor import EnhancedJosephsonProcessor

def main():
    """Main entry point"""
    processor = EnhancedJosephsonProcessor()
    processor.batch_process_files()

if __name__ == "__main__":
    main()
