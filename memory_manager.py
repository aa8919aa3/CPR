"""
Memory management and adaptive processing
"""
import psutil
import numpy as np
import gc
from typing import Tuple, List
import threading
import time

class MemoryManager:
    """Smart memory management with adaptive processing"""
    
    def __init__(self, config):
        self.config = config
        self.memory_threshold = config.get('MEMORY_THRESHOLD', 85)
        self.lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread = None
        
    def get_memory_info(self) -> dict:
        """Get current memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        return self.get_memory_info()['percent'] > self.memory_threshold
    
    def get_optimal_workers(self, file_sizes: List[int] = None) -> int:
        """Calculate optimal number of workers based on memory and file sizes"""
        memory_info = self.get_memory_info()
        max_workers = self.config.get('MAX_WORKERS', 8)
        
        # Base adjustment on available memory
        if memory_info['available_gb'] < 2:
            workers = 2
        elif memory_info['available_gb'] < 4:
            workers = min(4, max_workers)
        elif memory_info['available_gb'] < 8:
            workers = min(6, max_workers)
        else:
            workers = max_workers
        
        # Adjust based on file sizes if provided
        if file_sizes:
            avg_size_mb = np.mean(file_sizes) / (1024 * 1024)
            if avg_size_mb > 10:  # Large files
                workers = max(1, workers // 2)
            elif avg_size_mb > 5:  # Medium files
                workers = max(2, int(workers * 0.75))
        
        return max(1, workers)
    
    def cleanup_memory(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        
    def start_monitoring(self, callback=None):
        """Start memory monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory, 
            args=(callback,), 
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_memory(self, callback=None):
        """Background memory monitoring"""
        while self.monitoring:
            try:
                if self.is_memory_critical():
                    if callback:
                        callback("Memory usage critical!")
                    self.cleanup_memory()
                time.sleep(5)  # Check every 5 seconds
            except Exception:
                break

class AdaptiveProcessor:
    """Adaptive processing with dynamic resource management"""
    
    def __init__(self, config):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.processing_times = []
        self.error_count = 0
        
    def estimate_processing_time(self, file_count: int) -> dict:
        """Estimate processing time based on historical data"""
        if not self.processing_times:
            # Default estimates
            avg_time_per_file = 2.0  # seconds
        else:
            avg_time_per_file = np.mean(self.processing_times)
        
        workers = self.memory_manager.get_optimal_workers()
        estimated_parallel_time = (file_count * avg_time_per_file) / workers
        
        return {
            'total_files': file_count,
            'avg_time_per_file': avg_time_per_file,
            'optimal_workers': workers,
            'estimated_time_sequential': file_count * avg_time_per_file,
            'estimated_time_parallel': estimated_parallel_time,
            'speedup_factor': file_count * avg_time_per_file / estimated_parallel_time
        }
    
    def should_use_chunking(self, file_count: int) -> bool:
        """Determine if chunking should be used"""
        memory_info = self.memory_manager.get_memory_info()
        chunk_size = self.config.get('CHUNK_SIZE', 100)
        
        return (file_count > chunk_size and 
                memory_info['available_gb'] < 4)
    
    def get_chunk_size(self, file_count: int) -> int:
        """Calculate optimal chunk size"""
        memory_info = self.memory_manager.get_memory_info()
        base_chunk_size = self.config.get('CHUNK_SIZE', 100)
        
        if memory_info['available_gb'] < 2:
            return min(25, base_chunk_size)
        elif memory_info['available_gb'] < 4:
            return min(50, base_chunk_size)
        else:
            return base_chunk_size
    
    def record_processing_time(self, time_taken: float):
        """Record processing time for future estimates"""
        self.processing_times.append(time_taken)
        # Keep only last 100 measurements
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def record_error(self):
        """Record processing error"""
        self.error_count += 1
    
    def should_reduce_workers(self, current_workers: int) -> int:
        """Determine if workers should be reduced due to errors"""
        if self.error_count > current_workers * 2:
            return max(1, current_workers // 2)
        return current_workers