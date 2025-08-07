#!/usr/bin/env python3
"""
Test script to verify sequential loading logic works correctly.
This simulates the distributed loading without actually loading models.
"""

import os
import time
from accelerate import PartialState

def is_distributed_environment():
    """Check if we're running in a distributed environment with multiple processes."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    return world_size > 1

def simulate_model_loading(process_index, delay=2):
    """Simulate model loading with a delay."""
    print(f"[Process {process_index}] Starting model loading...")
    time.sleep(delay)  # Simulate loading time
    print(f"[Process {process_index}] Model loading completed!")

def main():
    print("Testing sequential loading logic...")
    
    # Handle distributed vs non-distributed execution
    if 'WORLD_SIZE' in os.environ:
        # Running with accelerate launch
        distributed_state = PartialState()
        process_index = distributed_state.process_index
        num_processes = distributed_state.num_processes
        use_distributed = is_distributed_environment()
        
        print(f"Process {process_index}: WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        print(f"Process {process_index}: num_processes={num_processes}")
        print(f"Process {process_index}: use_distributed={use_distributed}")
        
        if use_distributed:
            # Sequential loading: each process loads one at a time
            print(f"Process {process_index}: Using sequential loading")
            for rank in range(num_processes):
                if rank == process_index:
                    simulate_model_loading(process_index)
                
                # Wait for current process to finish before next one starts
                print(f"Process {process_index}: Waiting for everyone at rank {rank}")
                distributed_state.wait_for_everyone()
                print(f"Process {process_index}: Everyone ready after rank {rank}")
        else:
            # Single process with accelerate
            print(f"Process {process_index}: Single process mode")
            simulate_model_loading(process_index)
    else:
        # Running without accelerate
        print("Running without accelerate")
        simulate_model_loading(0)
    
    print("Test completed!")

if __name__ == "__main__":
    main()