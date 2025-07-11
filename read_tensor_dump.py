#!/usr/bin/env python3
"""
Script to read and process tensor dump files created by TensorRT backend
when large values (>600) are detected.

Binary file format:
- int32: number of tensors
For each tensor:
- int32: name length
- char[]: tensor name
- int32: tensor type (1=input, 0=output)
- int32: data type (TRITONSERVER_DataType)
- int32: number of dimensions
- int32[]: dimension sizes
- int32: number of elements
- binary data: tensor data
"""

import struct
import numpy as np
import pickle
import argparse
import os
from typing import Dict, List, Tuple, Any

# TRITONSERVER_DataType mappings
TRITON_TYPES = {
    0: ('INVALID', None),
    1: ('BOOL', np.bool_),
    2: ('UINT8', np.uint8),
    3: ('UINT16', np.uint16),
    4: ('UINT32', np.uint32),
    5: ('UINT64', np.uint64),
    6: ('INT8', np.int8),
    7: ('INT16', np.int16),
    8: ('INT32', np.int32),
    9: ('INT64', np.int64),
    10: ('FP16', np.float16),
    11: ('FP32', np.float32),
    12: ('FP64', np.float64),
    13: ('BYTES', None),
    14: ('BF16', None)  # bfloat16 not directly supported in numpy
}

def read_tensor_dump(filepath: str) -> Dict[str, Any]:
    """
    Read a tensor dump file and return parsed tensor data.
    
    Args:
        filepath: Path to the binary tensor dump file
        
    Returns:
        Dictionary containing parsed tensor information
    """
    tensors = {}
    
    with open(filepath, 'rb') as f:
        # Read number of tensors
        num_tensors = struct.unpack('<i', f.read(4))[0]
        print(f"Reading {num_tensors} tensors from {filepath}")
        
        for i in range(num_tensors):
            # Read tensor name
            name_len = struct.unpack('<i', f.read(4))[0]
            tensor_name = f.read(name_len).decode('utf-8')
            
            # Read tensor metadata
            tensor_type = struct.unpack('<i', f.read(4))[0]  # 1=input, 0=output
            data_type_id = struct.unpack('<i', f.read(4))[0]
            num_dims = struct.unpack('<i', f.read(4))[0]
            
            # Read dimensions
            dims = []
            for _ in range(num_dims):
                dim_size = struct.unpack('<i', f.read(4))[0]
                dims.append(dim_size)
            
            # Read number of elements
            num_elements = struct.unpack('<i', f.read(4))[0]
            
            # Read buffer byte size (actual size of tensor data in file)
            buffer_byte_size = struct.unpack('<Q', f.read(8))[0]  # size_t is 8 bytes (uint64)
            
            # Get data type info
            type_name, np_dtype = TRITON_TYPES.get(data_type_id, ('UNKNOWN', None))
            
            # Read tensor data
            tensor_data = None
            if np_dtype is not None:
                try:
                    # Calculate required data size based on dimensions and data type
                    if np_dtype == np.bool_:
                        required_byte_size = num_elements
                    else:
                        required_byte_size = num_elements * np_dtype().itemsize
                    
                    # Read only the required bytes (not the full buffer)
                    raw_data = f.read(required_byte_size)
                    
                    # Skip any remaining buffer bytes if buffer is larger than needed
                    remaining_bytes = buffer_byte_size - required_byte_size
                    if remaining_bytes > 0:
                        f.read(remaining_bytes)
                    
                    # Convert to numpy array
                    if np_dtype == np.bool_:
                        tensor_data = np.frombuffer(raw_data, dtype=np.uint8).astype(np.bool_)
                    else:
                        tensor_data = np.frombuffer(raw_data, dtype=np_dtype)
                    
                    # Reshape to original dimensions
                    if dims:
                        tensor_data = tensor_data.reshape(dims)
                        
                except Exception as e:
                    print(f"Error reading data for tensor {tensor_name}: {e}")
                    # Skip the data if we can't read it
                    f.read(buffer_byte_size)
            else:
                print(f"Unsupported data type {type_name} for tensor {tensor_name}")
                # Skip unsupported data types
                f.read(buffer_byte_size)
            
            # Store tensor information
            tensors[tensor_name] = {
                'type': 'input' if tensor_type == 1 else 'output',
                'data_type': type_name,
                'dimensions': dims,
                'num_elements': num_elements,
                'data': tensor_data
            }
            
            print(f"  {tensor_name}: {type_name} {dims} ({'input' if tensor_type == 1 else 'output'})")
            
            # Check for large values if data is available
            if tensor_data is not None and tensor_data.dtype in [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
                large_values = tensor_data[tensor_data > 600]
                if len(large_values) > 0:
                    print(f"    Found {len(large_values)} values > 600, max: {np.max(large_values)}")
    
    return tensors

def save_to_pickle(tensors: Dict[str, Any], output_path: str):
    """Save tensor data to Python pickle format."""
    with open(output_path, 'wb') as f:
        pickle.dump(tensors, f)
    print(f"Saved tensor data to {output_path}")

def print_tensor_values(tensors: Dict[str, Any], tensor_names: List[str]):
    """Print the actual values of specific tensors."""
    print("\n=== Tensor Values ===")
    
    for name in tensor_names:
        if name in tensors:
            info = tensors[name]
            print(f"\nTensor: {name}")
            print(f"  Type: {info['type']}")
            print(f"  Data Type: {info['data_type']}")
            print(f"  Shape: {info['dimensions']}")
            
            if info['data'] is not None:
                print(f"  Values: {info['data']}")
                if info['data'].size > 20:  # If tensor is large, show summary
                    print(f"  First 10 elements: {info['data'].flat[:10]}")
                    print(f"  Last 10 elements: {info['data'].flat[-10:]}")
                    print(f"  Min: {np.min(info['data'])}, Max: {np.max(info['data'])}, Mean: {np.mean(info['data']):.4f}")
            else:
                print("  No data available")
        else:
            print(f"\nTensor '{name}' not found in dump")

def analyze_tensors(tensors: Dict[str, Any]):
    """Analyze tensor data and print statistics."""
    print("\n=== Tensor Analysis ===")
    
    for name, info in tensors.items():
        print(f"\nTensor: {name}")
        print(f"  Type: {info['type']}")
        print(f"  Data Type: {info['data_type']}")
        print(f"  Shape: {info['dimensions']}")
        print(f"  Elements: {info['num_elements']}")
        
        if info['data'] is not None:
            data = info['data']
            if data.dtype in [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
                print(f"  Min: {np.min(data)}")
                print(f"  Max: {np.max(data)}")
                print(f"  Mean: {np.mean(data):.4f}")
                print(f"  Std: {np.std(data):.4f}")
                
                # Count values > 600
                large_count = np.sum(data > 600)
                if large_count > 0:
                    print(f"  Values > 600: {large_count} ({100.0 * large_count / data.size:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Read and process TensorRT tensor dump files')
    parser.add_argument('input_file', help='Path to tensor dump file (.dat)')
    parser.add_argument('--pickle', '-p', help='Save to pickle file')
    parser.add_argument('--analyze', '-a', action='store_true', help='Analyze tensor statistics')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found")
        return 1
    
    try:
        # Read tensor dump
        tensors = read_tensor_dump(args.input_file)
        
        # Save to pickle if requested
        if args.pickle:
            save_to_pickle(tensors, args.pickle)
        
        # Analyze tensors if requested
        if args.analyze:
            analyze_tensors(tensors)
        
        # Print specific tensor values (tokens and s_prev)
        print_tensor_values(tensors, ['tokens', 's_prev'])
            
        print(f"\nSuccessfully processed {len(tensors)} tensors")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
