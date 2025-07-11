# TensorRT Backend Tensor Dump Format

## Overview

The TensorRT backend automatically dumps all input and output tensors to binary files when any tensor values exceed 600. This feature is implemented in `instance_state.cc` and helps with debugging models that produce unexpectedly large values.

## File Naming Convention

Files are saved with the following format:
```
tensor_dump_large_values_detected_YYYYMMDD_HHMMSS_mmm.dat
```

Where:
- `YYYY`: Year (4 digits)
- `MM`: Month (2 digits) 
- `DD`: Day (2 digits)
- `HH`: Hour (24-hour format, 2 digits)
- `MM`: Minutes (2 digits)
- `SS`: Seconds (2 digits)
- `mmm`: Milliseconds (3 digits)

Example: `tensor_dump_large_values_detected_20250710_213457_123.dat`

## Binary File Format

The binary files use little-endian byte order throughout.

### File Header
| Field | Type | Description |
|-------|------|-------------|
| num_tensors | int32 | Total number of tensors in the file |

### Per-Tensor Format
For each tensor, the following data is stored sequentially:

| Field | Type | Description |
|-------|------|-------------|
| name_length | int32 | Length of tensor name string |
| tensor_name | char[] | UTF-8 encoded tensor name |
| tensor_type | int32 | 1 = input tensor, 0 = output tensor |
| data_type | int32 | TRITONSERVER_DataType enum value |
| num_dims | int32 | Number of dimensions |
| dimensions | int32[] | Size of each dimension |
| num_elements | int32 | Total number of elements |
| tensor_data | binary | Raw tensor data |

### Data Type Mapping

| TRITONSERVER_DataType | Value | C++ Type | NumPy Type |
|----------------------|-------|----------|------------|
| TRITONSERVER_TYPE_BOOL | 1 | bool | np.bool_ |
| TRITONSERVER_TYPE_UINT8 | 2 | uint8_t | np.uint8 |
| TRITONSERVER_TYPE_UINT16 | 3 | uint16_t | np.uint16 |
| TRITONSERVER_TYPE_UINT32 | 4 | uint32_t | np.uint32 |
| TRITONSERVER_TYPE_UINT64 | 5 | uint64_t | np.uint64 |
| TRITONSERVER_TYPE_INT8 | 6 | int8_t | np.int8 |
| TRITONSERVER_TYPE_INT16 | 7 | int16_t | np.int16 |
| TRITONSERVER_TYPE_INT32 | 8 | int32_t | np.int32 |
| TRITONSERVER_TYPE_INT64 | 9 | int64_t | np.int64 |
| TRITONSERVER_TYPE_FP16 | 10 | half | np.float16 |
| TRITONSERVER_TYPE_FP32 | 11 | float | np.float32 |
| TRITONSERVER_TYPE_FP64 | 12 | double | np.float64 |

## Trigger Conditions

Tensor dumping is triggered when **any** of the following conditions are met for **any** tensor:

- INT8 values > 600
- INT16 values > 600  
- INT32 values > 600
- INT64 values > 600
- FP32 values > 600.0f
- FP64 values > 600.0

When triggered, **ALL** input and output tensors are dumped, not just the ones with large values.

## Usage Examples

### Reading with Python Script

```bash
# Basic reading
python3 read_tensor_dump.py tensor_dump_large_values_detected_20250710_213457_123.dat

# Save to pickle format
python3 read_tensor_dump.py -p output.pkl tensor_dump_large_values_detected_20250710_213457_123.dat

# Analyze tensor statistics
python3 read_tensor_dump.py -a tensor_dump_large_values_detected_20250710_213457_123.dat
```

### Reading in Python Code

```python
from read_tensor_dump import read_tensor_dump

# Load tensor data
tensors = read_tensor_dump("tensor_dump_large_values_detected_20250710_213457_123.dat")

# Access specific tensor
input_tensor = tensors["input_tensor_name"]
print(f"Shape: {input_tensor['dimensions']}")
print(f"Data type: {input_tensor['data_type']}")
print(f"Data: {input_tensor['data']}")  # NumPy array
```

## Implementation Details

### Memory Management
- GPU tensors are copied to CPU memory before dumping
- Memory allocation is carefully managed to avoid leaks
- CUDA stream synchronization ensures data consistency

### Performance Considerations
- Dumping only occurs when large values are detected
- GPU-to-CPU memory transfers are minimized
- Binary format provides efficient storage and fast reading

### Error Handling
- Failed memory allocations are logged and handled gracefully
- Unsupported data types are skipped with appropriate logging
- File I/O errors are caught and reported

## Location in Code

The tensor dumping functionality is implemented in:
- **Header**: `/src/instance_state.h` - Function declaration
- **Implementation**: `/src/instance_state.cc` - Lines ~1074-1075 (trigger logic) and ~3980+ (dump function)
- **Python Reader**: `/read_tensor_dump.py` - Binary format parser
