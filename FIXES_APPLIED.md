# Fixes Applied to RadVLM Evaluation Script

## Problem
Your `accelerate launch --num_processes=4` command was causing all 4 processes to load the model simultaneously, leading to I/O contention and the script hanging at "Loading checkpoint shards: 25%".

## Root Cause
The original script was calling `load_model_and_processor()` on all processes at the same time, causing disk I/O bottlenecks when multiple processes tried to read the same large model files simultaneously.

## Solution Applied

### 1. Sequential Model Loading in `evaluate_instructions.py`
**Before:**
```python
# Load model
print("Loading model...")
tokenizer, model, processor = load_model_and_processor(args.model_name)
print("Model loaded successfully")

# Setup distributed state
distributed_state = PartialState()
```

**After:**
```python
# Setup distributed state first
distributed_state = PartialState()
print(f"Process {distributed_state.process_index}/{distributed_state.num_processes}")

# Sequential model loading to avoid I/O contention
print("Loading model sequentially...")
for rank in range(distributed_state.num_processes):
    if distributed_state.process_index == rank:
        print(f"Process {rank}: Loading model now")
        tokenizer, model, processor = load_model_and_processor(args.model_name)
        print(f"Process {rank}: Model loaded successfully")
        model.to(distributed_state.device)
        model.eval()
    distributed_state.wait_for_everyone()

print("All models loaded, continuing...")
```

### 2. Added Qwen Model Support in `models_loading_inference.py`
- Added proper Qwen checkpoint loading logic that detects checkpoint paths containing "qwen"
- Added `inference_qwen2vl()` function for Qwen model inference
- Added fallback for both `Qwen2_5VLForConditionalGeneration` and `Qwen2VLForConditionalGeneration` classes

### 3. Enhanced R1 Mode Support
- The script now properly extracts answers from `<answer>` tags when `--r1` flag is used
- Falls back to raw output if no tags are found
- Includes debug output for the first few examples to help troubleshoot

## Expected Behavior
With these fixes, when you run:
```bash
accelerate launch --num_processes=4 radvlm/evaluation/evaluate_instructions.py --task abnormality_grounding --r1 --model_name $SCRATCH/checkpoints/qwen_step_540 --num_batches=10
```

You should see:
1. **Sequential Loading**: Each process loads the model one at a time
   - Process 0 loads first, others wait
   - Process 1 loads second, others wait  
   - Process 2 loads third, others wait
   - Process 3 loads last
2. **Proper NCCL Initialization**: After all models are loaded, NCCL should initialize successfully
3. **Qwen Inference**: The script will correctly use `inference_qwen2vl()` for your Qwen checkpoint
4. **R1 Output Parsing**: Answers will be extracted from `<answer>` tags as expected

## Files Modified
- `/workspace/radvlm/evaluation/evaluate_instructions.py` - Main evaluation script with sequential loading
- `/workspace/radvlm/evaluation/models_loading_inference.py` - Added Qwen support and inference function

The script should now run successfully without hanging and produce meaningful evaluation results.