#!/usr/bin/env python
"""
Validation script for VLM-R1 compatible JSONL files.
Checks format compliance, data integrity, and provides statistics.
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import re


def validate_vlmr1_sample(sample, line_num, image_base_dir=None):
    """
    Validate a single VLM-R1 sample.
    
    Args:
        sample: Dict representing one JSONL line
        line_num: Line number for error reporting
        image_base_dir: Base directory to check image file existence
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['id', 'image', 'conversations']
    for field in required_fields:
        if field not in sample:
            errors.append(f"Line {line_num}: Missing required field '{field}'")
    
    if errors:
        return errors
    
    # Validate ID
    if not isinstance(sample['id'], (str, int)):
        errors.append(f"Line {line_num}: 'id' must be string or integer")
    
    # Validate image field
    image = sample['image']
    if isinstance(image, str):
        # Single image
        image_paths = [image]
        image_count = 1
    elif isinstance(image, list):
        # Multi-image
        if not image:
            errors.append(f"Line {line_num}: 'image' list cannot be empty")
        image_paths = image
        image_count = len(image)
        for i, img_path in enumerate(image_paths):
            if not isinstance(img_path, str):
                errors.append(f"Line {line_num}: image[{i}] must be string")
    else:
        errors.append(f"Line {line_num}: 'image' must be string or list of strings")
        return errors
    
    # Check image file existence if base directory provided
    if image_base_dir:
        for i, img_path in enumerate(image_paths):
            full_path = os.path.join(image_base_dir, img_path)
            if not os.path.exists(full_path):
                errors.append(f"Line {line_num}: Image file does not exist: {full_path}")
    
    # Validate conversations
    conversations = sample['conversations']
    if not isinstance(conversations, list):
        errors.append(f"Line {line_num}: 'conversations' must be a list")
        return errors
    
    if not conversations:
        errors.append(f"Line {line_num}: 'conversations' cannot be empty")
        return errors
    
    # Check conversation format
    image_token_count = 0
    for i, conv in enumerate(conversations):
        if not isinstance(conv, dict):
            errors.append(f"Line {line_num}: conversations[{i}] must be dict")
            continue
        
        if 'from' not in conv or 'value' not in conv:
            errors.append(f"Line {line_num}: conversations[{i}] missing 'from' or 'value'")
            continue
        
        if conv['from'] not in ['human', 'gpt']:
            errors.append(f"Line {line_num}: conversations[{i}]['from'] must be 'human' or 'gpt'")
        
        if not isinstance(conv['value'], str):
            errors.append(f"Line {line_num}: conversations[{i}]['value'] must be string")
        
        # Count <image> tokens in human messages
        if conv['from'] == 'human':
            image_tokens = conv['value'].count('<image>')
            image_token_count += image_tokens
    
    # Validate image token count matches image count
    if image_token_count != image_count:
        errors.append(f"Line {line_num}: Image token count ({image_token_count}) doesn't match image count ({image_count})")
    
    return errors


def analyze_dataset_statistics(samples):
    """
    Analyze dataset statistics.
    
    Args:
        samples: List of valid VLM-R1 samples
    
    Returns:
        Dict with statistics
    """
    stats = {
        'total_samples': len(samples),
        'single_image_samples': 0,
        'multi_image_samples': 0,
        'max_images_per_sample': 0,
        'conversation_lengths': [],
        'image_extensions': Counter(),
        'id_prefixes': Counter(),
        'sample_types': defaultdict(int)
    }
    
    for sample in samples:
        # Image statistics
        if isinstance(sample['image'], str):
            stats['single_image_samples'] += 1
            image_count = 1
            image_paths = [sample['image']]
        else:
            stats['multi_image_samples'] += 1
            image_count = len(sample['image'])
            image_paths = sample['image']
        
        stats['max_images_per_sample'] = max(stats['max_images_per_sample'], image_count)
        
        # Image extensions
        for img_path in image_paths:
            ext = Path(img_path).suffix.lower()
            stats['image_extensions'][ext] += 1
        
        # Conversation statistics
        conv_length = len(sample['conversations'])
        stats['conversation_lengths'].append(conv_length)
        
        # ID prefixes
        sample_id = str(sample['id'])
        if '_' in sample_id:
            prefix = sample_id.split('_')[0]
            stats['id_prefixes'][prefix] += 1
        
        # Sample type classification based on content
        first_human_msg = None
        for conv in sample['conversations']:
            if conv['from'] == 'human':
                first_human_msg = conv['value'].lower()
                break
        
        if first_human_msg:
            if 'report' in first_human_msg or 'findings' in first_human_msg:
                stats['sample_types']['report_generation'] += 1
            elif 'abnormal' in first_human_msg or 'lesion' in first_human_msg:
                stats['sample_types']['abnormality_detection'] += 1
            elif 'location' in first_human_msg or 'where' in first_human_msg:
                stats['sample_types']['grounding'] += 1
            elif 'classify' in first_human_msg or 'diagnosis' in first_human_msg:
                stats['sample_types']['classification'] += 1
            else:
                stats['sample_types']['other'] += 1
    
    # Calculate conversation length statistics
    if stats['conversation_lengths']:
        stats['avg_conversation_length'] = sum(stats['conversation_lengths']) / len(stats['conversation_lengths'])
        stats['min_conversation_length'] = min(stats['conversation_lengths'])
        stats['max_conversation_length'] = max(stats['conversation_lengths'])
    else:
        stats['avg_conversation_length'] = 0
        stats['min_conversation_length'] = 0
        stats['max_conversation_length'] = 0
    
    return stats


def validate_jsonl_file(file_path, image_base_dir=None, max_lines=None):
    """
    Validate a complete JSONL file.
    
    Args:
        file_path: Path to JSONL file
        image_base_dir: Base directory for image existence checking
        max_lines: Maximum lines to validate (for large files)
    
    Returns:
        Tuple of (valid_samples, errors, warnings)
    """
    valid_samples = []
    errors = []
    warnings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_lines and line_num > max_lines:
                    warnings.append(f"Stopped validation at line {max_lines} due to limit")
                    break
                
                line = line.strip()
                if not line:
                    warnings.append(f"Line {line_num}: Empty line")
                    continue
                
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON decode error: {e}")
                    continue
                
                # Validate sample
                sample_errors = validate_vlmr1_sample(sample, line_num, image_base_dir)
                errors.extend(sample_errors)
                
                if not sample_errors:
                    valid_samples.append(sample)
    
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as e:
        errors.append(f"Error reading file: {e}")
    
    return valid_samples, errors, warnings


def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Single image samples: {stats['single_image_samples']:,}")
    print(f"Multi-image samples: {stats['multi_image_samples']:,}")
    print(f"Max images per sample: {stats['max_images_per_sample']}")
    
    print(f"\nConversation lengths:")
    print(f"  Average: {stats['avg_conversation_length']:.1f}")
    print(f"  Min: {stats['min_conversation_length']}")
    print(f"  Max: {stats['max_conversation_length']}")
    
    print(f"\nImage extensions:")
    for ext, count in stats['image_extensions'].most_common():
        print(f"  {ext or 'no_extension'}: {count:,}")
    
    print(f"\nDataset prefixes:")
    for prefix, count in stats['id_prefixes'].most_common():
        print(f"  {prefix}: {count:,}")
    
    print(f"\nSample types:")
    for sample_type, count in stats['sample_types'].items():
        print(f"  {sample_type}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate VLM-R1 compatible JSONL files"
    )
    parser.add_argument(
        "jsonl_file",
        help="Path to JSONL file to validate"
    )
    parser.add_argument(
        "--image-base-dir",
        help="Base directory for checking image file existence"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        help="Maximum lines to validate (useful for large files)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, skip detailed validation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and warnings"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_file):
        print(f"❌ File not found: {args.jsonl_file}")
        return 1
    
    print(f"Validating: {args.jsonl_file}")
    if args.image_base_dir:
        print(f"Image base directory: {args.image_base_dir}")
    
    # Validate file
    valid_samples, errors, warnings = validate_jsonl_file(
        args.jsonl_file, 
        args.image_base_dir,
        args.max_lines
    )
    
    # Print results
    if not args.quiet:
        print(f"\n✅ Valid samples: {len(valid_samples):,}")
    
    if warnings and not args.quiet:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10 warnings
            print(f"  {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for error in errors[:20]:  # Show first 20 errors
            print(f"  {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    # Generate statistics for valid samples
    if valid_samples and not args.stats_only:
        stats = analyze_dataset_statistics(valid_samples)
        if not args.quiet:
            print_statistics(stats)
    
    # Summary
    total_lines = len(valid_samples) + len(errors)
    success_rate = len(valid_samples) / total_lines * 100 if total_lines > 0 else 0
    
    print(f"\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Valid samples: {len(valid_samples):,}")
    print(f"Errors: {len(errors):,}")
    print(f"Warnings: {len(warnings):,}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if errors:
        print(f"\n❌ Validation failed with {len(errors)} errors")
        return 1
    else:
        print(f"\n✅ Validation passed!")
        return 0


if __name__ == "__main__":
    exit(main())