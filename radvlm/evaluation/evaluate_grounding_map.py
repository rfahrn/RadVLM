import torch
import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Tuple
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm

def parse_bbox_response(response: str) -> List[List[float]]:
    """
    Parse bounding box coordinates from model response.
    Expected format: Various formats including JSON, lists, or plain text.
    """
    import re
    
    # Try to extract JSON format first
    try:
        # Look for JSON-like structures
        json_match = re.search(r'\{[^}]*"bbox_2d"[^}]*\}', response)
        if json_match:
            bbox_data = json.loads(json_match.group())
            if 'bbox_2d' in bbox_data:
                return bbox_data['bbox_2d']
    except:
        pass
    
    # Try to extract coordinate lists
    try:
        # Look for patterns like [x1, y1, x2, y2] or [[x1, y1, x2, y2]]
        coord_pattern = r'\[([^\]]+)\]'
        matches = re.findall(coord_pattern, response)
        
        coords = []
        for match in matches:
            # Split by comma and try to convert to float
            values = [float(x.strip()) for x in match.split(',')]
            if len(values) == 4:
                coords.append(values)
        
        if coords:
            return coords
    except:
        pass
    
    # Try to extract space-separated coordinates
    try:
        # Look for patterns like "0.1 0.2 0.3 0.4"
        number_pattern = r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)'
        matches = re.findall(number_pattern, response)
        
        if matches:
            coords = []
            for match in matches:
                coords.append([float(x) for x in match])
            return coords
    except:
        pass
    
    return []

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes.
    Both boxes should be in format [x1, y1, x2, y2] (normalized coordinates).
    """
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def calculate_best_iou(pred_boxes: List[List[float]], gt_boxes: List[List[float]]) -> float:
    """
    Calculate the best IoU between predicted and ground truth boxes.
    """
    if not pred_boxes or not gt_boxes:
        return 0.0
    
    max_iou = 0.0
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            iou = calculate_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
    
    return max_iou

def evaluate_grounding_performance(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict[str, float]:
    """
    Evaluate grounding performance using mAP and IoU metrics.
    
    Args:
        predictions: List of prediction dictionaries with 'response' and 'sample_id'
        ground_truth: List of ground truth dictionaries with 'boxes' and 'sample_id'
        iou_thresholds: List of IoU thresholds for mAP calculation
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create mapping from sample_id to ground truth
    gt_mapping = {item['sample_id']: item for item in ground_truth}
    
    results = []
    
    print("Processing predictions...")
    for pred in tqdm(predictions):
        sample_id = pred['sample_id']
        response = pred['response']
        
        if sample_id not in gt_mapping:
            continue
        
        gt_data = gt_mapping[sample_id]
        gt_boxes = gt_data['ground_truth']
        
        # Parse predicted boxes
        pred_boxes = parse_bbox_response(response)
        
        # Calculate IoU
        iou = calculate_best_iou(pred_boxes, gt_boxes)
        
        results.append({
            'sample_id': sample_id,
            'iou': iou,
            'has_prediction': len(pred_boxes) > 0,
            'has_ground_truth': len(gt_boxes) > 0,
            'dataset_source': gt_data.get('dataset_source', 'unknown')
        })
    
    # Calculate metrics
    metrics = {}
    
    # Overall IoU statistics
    ious = [r['iou'] for r in results]
    metrics['mean_iou'] = np.mean(ious)
    metrics['median_iou'] = np.median(ious)
    metrics['std_iou'] = np.std(ious)
    
    # mAP calculation for different IoU thresholds
    for threshold in iou_thresholds:
        # Binary classification: IoU >= threshold
        y_true = [1 if r['has_ground_truth'] else 0 for r in results]
        y_scores = [1.0 if r['iou'] >= threshold else 0.0 for r in results]
        
        if len(set(y_true)) > 1:  # Need both positive and negative samples
            ap = average_precision_score(y_true, y_scores)
            metrics[f'mAP@{threshold}'] = ap
        else:
            metrics[f'mAP@{threshold}'] = 0.0
    
    # Detection rate (has valid prediction)
    detection_rate = np.mean([r['has_prediction'] for r in results])
    metrics['detection_rate'] = detection_rate
    
    # Per-dataset metrics
    datasets = set(r['dataset_source'] for r in results)
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset_source'] == dataset]
        dataset_ious = [r['iou'] for r in dataset_results]
        
        metrics[f'{dataset}_mean_iou'] = np.mean(dataset_ious)
        metrics[f'{dataset}_count'] = len(dataset_results)
    
    # IoU distribution
    iou_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iou_hist, _ = np.histogram(ious, bins=iou_bins)
    
    for i, (low, high) in enumerate(zip(iou_bins[:-1], iou_bins[1:])):
        metrics[f'iou_{low}_{high}'] = iou_hist[i]
    
    return metrics, results

def load_predictions_from_file(filepath: str) -> List[Dict]:
    """
    Load predictions from a file (JSON or parquet).
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return data

def load_ground_truth_from_parquet(filepath: str) -> List[Dict]:
    """
    Load ground truth from the test parquet file created by the dataset script.
    """
    df = pd.read_parquet(filepath)
    
    ground_truth = []
    for _, row in df.iterrows():
        ground_truth.append({
            'sample_id': row['extra_info']['sample_id'],
            'ground_truth': row['reward_model']['ground_truth'],
            'label': row['reward_model']['label'],
            'dataset_source': row['extra_info']['dataset_source']
        })
    
    return ground_truth

def main():
    parser = argparse.ArgumentParser(description="Evaluate grounding performance using mAP and IoU")
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions file (JSON or parquet)')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth test parquet file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory for results')
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7], help='IoU thresholds for mAP')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading predictions...")
    predictions = load_predictions_from_file(args.predictions)
    
    print("Loading ground truth...")
    ground_truth = load_ground_truth_from_parquet(args.ground_truth)
    
    # Evaluate
    print("Evaluating performance...")
    metrics, detailed_results = evaluate_grounding_performance(
        predictions, ground_truth, args.iou_thresholds
    )
    
    # Save results
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    detailed_path = os.path.join(args.output_dir, 'detailed_results.json')
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print("\n=== Grounding Evaluation Results ===")
    print(f"Total samples: {len(detailed_results)}")
    print(f"Mean IoU: {metrics['mean_iou']:.3f}")
    print(f"Median IoU: {metrics['median_iou']:.3f}")
    print(f"Detection rate: {metrics['detection_rate']:.3f}")
    
    print("\nmAP scores:")
    for threshold in args.iou_thresholds:
        print(f"  mAP@{threshold}: {metrics[f'mAP@{threshold}']:.3f}")
    
    print("\nPer-dataset performance:")
    datasets = set(r['dataset_source'] for r in detailed_results)
    for dataset in sorted(datasets):
        mean_iou = metrics[f'{dataset}_mean_iou']
        count = metrics[f'{dataset}_count']
        print(f"  {dataset}: {mean_iou:.3f} (n={count})")
    
    print("\nIoU distribution:")
    iou_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, (low, high) in enumerate(zip(iou_bins[:-1], iou_bins[1:])):
        count = metrics[f'iou_{low}_{high}']
        print(f"  {low}-{high}: {count}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main()