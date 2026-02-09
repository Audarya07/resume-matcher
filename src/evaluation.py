"""
Evaluation module for the resume matching system.

This module provides functions to evaluate the performance of the matching
algorithms using various metrics commonly used in information retrieval
and ranking systems.

Metrics included:
- Precision@K: Of the top K results, how many are relevant?
- Recall@K: Of all relevant items, how many appear in the top K?
- nDCG (Normalized Discounted Cumulative Gain): Measures ranking quality
- Spearman Correlation: How well does predicted ranking match true ranking?
- Mean Absolute Error: Average error between predicted and true scores
"""

from typing import Dict, List, Tuple
import json
import numpy as np
from scipy import stats


def load_evaluation_labels(file_path: str) -> Dict:
    """
    Load evaluation labels from a JSON file.
    
    Args:
        file_path: Path to the evaluation_labels.json file
        
    Returns:
        Dictionary with evaluation data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def precision_at_k(
    predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
    k: int,
    relevance_threshold: float = 0.5,
) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# of relevant items in top K) / K
    
    Args:
        predictions: List of (identifier, score) sorted by score descending
        ground_truth: Dictionary mapping identifiers to true relevance labels
        k: Number of top results to consider
        relevance_threshold: Minimum label to be considered relevant
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant_count = sum(
        1 for identifier, _ in top_k
        if ground_truth.get(identifier, 0) >= relevance_threshold
    )
    
    return relevant_count / k


def recall_at_k(
    predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
    k: int,
    relevance_threshold: float = 0.5,
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# of relevant items in top K) / (# of total relevant items)
    
    Args:
        predictions: List of (identifier, score) sorted by score descending
        ground_truth: Dictionary mapping identifiers to true relevance labels
        k: Number of top results to consider
        relevance_threshold: Minimum label to be considered relevant
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    total_relevant = sum(
        1 for label in ground_truth.values()
        if label >= relevance_threshold
    )
    
    if total_relevant == 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant_in_top_k = sum(
        1 for identifier, _ in top_k
        if ground_truth.get(identifier, 0) >= relevance_threshold
    )
    
    return relevant_in_top_k / total_relevant


def dcg_at_k(scores: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG@K = sum(relevance[i] / log2(i + 2)) for i in 0..k-1
    
    Args:
        scores: List of relevance scores in prediction order
        k: Number of results to consider
        
    Returns:
        DCG@K score
    """
    scores = scores[:k]
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return np.sum(scores / discounts)


def ndcg_at_k(
    predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
    k: int,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    nDCG@K = DCG@K / IDCG@K
    
    where IDCG@K is the DCG of the ideal ranking.
    
    Args:
        predictions: List of (identifier, score) sorted by score descending
        ground_truth: Dictionary mapping identifiers to true relevance labels
        k: Number of results to consider
        
    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    # Get relevance scores in prediction order
    relevance_scores = [
        ground_truth.get(identifier, 0)
        for identifier, _ in predictions[:k]
    ]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance_scores, k)
    
    # Calculate ideal DCG (best possible ranking)
    ideal_scores = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def spearman_correlation(
    predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation between predicted and true rankings.
    
    A correlation of 1.0 means perfect agreement in ranking order.
    
    Args:
        predictions: List of (identifier, score) sorted by score descending
        ground_truth: Dictionary mapping identifiers to true relevance labels
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    # Get predicted scores in order
    predicted_scores = [score for _, score in predictions]
    
    # Get true scores in same order
    true_scores = [ground_truth.get(identifier, 0) for identifier, _ in predictions]
    
    if len(predicted_scores) < 2:
        return 0.0, 1.0
    
    correlation, p_value = stats.spearmanr(predicted_scores, true_scores)
    
    # Handle NaN (can occur if all scores are the same)
    if np.isnan(correlation):
        correlation = 0.0
    
    return float(correlation), float(p_value)


def mean_absolute_error(
    predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
) -> float:
    """
    Calculate Mean Absolute Error between predicted scores and true labels.
    
    Note: This assumes predicted scores are on the same scale as labels (0-1).
    
    Args:
        predictions: List of (identifier, score) tuples
        ground_truth: Dictionary mapping identifiers to true labels
        
    Returns:
        MAE (lower is better)
    """
    errors = []
    for identifier, predicted_score in predictions:
        true_score = ground_truth.get(identifier, 0)
        errors.append(abs(predicted_score - true_score))
    
    if not errors:
        return 0.0
    
    return float(np.mean(errors))


def evaluate_matcher(
    predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
    k_values: List[int] = [3, 5, 10],
) -> Dict:
    """
    Run comprehensive evaluation of matcher predictions.
    
    Args:
        predictions: List of (identifier, score) sorted by score descending
        ground_truth: Dictionary mapping identifiers to true labels
        k_values: List of K values for Precision@K, Recall@K, nDCG@K
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {
        'num_predictions': len(predictions),
        'num_ground_truth': len(ground_truth),
    }
    
    # Calculate metrics for each K
    for k in k_values:
        results[f'precision@{k}'] = round(precision_at_k(predictions, ground_truth, k), 4)
        results[f'recall@{k}'] = round(recall_at_k(predictions, ground_truth, k), 4)
        results[f'ndcg@{k}'] = round(ndcg_at_k(predictions, ground_truth, k), 4)
    
    # Overall metrics
    correlation, p_value = spearman_correlation(predictions, ground_truth)
    results['spearman_correlation'] = round(correlation, 4)
    results['spearman_p_value'] = round(p_value, 4)
    
    results['mean_absolute_error'] = round(mean_absolute_error(predictions, ground_truth), 4)
    
    return results


def compare_methods(
    embedding_predictions: List[Tuple[str, float]],
    tfidf_predictions: List[Tuple[str, float]],
    ground_truth: Dict[str, float],
    k_values: List[int] = [3, 5],
) -> Dict:
    """
    Compare embedding-based and TF-IDF methods.
    
    Args:
        embedding_predictions: Predictions from embedding matcher
        tfidf_predictions: Predictions from TF-IDF matcher
        ground_truth: True relevance labels
        k_values: K values for evaluation
        
    Returns:
        Dictionary comparing both methods
    """
    embedding_results = evaluate_matcher(embedding_predictions, ground_truth, k_values)
    tfidf_results = evaluate_matcher(tfidf_predictions, ground_truth, k_values)
    
    comparison = {
        'embedding': embedding_results,
        'tfidf': tfidf_results,
        'improvement': {},
    }
    
    # Calculate improvement (embedding over TF-IDF)
    for key in embedding_results:
        if key.startswith(('precision', 'recall', 'ndcg', 'spearman_correlation')):
            emb_value = embedding_results[key]
            tfidf_value = tfidf_results[key]
            if tfidf_value != 0:
                improvement = ((emb_value - tfidf_value) / abs(tfidf_value)) * 100
            else:
                improvement = 0 if emb_value == 0 else 100
            comparison['improvement'][key] = round(improvement, 2)
    
    return comparison


def create_evaluation_report(
    predictions: Dict[str, Dict],
    ground_truth: Dict[str, float],
) -> str:
    """
    Create a human-readable evaluation report.
    
    Args:
        predictions: Dictionary with 'embedding' and 'tfidf' predictions
        ground_truth: True relevance labels
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("RESUME MATCHER EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Ground truth summary
    lines.append("GROUND TRUTH SUMMARY")
    lines.append("-" * 40)
    good_matches = sum(1 for v in ground_truth.values() if v >= 1.0)
    partial_matches = sum(1 for v in ground_truth.values() if 0 < v < 1.0)
    poor_matches = sum(1 for v in ground_truth.values() if v == 0)
    lines.append(f"Total resumes: {len(ground_truth)}")
    lines.append(f"Good matches (1.0): {good_matches}")
    lines.append(f"Partial matches (0.5): {partial_matches}")
    lines.append(f"Poor matches (0.0): {poor_matches}")
    lines.append("")
    
    # Predictions
    if 'embedding' in predictions:
        lines.append("EMBEDDING-BASED MATCHING")
        lines.append("-" * 40)
        for identifier, score in predictions['embedding']:
            true_label = ground_truth.get(identifier, '?')
            lines.append(f"  {identifier}: {score:.4f} (true: {true_label})")
        lines.append("")
    
    if 'tfidf' in predictions:
        lines.append("TF-IDF BASELINE")
        lines.append("-" * 40)
        for identifier, score in predictions['tfidf']:
            true_label = ground_truth.get(identifier, '?')
            lines.append(f"  {identifier}: {score:.4f} (true: {true_label})")
        lines.append("")
    
    return "\n".join(lines)


# Helper function to load labels from evaluation file
def extract_ground_truth(evaluation_data: Dict) -> Dict[str, float]:
    """
    Extract ground truth labels from evaluation data.
    
    Args:
        evaluation_data: Data loaded from evaluation_labels.json
        
    Returns:
        Dictionary mapping resume filename to label
    """
    return {
        item['resume_file']: item['label']
        for item in evaluation_data['evaluations']
    }
