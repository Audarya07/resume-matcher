#!/usr/bin/env python3
"""
Command-line interface for the resume matching tool.

Usage:
    python main.py --help
    python main.py match --jd data/job_description.txt --resumes data/resumes/
    python main.py evaluate --labels data/evaluation_labels.json
"""

import argparse
import json
from pathlib import Path


from src.matcher import ResumeMatcher
from src.evaluation import (
    load_evaluation_labels,
    extract_ground_truth,
    compare_methods,
    create_evaluation_report,
)


def run_matching(args):
    """Run the resume matching process."""
    print(f"\n{'='*60}")
    print("RESUME MATCHER")
    print(f"{'='*60}")
    print(f"\nJob Description: {args.jd}")
    print(f"Resumes Directory: {args.resumes}")
    print(f"Model: {args.model}")
    print()
    
    # Initialize matcher
    print("Loading and processing documents...")
    matcher = ResumeMatcher(embedding_model=args.model)
    matcher.load_job_description(args.jd)
    matcher.load_resumes(args.resumes)
    
    print(f"Loaded {len(matcher.resumes_raw)} resumes")
    print("\nGenerating embeddings and scoring...")
    
    # Get results
    results = matcher.get_detailed_results()
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS (sorted by embedding score)")
    print("="*60)
    print(f"\n{'Rank':<6}{'Resume':<35}{'Embedding':<12}{'TF-IDF':<12}")
    print("-" * 65)
    
    for i, result in enumerate(results, 1):
        resume_id = result['resume_id'][:32] + "..." if len(result['resume_id']) > 32 else result['resume_id']
        print(f"{i:<6}{resume_id:<35}{result['embedding_score']:<12.4f}{result['tfidf_score']:<12.4f}")
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print()


def run_evaluation(args):
    """Run evaluation with labeled data."""
    print(f"\n{'='*60}")
    print("RESUME MATCHER EVALUATION")
    print(f"{'='*60}")
    
    # Load evaluation labels
    eval_data = load_evaluation_labels(args.labels)
    ground_truth = extract_ground_truth(eval_data)
    
    print(f"\nLoaded {len(ground_truth)} labeled resumes")
    print(f"Job Description: {eval_data['job_description']}")
    
    # Find resumes directory (relative to labels file)
    labels_path = Path(args.labels)
    resumes_dir = labels_path.parent / 'resumes'
    jd_path = labels_path.parent / 'job_description.txt'
    
    # Initialize and run matcher
    print("\nRunning matcher...")
    matcher = ResumeMatcher(embedding_model=args.model)
    matcher.load_job_description(str(jd_path))
    matcher.load_resumes(str(resumes_dir))
    
    # Get rankings from both methods
    embedding_ranked = matcher.rank_resumes('embedding')
    tfidf_ranked = matcher.rank_resumes('tfidf')
    
    # Compare methods
    comparison = compare_methods(
        embedding_ranked,
        tfidf_ranked,
        ground_truth,
        k_values=[3, 5, len(ground_truth)]
    )
    
    # Display results
    print("\n" + "="*60)
    print("EMBEDDING-BASED RESULTS")
    print("="*60)
    print(f"\n{'Rank':<6}{'Resume':<40}{'Score':<10}{'Label':<10}")
    print("-" * 66)
    
    for i, (resume_id, score) in enumerate(embedding_ranked, 1):
        label = ground_truth.get(resume_id, '?')
        resume_short = resume_id[:37] + "..." if len(resume_id) > 37 else resume_id
        print(f"{i:<6}{resume_short:<40}{score:<10.4f}{label:<10}")
    
    print("\n" + "="*60)
    print("TF-IDF BASELINE RESULTS")
    print("="*60)
    print(f"\n{'Rank':<6}{'Resume':<40}{'Score':<10}{'Label':<10}")
    print("-" * 66)
    
    for i, (resume_id, score) in enumerate(tfidf_ranked, 1):
        label = ground_truth.get(resume_id, '?')
        resume_short = resume_id[:37] + "..." if len(resume_id) > 37 else resume_id
        print(f"{i:<6}{resume_short:<40}{score:<10.4f}{label:<10}")
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print("\nEmbedding-based Matching:")
    for key, value in comparison['embedding'].items():
        if not key.startswith('num_'):
            print(f"  {key}: {value}")
    
    print("\nTF-IDF Baseline:")
    for key, value in comparison['tfidf'].items():
        if not key.startswith('num_'):
            print(f"  {key}: {value}")
    
    print("\nImprovement (Embedding over TF-IDF):")
    for key, value in comparison['improvement'].items():
        print(f"  {key}: {value:+.1f}%")
    
    # Generate and save detailed report if requested
    if args.report:
        predictions_dict = {
            'embedding': embedding_ranked,
            'tfidf': tfidf_ranked,
        }
        report = create_evaluation_report(predictions_dict, ground_truth)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nDetailed evaluation report saved to: {args.report}")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-powered Resume Matching System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Match resumes to a job description:
    python main.py match --jd data/job_description.txt --resumes data/resumes/

  Run evaluation with labeled data:
    python main.py evaluate --labels data/evaluation_labels.json

  Use a faster model:
    python main.py match --jd jd.txt --resumes resumes/ --model all-MiniLM-L6-v2
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match resumes to job description')
    match_parser.add_argument('--jd', required=True, help='Path to job description file')
    match_parser.add_argument('--resumes', required=True, help='Path to resumes directory')
    match_parser.add_argument('--output', '-o', help='Path to save results JSON')
    match_parser.add_argument('--model', default='all-mpnet-base-v2',
                             help='Sentence Transformer model (default: all-mpnet-base-v2)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation with labeled data')
    eval_parser.add_argument('--labels', required=True, help='Path to evaluation_labels.json')
    eval_parser.add_argument('--model', default='all-mpnet-base-v2',
                            help='Sentence Transformer model (default: all-mpnet-base-v2)')
    eval_parser.add_argument('--report', '-r', help='Path to save detailed evaluation report')
    
    args = parser.parse_args()
    
    if args.command == 'match':
        run_matching(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
