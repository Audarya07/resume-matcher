"""
Unified Resume Matcher module.

This module provides a high-level interface that combines both TF-IDF and
embedding-based matching approaches. Main entry point for the resume matching system.
"""

from typing import Dict, List, Tuple, Optional
import json

try:
    # Try relative imports (when used as a package)
    from .preprocessing import (
        load_text_file,
        preprocess_for_matching,
        load_resumes_from_directory,
        preprocess_resumes,
    )
    from .tfidf_matcher import TFIDFMatcher
    from .embeddings import EmbeddingMatcher
except ImportError:
    # Fall back to absolute imports (when used directly)
    from preprocessing import (
        load_text_file,
        preprocess_for_matching,
        load_resumes_from_directory,
        preprocess_resumes,
    )
    from tfidf_matcher import TFIDFMatcher
    from embeddings import EmbeddingMatcher


class ResumeMatcher:
    """
    High-level resume matching system combining multiple approaches.
    
    This class provides a unified interface for matching resumes to job descriptions
    using both TF-IDF (baseline) and semantic embeddings (primary approach).
    
    Example usage:
        matcher = ResumeMatcher()
        matcher.load_job_description("path/to/jd.txt")
        matcher.load_resumes("path/to/resumes/")
        results = matcher.match_all()
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-mpnet-base-v2',
        use_preprocessing: bool = True,
        preserve_structure: bool = False,
    ):
        """
        Initialize the resume matcher.
        
        Args:
            embedding_model: Sentence Transformer model name
            use_preprocessing: Whether to preprocess texts before matching
            preserve_structure: Whether to preserve section structure in preprocessing
        """
        self.embedding_model = embedding_model
        self.use_preprocessing = use_preprocessing
        self.preserve_structure = preserve_structure
        
        # Initialize matchers
        self.embedding_matcher = EmbeddingMatcher(model_name=embedding_model)
        self.tfidf_matcher = TFIDFMatcher()
        
        # Storage for loaded data
        self.job_description_raw: Optional[str] = None
        self.job_description_processed: Optional[str] = None
        self.resumes_raw: Dict[str, str] = {}
        self.resumes_processed: Dict[str, str] = {}
        
        self._is_ready = False
    
    def load_job_description(self, file_path: str) -> 'ResumeMatcher':
        """
        Load a job description from a file.
        
        Args:
            file_path: Path to the job description text file
            
        Returns:
            self for method chaining
        """
        self.job_description_raw = load_text_file(file_path)
        
        if self.use_preprocessing:
            self.job_description_processed = preprocess_for_matching(
                self.job_description_raw,
                preserve_structure=self.preserve_structure
            )
        else:
            self.job_description_processed = self.job_description_raw
        
        return self
    
    def set_job_description(self, text: str) -> 'ResumeMatcher':
        """
        Set a job description from text directly.
        
        Args:
            text: Job description text
            
        Returns:
            self for method chaining
        """
        self.job_description_raw = text
        
        if self.use_preprocessing:
            self.job_description_processed = preprocess_for_matching(
                text, preserve_structure=self.preserve_structure
            )
        else:
            self.job_description_processed = text
        
        return self
    
    def load_resumes(self, directory: str) -> 'ResumeMatcher':
        """
        Load all resumes from a directory.
        
        Args:
            directory: Path to directory containing resume text files
            
        Returns:
            self for method chaining
        """
        self.resumes_raw = load_resumes_from_directory(directory)
        
        if self.use_preprocessing:
            self.resumes_processed = preprocess_resumes(
                self.resumes_raw,
                preserve_structure=self.preserve_structure
            )
        else:
            self.resumes_processed = self.resumes_raw.copy()
        
        return self
    
    def add_resume(self, identifier: str, text: str) -> 'ResumeMatcher':
        """
        Add a single resume.
        
        Args:
            identifier: Unique identifier for the resume
            text: Resume text content
            
        Returns:
            self for method chaining
        """
        self.resumes_raw[identifier] = text
        
        if self.use_preprocessing:
            self.resumes_processed[identifier] = preprocess_for_matching(
                text, preserve_structure=self.preserve_structure
            )
        else:
            self.resumes_processed[identifier] = text
        
        return self
    
    def prepare(self) -> 'ResumeMatcher':
        """
        Prepare matchers for scoring.
        
        Must be called after loading job description and resumes.
        
        Returns:
            self for method chaining
        """
        if not self.job_description_processed:
            raise ValueError("Job description not loaded. Call load_job_description() first.")
        
        if not self.resumes_processed:
            raise ValueError("No resumes loaded. Call load_resumes() first.")
        
        # Prepare embedding matcher
        self.embedding_matcher.set_job_description(self.job_description_processed)
        
        # Prepare TF-IDF matcher
        resume_texts = list(self.resumes_processed.values())
        self.tfidf_matcher.fit(self.job_description_processed, resume_texts)
        
        self._is_ready = True
        return self
    
    def match_all(self) -> Dict[str, Dict]:
        """
        Match all loaded resumes against the job description.
        
        Returns:
            Dictionary mapping resume identifiers to their scores:
            {
                'resume_id': {
                    'embedding_score': float,
                    'tfidf_score': float,
                    'combined_score': float,
                }
            }
        """
        if not self._is_ready:
            self.prepare()
        
        results = {}
        
        for identifier, text in self.resumes_processed.items():
            embedding_score = self.embedding_matcher.score_resume(text)
            tfidf_score = self.tfidf_matcher.score_resume(text)
            
            # Combined score: weighted average (embedding is primary)
            combined_score = 0.7 * embedding_score + 0.3 * tfidf_score
            
            results[identifier] = {
                'embedding_score': round(embedding_score, 4),
                'tfidf_score': round(tfidf_score, 4),
                'combined_score': round(combined_score, 4),
            }
        
        return results
    
    def rank_resumes(self, method: str = 'embedding') -> List[Tuple[str, float]]:
        """
        Rank resumes by their match scores.
        
        Args:
            method: Scoring method to use ('embedding', 'tfidf', or 'combined')
            
        Returns:
            List of (identifier, score) tuples sorted by score descending
        """
        results = self.match_all()
        
        score_key = {
            'embedding': 'embedding_score',
            'tfidf': 'tfidf_score',
            'combined': 'combined_score',
        }.get(method, 'embedding_score')
        
        ranked = [
            (identifier, scores[score_key])
            for identifier, scores in results.items()
        ]
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_detailed_results(self) -> List[Dict]:
        """
        Get detailed results for all resumes.
        
        Returns:
            List of dictionaries with detailed information for each resume
        """
        results = self.match_all()
        
        detailed = []
        for identifier, scores in results.items():
            detailed.append({
                'resume_id': identifier,
                **scores,
                'rank_embedding': None,  # Will be filled
                'rank_tfidf': None,
            })
        
        # Add rankings
        embedding_ranked = self.rank_resumes('embedding')
        tfidf_ranked = self.rank_resumes('tfidf')
        
        embedding_ranks = {id: i+1 for i, (id, _) in enumerate(embedding_ranked)}
        tfidf_ranks = {id: i+1 for i, (id, _) in enumerate(tfidf_ranked)}
        
        for item in detailed:
            item['rank_embedding'] = embedding_ranks[item['resume_id']]
            item['rank_tfidf'] = tfidf_ranks[item['resume_id']]
        
        # Sort by embedding score
        detailed.sort(key=lambda x: x['embedding_score'], reverse=True)
        
        return detailed


def match_resumes(
    job_description_path: str,
    resumes_directory: str,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Convenience function to match resumes in one call.
    
    Args:
        job_description_path: Path to job description file
        resumes_directory: Path to directory with resume files
        output_path: Optional path to save results as JSON
        
    Returns:
        List of detailed results
    """
    matcher = ResumeMatcher()
    matcher.load_job_description(job_description_path)
    matcher.load_resumes(resumes_directory)
    
    results = matcher.get_detailed_results()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results
