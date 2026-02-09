"""
TF-IDF based resume matching module.

This module implements a baseline matching approach using Term Frequency-Inverse
Document Frequency (TF-IDF) vectors and cosine similarity. This serves as a
comparison point to demonstrate the advantages of semantic embeddings.
"""

from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:
    """
    TF-IDF based resume matcher.
    
    This class provides a baseline matching approach that uses TF-IDF vectors
    to compare job descriptions with resumes.
    
    Attributes:
        vectorizer: sklearn TfidfVectorizer instance
        job_description_vector: TF-IDF vector for the current job description
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
    ):
        """
        Initialize the TF-IDF matcher.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to consider (1,2) means unigrams and bigrams
            min_df: Minimum document frequency for a term to be included
            max_df: Maximum document frequency (as a proportion) for a term
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',  # Remove common English stop words
        )
        self.job_description_vector = None
        self._is_fitted = False
    
    def fit(self, job_description: str, resumes: List[str]) -> 'TFIDFMatcher':
        """
        Fit the vectorizer on the job description and resumes.
        
        The vectorizer needs to see all documents to build its vocabulary
        and calculate IDF weights.
        
        Args:
            job_description: The job description text
            resumes: List of resume texts
            
        Returns:
            self for method chaining
        """
        # Combine all documents for fitting
        all_documents = [job_description] + resumes
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(all_documents)
        
        self.job_description_vector = tfidf_matrix[0:1]
        
        self._is_fitted = True
        return self
    
    def score_resume(self, resume: str) -> float:
        """
        Calculate the similarity score for a single resume.
        
        Args:
            resume: Preprocessed resume text
            
        Returns:
            Similarity score between 0.0 and 1.0
            
        Raises:
            ValueError: If fit() hasn't been called
        """
        if not self._is_fitted:
            raise ValueError("Matcher must be fitted before scoring. Call fit() first.")
        
        # Transform the resume using the fitted vectorizer
        resume_vector = self.vectorizer.transform([resume])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(self.job_description_vector, resume_vector)[0][0]
        
        # Ensure score is in [0, 1] range (cosine similarity can sometimes be slightly negative)
        return float(max(0.0, min(1.0, similarity)))
    
    def score_resumes(self, resumes: List[str]) -> List[float]:
        """
        Calculate similarity scores for multiple resumes.
        
        Args:
            resumes: List of resume texts
            
        Returns:
            List of similarity scores
        """
        return [self.score_resume(resume) for resume in resumes]
    
    def rank_resumes(
        self,
        resumes: Dict[str, str],
    ) -> List[Tuple[str, float]]:
        """
        Rank resumes by their similarity to the job description.
        
        Args:
            resumes: Dictionary mapping resume identifiers to resume texts
            
        Returns:
            List of (identifier, score) tuples sorted by score descending
        """
        if not self._is_fitted:
            raise ValueError("Matcher must be fitted before ranking. Call fit() first.")
        
        results = []
        for identifier, text in resumes.items():
            score = self.score_resume(text)
            results.append((identifier, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get the top n features (terms) from the vocabulary.
        
        Useful for understanding what terms the model considers important.
        
        Args:
            n: Number of features to return
            
        Returns:
            List of feature names
        """
        if not self._is_fitted:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        return list(feature_names[:n])
    
    def get_matching_terms(self, resume: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top matching terms between job description and resume.
        
        This helps explain why a resume received a particular score.
        
        Args:
            resume: Resume text
            n: Number of terms to return
            
        Returns:
            List of (term, importance) tuples
        """
        if not self._is_fitted:
            return []
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Transform resume
        resume_vector = self.vectorizer.transform([resume]).toarray()[0]
        jd_vector = self.job_description_vector.toarray()[0]
        
        # Find terms that are present in both
        matching_terms = []
        for i, (jd_weight, resume_weight) in enumerate(zip(jd_vector, resume_vector)):
            if jd_weight > 0 and resume_weight > 0:
                # Use the minimum as the "match strength"
                match_strength = min(jd_weight, resume_weight)
                matching_terms.append((feature_names[i], match_strength))
        
        # Sort by match strength and return top n
        matching_terms.sort(key=lambda x: x[1], reverse=True)
        return matching_terms[:n]


def create_tfidf_matcher(
    job_description: str,
    resumes: List[str],
    **kwargs
) -> TFIDFMatcher:
    """
    Convenience function to create and fit a TF-IDF matcher.
    
    Args:
        job_description: Job description text
        resumes: List of resume texts
        **kwargs: Additional arguments to pass to TFIDFMatcher
        
    Returns:
        Fitted TFIDFMatcher instance
    """
    matcher = TFIDFMatcher(**kwargs)
    matcher.fit(job_description, resumes)
    return matcher
