"""
Sentence Transformers embedding-based resume matching module.

This module implements semantic matching using pre-trained Sentence Transformer
models. This approach understands meaning and can match:
- Synonyms ("software engineer" ~= "developer")
- Related concepts ("Python" relates to "programming")
- Paraphrases ("5 years experience" ~= "half decade of expertise")

Score of 1.0 = identical meaning, 0.0 = completely unrelated

Model Choice:
- all-mpnet-base-v2: Best quality for semantic similarity
- all-MiniLM-L6-v2: Faster, smaller, slightly lower quality
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingMatcher:
    """
    Semantic embedding-based resume matcher.
    
    This class uses pre-trained transformer models to create semantic embeddings
    of job descriptions and resumes, then compares them using cosine similarity.
    
    Attributes:
        model: SentenceTransformer model instance
        model_name: Name of the loaded model
        job_description_embedding: Embedding vector for the current job description
    """
    
    RECOMMENDED_MODELS = {
        'high_quality': 'all-mpnet-base-v2',
        'balanced': 'all-MiniLM-L6-v2',
        'fast': 'paraphrase-MiniLM-L3-v2',
    }
    
    def __init__(
        self,
        model_name: str = 'all-mpnet-base-v2',
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Name of the Sentence Transformer model to use.
                       See RECOMMENDED_MODELS for options.
            device: Device to run the model on ('cpu', 'cuda', 'mps').
                   If None, will be auto-detected.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.job_description_embedding = None
        self._job_description_text = None
    
    def set_job_description(self, job_description: str) -> 'EmbeddingMatcher':
        """
        Set and encode the job description.
        
        Args:
            job_description: The job description text (should be preprocessed)
            
        Returns:
            self for method chaining
        """
        self._job_description_text = job_description
        self.job_description_embedding = self.model.encode(
            job_description,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return self
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    
    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors.
        
        More efficient than encoding one at a time.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show a progress bar
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
    
    def score_resume(self, resume: str) -> float:
        """
        Calculate the similarity score for a single resume.
        
        Args:
            resume: Resume text (should be preprocessed)
            
        Returns:
            Similarity score between 0.0 and 1.0
            
        Raises:
            ValueError: If set_job_description() hasn't been called
        """
        if self.job_description_embedding is None:
            raise ValueError(
                "Job description not set. Call set_job_description() first."
            )
        
        # Encode the resume
        resume_embedding = self.encode(resume)
        
        # Calculate cosine similarity
        # Since we normalized embeddings, dot product = cosine similarity
        similarity = np.dot(self.job_description_embedding, resume_embedding)
        
        # Ensure score is in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))
    
    def score_resumes(
        self,
        resumes: List[str],
        show_progress: bool = False,
    ) -> List[float]:
        """
        Calculate similarity scores for multiple resumes efficiently.
        
        Uses batch encoding for better performance.
        
        Args:
            resumes: List of resume texts
            show_progress: Whether to show a progress bar
            
        Returns:
            List of similarity scores
        """
        if self.job_description_embedding is None:
            raise ValueError(
                "Job description not set. Call set_job_description() first."
            )
        
        # Batch encode all resumes
        resume_embeddings = self.encode_batch(resumes, show_progress)
        
        # Calculate similarities using matrix multiplication
        # job_description_embedding: (embedding_dim,)
        # resume_embeddings: (num_resumes, embedding_dim)
        similarities = np.dot(resume_embeddings, self.job_description_embedding)
        
        # Clip to [0, 1] range
        similarities = np.clip(similarities, 0.0, 1.0)
        
        return similarities.tolist()
    
    def rank_resumes(
        self,
        resumes: Dict[str, str],
        show_progress: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Rank resumes by their semantic similarity to the job description.
        
        Args:
            resumes: Dictionary mapping resume identifiers to resume texts
            show_progress: Whether to show a progress bar
            
        Returns:
            List of (identifier, score) tuples sorted by score descending
        """
        if self.job_description_embedding is None:
            raise ValueError(
                "Job description not set. Call set_job_description() first."
            )
        
        identifiers = list(resumes.keys())
        texts = list(resumes.values())
        
        scores = self.score_resumes(texts, show_progress)
        
        # Combine with identifiers and sort
        results = list(zip(identifiers, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer dimension (e.g., 768 for all-mpnet-base-v2)
        """
        return self.model.get_sentence_embedding_dimension()
    
    def explain_similarity(
        self,
        text1: str,
        text2: str,
    ) -> Dict[str, Union[float, int]]:
        """
        Get detailed similarity information between two texts.
        
        Useful for understanding why two texts are considered similar or different.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with similarity metrics
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity (dot product of normalized vectors)
        cosine_sim = float(np.dot(emb1, emb2))
        
        # Euclidean distance
        euclidean_dist = float(np.linalg.norm(emb1 - emb2))
        
        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'embedding_dimension': len(emb1),
            'model': self.model_name,
        }


def create_embedding_matcher(
    job_description: str,
    model_name: str = 'all-mpnet-base-v2',
    device: Optional[str] = None,
) -> EmbeddingMatcher:
    """
    Convenience function to create an embedding matcher with a job description set.
    
    Args:
        job_description: Job description text
        model_name: Sentence Transformer model name
        device: Device to run on (None for auto-detect)
        
    Returns:
        EmbeddingMatcher instance with job description set
    """
    matcher = EmbeddingMatcher(model_name=model_name, device=device)
    matcher.set_job_description(job_description)
    return matcher


# Utility function for quick comparison
def compute_similarity(text1: str, text2: str, model_name: str = 'all-mpnet-base-v2') -> float:
    """
    Compute semantic similarity between two texts.
    
    This is a convenience function for one-off comparisons.
    For batch processing, use EmbeddingMatcher directly.
    
    Args:
        text1: First text
        text2: Second text
        model_name: Model to use
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode([text1, text2], normalize_embeddings=True)
    similarity = np.dot(embeddings[0], embeddings[1])
    return float(max(0.0, min(1.0, similarity)))
