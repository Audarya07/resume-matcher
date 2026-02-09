"""
Text preprocessing module for resume and job description processing.

This module provides functions to clean, normalize, and prepare text data
for the matching engine. The preprocessing preserves meaningful
technical terms while removing noise.
"""

import re
from pathlib import Path
from typing import Dict, Union


def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load text content from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        String content of the file
        
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def clean_text(text: str) -> str:
    """
    Clean and normalize text while preserving technical terms.
    
    This function performs the following operations:
    1. Converts to lowercase for consistency
    2. Normalizes whitespace (multiple spaces, tabs, newlines)
    3. Removes special characters but preserves alphanumeric, spaces, and common punctuation
    4. Preserves technical terms like "C++", "C#", ".NET", version numbers
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Preserve common technical terms before cleaning
    # Map special terms to placeholders
    tech_term_map = {
        'c++': 'cplusplus',
        'c#': 'csharp',
        '.net': 'dotnet',
        'node.js': 'nodejs',
        'vue.js': 'vuejs',
        'react.js': 'reactjs',
        'next.js': 'nextjs',
        'express.js': 'expressjs',
    }
    
    for term, placeholder in tech_term_map.items():
        text = text.replace(term, placeholder)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove phone numbers (various formats)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    
    # Keep alphanumeric characters, spaces, and some punctuation
    # Preserve: letters, numbers, spaces, hyphens, slashes, periods (for versions)
    text = re.sub(r'[^\w\s\-/.]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_sections(text: str) -> Dict[str, str]:
    """
    Attempt to extract common resume sections.
    
    This is a simple heuristic-based approach that identifies common
    section headers and extracts content between them.
    
    Args:
        text: Resume text
        
    Returns:
        Dictionary mapping section names to their content
    """
    # Common section headers (lowercase)
    section_patterns = [
        r'(?:professional\s+)?summary',
        r'(?:work\s+)?experience',
        r'(?:professional\s+)?experience',
        r'employment(?:\s+history)?',
        r'(?:technical\s+)?skills',
        r'education',
        r'certifications?',
        r'projects?',
        r'(?:about\s+)?me',
        r'objective',
        r'qualifications',
    ]
    
    # Build regex pattern to find section headers
    pattern = r'^(' + '|'.join(section_patterns) + r')\s*$'
    
    sections = {}
    current_section = 'header'
    current_content = []
    
    for line in text.split('\n'):
        line_lower = line.lower().strip()
        
        # Check if this line is a section header
        if re.match(pattern, line_lower, re.IGNORECASE):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            
            # Start new section
            current_section = line_lower
            current_content = []
        else:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def preprocess_for_matching(text: str, preserve_structure: bool = False) -> str:
    """
    Main preprocessing function that prepares text for the matching engine.
    
    Args:
        text: Raw text (resume or job description)
        preserve_structure: If True, attempts to preserve section information
        
    Returns:
        Preprocessed text ready for embedding or TF-IDF
    """
    if not text:
        return ""
    
    if preserve_structure:
        # Extract sections and create weighted text
        sections = extract_sections(text)
        
        # Weight important sections by repeating them
        # Skills and experience sections are typically more important for matching
        weighted_parts = []
        
        for section_name, content in sections.items():
            cleaned = clean_text(content)
            if not cleaned:
                continue
                
            # Give more weight to skills-related sections
            if 'skill' in section_name:
                weighted_parts.extend([cleaned] * 2)  # Double weight
            elif 'experience' in section_name or 'employment' in section_name:
                weighted_parts.extend([cleaned] * 2)  # Double weight
            else:
                weighted_parts.append(cleaned)
        
        return ' '.join(weighted_parts)
    else:
        # Simple cleaning without section weighting
        return clean_text(text)


def load_and_preprocess(file_path: Union[str, Path], preserve_structure: bool = False) -> str:
    """
    Convenience function to load a file and preprocess its content.
    
    Args:
        file_path: Path to the text file
        preserve_structure: Whether to preserve section structure
        
    Returns:
        Preprocessed text content
    """
    raw_text = load_text_file(file_path)
    return preprocess_for_matching(raw_text, preserve_structure)


# Utility functions for batch processing

def load_resumes_from_directory(directory: Union[str, Path]) -> Dict[str, str]:
    """
    Load all resume files from a directory.
    
    Args:
        directory: Path to directory containing resume files
        
    Returns:
        Dictionary mapping filename to raw text content
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    resumes = {}
    for file_path in sorted(directory.glob('*.txt')):
        resumes[file_path.name] = load_text_file(file_path)
    
    return resumes


def preprocess_resumes(resumes: Dict[str, str], preserve_structure: bool = False) -> Dict[str, str]:
    """
    Preprocess a batch of resumes.
    
    Args:
        resumes: Dictionary mapping filename to raw text
        preserve_structure: Whether to preserve section structure
        
    Returns:
        Dictionary mapping filename to preprocessed text
    """
    return {
        filename: preprocess_for_matching(text, preserve_structure)
        for filename, text in resumes.items()
    }
