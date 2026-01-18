# answer_extractor.py
"""
Answer extraction and choice disambiguation for AF-Think dataset.

This module contains functions to:
1. Clean question text by removing boilerplate instructions
2. Extract multiple choice options from questions
3. Analyze answers to determine the correct choice
4. Format answers with the extracted choice appended

The extraction uses multiple strategies:
- Explicit mentions: "Hence, (B) is correct", "The answer is (A)"
- Elimination patterns: "Options A, B, D do not align" → C is correct
- Text matching: Matching choice text and paraphrases in the answer
- LLM-based extraction: Use a language model to analyze reasoning (optional)
"""
import re
from typing import Dict, Optional, Callable, Any


def clean_question(question: str) -> str:
    """
    Clean up the question text by removing boilerplate instructions.
    
    Args:
        question: The raw question text
        
    Returns:
        Cleaned question text
    """
    # Remove the "Please think and reason..." instruction at the end
    # Variations: "input audio", "input music", etc.
    patterns_to_remove = [
        "Please think and reason about the input audio before you respond.",
        "Please think and reason about the input music before you respond.",
        "Please think and reason about the input sound before you respond.",
    ]
    
    for pattern in patterns_to_remove:
        question = question.replace(pattern, "")
    
    # Clean up extra whitespace
    question = " ".join(question.split()).strip()
    
    return question


def extract_choices_from_question(question: str) -> Dict[str, str]:
    """
    Extract multiple choice options from the question text.
    
    Args:
        question: The question text containing choices like (A), (B), etc.
    
    Returns:
        Dictionary mapping choice letters to their text, e.g.:
        {"A": "Engaging in a playful activity", "B": "Recounting a humorous anecdote", ...}
    """
    choices = {}
    
    # First, remove any trailing instructions that might interfere
    # (in case clean_question wasn't called yet)
    question = re.sub(r'Please think and reason.*$', '', question, flags=re.IGNORECASE)
    
    # Pattern to match choices like (A), (B), (C), (D) followed by text
    # Handles both period-terminated and space-separated choices
    pattern = r'\(([A-Z])\)\s*([^(]+?)(?=\s*\([A-Z]\)|$)'
    
    matches = re.findall(pattern, question)
    
    for letter, text in matches:
        # Clean up the choice text
        text = text.strip().rstrip('.').strip()
        # Remove any remaining instruction text
        text = re.sub(r'Please think.*$', '', text, flags=re.IGNORECASE).strip()
        if text:
            choices[letter] = text
    
    return choices


# ============================================================================
# Strategy 1: Explicit Answer Patterns
# ============================================================================

# Patterns that explicitly state which answer is correct
EXPLICIT_PATTERNS = [
    r'correct (?:answer|option|choice) is \(([A-Z])\)',
    r'answer is \(([A-Z])\)',
    r'\(([A-Z])\) is (?:the )?correct',
    r'option ([A-Z]) is (?:the )?correct',
    r'(?:hence|therefore|thus|so),?\s*\(([A-Z])\)\s*is',  # "Hence, (B) is..."
    r'(?:hence|therefore|thus|so),?\s*option\s*\(([A-Z])\)',  # "Hence, option (B)..."
    r'(?:hence|therefore|thus|so),?\s*([A-Z])\s*is\s*(?:the\s*)?(?:correct|most|best)',  # "Hence, B is correct"
    r'making\s*\(([A-Z])\)\s*(?:the\s*)?(?:correct|best|most)',  # "making (B) the correct"
    r'points?\s*(?:to|towards?)\s*\(([A-Z])\)',  # "points to (B)"
    r'best\s*(?:answer|option|choice)\s*is\s*\(([A-Z])\)',  # "best answer is (B)"
]


def find_explicit_answer(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Look for explicit mentions of the correct answer.
    
    Examples:
        - "The correct answer is (B)"
        - "Hence, (B) is the most accurate"
        - "Option B is correct"
    """
    for pattern in EXPLICIT_PATTERNS:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in choices:
                return f"Answer: ({letter}) {choices[letter]}."
    return None


# ============================================================================
# Strategy 2: Elimination Patterns
# ============================================================================

ELIMINATION_PATTERNS = [
    r'options?\s+([A-Z](?:,\s*[A-Z])*(?:\s*,?\s*and\s+[A-Z])?)\s+(?:do not|don\'t|are not|aren\'t|describe scenarios that do not)',
    r'options?\s+([A-Z](?:,\s*[A-Z])*(?:\s*,?\s*and\s+[A-Z])?)\s+(?:can be ruled out|are ruled out|are eliminated)',
]


def find_answer_by_elimination(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Find the correct answer by identifying eliminated options.
    
    Examples:
        - "Options A, B, and D do not align" → C is correct
        - "Options A, C can be ruled out" → B or D (if only 2 remaining)
    """
    answer_lower = answer.lower()
    
    for pattern in ELIMINATION_PATTERNS:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            eliminated_str = match.group(1).upper()
            eliminated = set(re.findall(r'[A-Z]', eliminated_str))
            remaining = set(choices.keys()) - eliminated
            if len(remaining) == 1:
                letter = remaining.pop()
                return f"Answer: ({letter}) {choices[letter]}."
    
    # "The other options are not supported" pattern
    other_options_pattern = r'(?:the\s+)?other\s+options?\s+(?:are|is)\s+(?:not|n\'t)'
    if re.search(other_options_pattern, answer, re.IGNORECASE):
        # Try to find the one option that IS supported
        for letter in choices.keys():
            choice_text = choices[letter].lower()
            key_words = [w for w in choice_text.split() if len(w) > 4]
            matches = sum(1 for w in key_words if w in answer_lower)
            if matches >= 2:
                return f"Answer: ({letter}) {choices[letter]}."
    
    return None


# ============================================================================
# Strategy 3: Text Matching and Scoring
# ============================================================================

# Common stop words to filter out
STOP_WORDS = {'the', 'a', 'an', 'of', 'on', 'in', 'to', 'for', 'with', 'and', 'or', 'is', 'are', 'was', 'were'}

# Key concepts and their variants for semantic matching
KEY_CONCEPTS = {
    'campaign': ['campaign event', 'political campaign', 'campaign gathering'],
    'product': ['promoting a product', 'product promotion', 'endorse or sell'],
    'competitive': ['competitive event', 'competition', 'race'],
    'commentary': ['commentating', 'commentary on', 'observation and commentary'],
    'weather': ['weather forecast', 'weather update', 'weather report'],
    'announcement': ['announcing', 'event announcement'],
    'promotion': ['promoting', 'product promotion', 'endorse'],
    'recap': ['daily recap', 'summarizing', 'summary'],
}

# Key nouns for "involving X" patterns
KEY_NOUNS = [
    'children', 'equipment', 'feedback', 'incident', 'failure', 'content',
    'audience', 'recording', 'campaign', 'product', 'event', 'weather',
    'competitive', 'commentary'
]

# Alignment phrases template
ALIGNMENT_PHRASES = [
    "aligns most closely with the idea of {choice}",
    "aligns best with the purpose of {choice}",
    "aligns best with {choice}",
    "aligns with {choice}",
    "fitting the description of {choice}",
    "description of {choice}",
    "suggests {choice}",
    "indicates {choice}",
    "points to {choice}",
    "consistent with {choice}",
]


def score_choice_match(choice_text: str, answer_lower: str) -> int:
    """
    Score how well a choice matches the answer text.
    
    Args:
        choice_text: The text of a choice option
        answer_lower: The answer text (lowercase)
        
    Returns:
        Integer score (higher = better match)
    """
    choice_lower = choice_text.lower()
    score = 0
    
    # Extract key content words from choice
    choice_words = set(w for w in re.findall(r'\b\w{3,}\b', choice_lower) if w not in STOP_WORDS)
    answer_words = set(re.findall(r'\b\w{3,}\b', answer_lower))
    
    # Count matching significant words
    matching_words = choice_words & answer_words
    score += len(matching_words) * 2
    
    # Check alignment phrases
    for phrase_template in ALIGNMENT_PHRASES:
        phrase = phrase_template.format(choice=choice_lower)
        if phrase in answer_lower:
            score += 10
    
    # Check key part of choice (first 3 words)
    choice_key_phrase = ' '.join(choice_lower.split()[0:3])
    if len(choice_key_phrase) > 5 and choice_key_phrase in answer_lower:
        score += 5
    
    # Check for "involving X" patterns
    for noun in KEY_NOUNS:
        if noun in choice_lower:
            if f"involving the {noun}" in answer_lower or f"involving {noun}" in answer_lower:
                score += 8
            if f"happening involving the {noun}" in answer_lower:
                score += 10
    
    # Check for exact choice text
    if choice_lower in answer_lower:
        score += 15
    
    # Check key concept matches
    for concept, variants in KEY_CONCEPTS.items():
        if concept in choice_lower:
            for variant in variants:
                if variant in answer_lower:
                    score += 6
    
    # Check for "indicates a X" pattern
    indicates_match = re.search(r'indicates\s+(?:a\s+)?(.{10,50}?)(?:\.|,|$)', answer_lower)
    if indicates_match:
        indicated_text = indicates_match.group(1)
        indicated_words = set(re.findall(r'\b\w{4,}\b', indicated_text))
        overlap = indicated_words & choice_words
        if len(overlap) >= 2:
            score += 8
    
    # Check for "aligns best with the purpose of X" pattern
    aligns_match = re.search(r'aligns\s+best\s+with\s+(?:the\s+purpose\s+of\s+)?(.{5,40}?)(?:\.|,|rather|$)', answer_lower)
    if aligns_match:
        aligned_text = aligns_match.group(1)
        aligned_words = set(re.findall(r'\b\w{4,}\b', aligned_text))
        overlap = aligned_words & choice_words
        if len(overlap) >= 1:
            score += 10
    
    return score


def find_answer_by_text_matching(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Find the correct answer by matching choice text in the answer.
    
    Uses scoring to find the best matching choice based on:
    - Word overlap
    - Alignment phrases
    - Key concept matches
    - Paraphrase detection
    """
    answer_lower = answer.lower()
    best_match = None
    best_score = 0
    
    for letter, choice_text in choices.items():
        score = score_choice_match(choice_text, answer_lower)
        
        if score > best_score:
            best_score = score
            best_match = letter
    
    if best_match and best_score >= 4:
        return f"Answer: ({best_match}) {choices[best_match]}."
    
    return None


# ============================================================================
# Strategy 4: LLM-based Extraction
# ============================================================================

# Prompt template for LLM extraction
LLM_EXTRACTION_PROMPT = """Given a multiple choice question and a reasoning explanation, determine which choice is correct.

Question:
{question}

Choices:
{choices_text}

Reasoning/Explanation:
{reasoning}

Based on the reasoning above, which choice (A, B, C, D, etc.) is the correct answer?
Respond with ONLY the letter of the correct choice, nothing else."""


def find_answer_with_llm(
    question: str,
    answer: str,
    choices: Dict[str, str],
    llm_fn: Callable[[str], str],
) -> Optional[str]:
    """
    Use an LLM to determine the correct answer from the reasoning.
    
    Args:
        question: The question text
        answer: The reasoning/explanation text
        choices: Dictionary of choice letters to text
        llm_fn: Function that takes a prompt string and returns the LLM response
        
    Returns:
        Formatted answer string or None if extraction failed
    """
    # Format choices for the prompt
    choices_text = "\n".join([f"({letter}) {text}" for letter, text in sorted(choices.items())])
    
    # Create the prompt
    prompt = LLM_EXTRACTION_PROMPT.format(
        question=question,
        choices_text=choices_text,
        reasoning=answer,
    )
    
    try:
        # Call the LLM
        response = llm_fn(prompt).strip().upper()
        
        # Extract the letter from the response
        # Handle responses like "A", "(A)", "A.", "The answer is A", etc.
        letter_match = re.search(r'\(?([A-Z])\)?', response)
        if letter_match:
            letter = letter_match.group(1)
            if letter in choices:
                return f"Answer: ({letter}) {choices[letter]}."
    except Exception as e:
        print(f"LLM extraction failed: {e}")
    
    return None


def create_hf_llm_fn(model_path: str, device: str = "cuda"):
    """
    Create an LLM function using HuggingFace transformers.
    
    Args:
        model_path: Path to the model (local or HuggingFace hub)
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Function that takes a prompt and returns the response
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    def llm_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # We only need a letter
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    return llm_fn


def create_vicuna_llm_fn(device: str = "cuda"):
    """
    Create an LLM function using the pretrained Vicuna model from SALMONN.
    
    Uses the model downloaded by recipes/download_pretrained.sh:
    pretrained/vicuna-13b-v1.1
    
    Args:
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Function that takes a prompt and returns the response
        
    Example:
        llm_fn = create_vicuna_llm_fn(device="cuda")
        set_llm_extractor(llm_fn)
    """
    from pathlib import Path
    
    # Find the project root (where pretrained/ directory is)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    # Path to Vicuna model
    vicuna_path = project_root / "pretrained" / "vicuna-13b-v1.1"
    
    if not vicuna_path.exists():
        raise FileNotFoundError(
            f"Vicuna model not found at {vicuna_path}\n"
            f"Please run: bash recipes/download_pretrained.sh"
        )
    
    return create_hf_llm_fn(str(vicuna_path), device=device)


# Global LLM function (set via set_llm_extractor)
_llm_fn: Optional[Callable[[str], str]] = None


def set_llm_extractor(llm_fn: Optional[Callable[[str], str]]):
    """
    Set the LLM function to use for answer extraction.
    
    Args:
        llm_fn: Function that takes a prompt string and returns the response,
                or None to disable LLM extraction
                
    Example:
        # Using the pretrained Vicuna model from SALMONN
        llm_fn = create_vicuna_llm_fn(device="cuda")
        set_llm_extractor(llm_fn)
        
        # Using any HuggingFace model
        llm_fn = create_hf_llm_fn("lmsys/vicuna-7b-v1.5")
        set_llm_extractor(llm_fn)
        
        # Using a custom function
        def my_llm(prompt: str) -> str:
            return my_model.generate(prompt)
        set_llm_extractor(my_llm)
    """
    global _llm_fn
    _llm_fn = llm_fn


def get_llm_extractor() -> Optional[Callable[[str], str]]:
    """Get the current LLM extractor function."""
    return _llm_fn


# ============================================================================
# Main API
# ============================================================================

def find_correct_answer(
    question: str,
    answer: str,
    use_llm_fallback: bool = True,
) -> Optional[str]:
    """
    Analyze the answer text to determine which choice is correct.
    
    Tries multiple strategies in order:
    1. Explicit mentions (e.g., "Hence, (B) is correct")
    2. Elimination patterns (e.g., "Options A, B, D do not align")
    3. Text matching and scoring
    4. LLM-based extraction (if enabled and set up)
    
    Args:
        question: The question text containing choices
        answer: The answer text to analyze
        use_llm_fallback: Whether to use LLM as fallback if other methods fail
        
    Returns:
        Formatted string like "Answer: (B) Energetic call." or None if not found
    """
    # Extract choices from the question
    choices = extract_choices_from_question(question)
    
    if not choices:
        # Not a multiple choice question
        return None
    
    # Strategy 1: Explicit mentions
    result = find_explicit_answer(answer, choices)
    if result:
        return result
    
    # Strategy 2: Elimination patterns
    result = find_answer_by_elimination(answer, choices)
    if result:
        return result
    
    # Strategy 3: Text matching
    result = find_answer_by_text_matching(answer, choices)
    if result:
        return result
    
    # Strategy 4: LLM-based extraction (fallback)
    if use_llm_fallback and _llm_fn is not None:
        result = find_answer_with_llm(question, answer, choices, _llm_fn)
        if result:
            return result
    
    return None


def format_answer_with_choice(question: str, answer: str) -> str:
    """
    Append the extracted correct answer choice to the answer text.
    
    Args:
        question: The question text
        answer: The original answer text
        
    Returns:
        Answer text with "Answer: (X) Choice text." appended if found,
        otherwise the original answer
    """
    correct_answer = find_correct_answer(question, answer)
    
    if correct_answer:
        # Ensure answer ends with proper punctuation before appending
        answer = answer.rstrip()
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        return f"{answer} {correct_answer}"
    
    return answer
