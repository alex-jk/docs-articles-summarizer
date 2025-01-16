import streamlit as st
# from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import fitz  # PyMuPDF
import os
import re
import easyocr
import numpy as np
from PIL import Image
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import pandas as pd
import enchant

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

en_dict = enchant.Dict("en_US")

from transformers.utils.logging import set_verbosity_debug
# Load model
def load_model(debug=False):
    # model_directory = "t5-base"  # Using T5 for multilingual support
    # model = T5ForConditionalGeneration.from_pretrained(model_directory)
    # tokenizer = T5Tokenizer.from_pretrained(model_directory)

    if debug:
        set_verbosity_debug()

    model_name = "sshleifer/distilbart-cnn-12-6"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Preprocess text function
def preprocess_text(text):
    # Keep important punctuation marks: ., !, ?, ,, ; (and remove everything else)
    cleaned_text = re.sub(r'[^\w\s.,!?;_–-]', '', text)  # Keep ., !, ?, ,, ;
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing spaces
    return cleaned_text

# Summarize text function
def summarize_text(text, tokenizer, model, min_length, max_length, prompts=None, print_num_tokens=False):
    cleaned_text = preprocess_text(text)  # Preprocess the text

    tokenized_no_trunc = tokenizer(
        f"summarize: {cleaned_text}",
        return_tensors="pt",
        truncation=False
    )
    full_length = tokenized_no_trunc['input_ids'].shape[-1]
    print("Full length without truncation:", full_length)

    
    # Tokenize the input text for summarization
    tokenized_text = tokenizer.encode(
        f"summarize: {cleaned_text}", 
        return_tensors="pt", 
        max_length=1024,  
        truncation=True, 
        padding=True
    )

    # Print the number of tokens if requested
    if print_num_tokens:
        num_tokens = tokenized_text.shape[-1]  # tokenized_text is shape [batch, seq_length]
        print(f"Number of tokens in input text: {num_tokens}")
    
    # Generate the summary with adjusted parameters to reduce repetition
    summary_ids = model.generate(
        tokenized_text,
        max_length=max_length,  # Adjust max_length for longer or shorter summaries
        min_length=min_length,
        num_beams=4,  # Beam search to generate multiple candidates
        repetition_penalty=3.0,  # Higher penalty to avoid repetition
        early_stopping=False,  # Stop once the model generates a full sentence
        no_repeat_ngram_size=3
    )

    # Decode the generated tokens into the final summary text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Function to summarize in chunks
def split_text_into_chunks(text, tokenizer, max_length=1024):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Group sentences into chunks that fit within the token limit
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        # Tokenize just the new sentence
        sentence_tokens = tokenizer.encode(sentence, return_tensors="pt", truncation=False)

        # Check if adding this sentence will exceed the token limit
        if current_tokens + len(sentence_tokens[0]) > max_length:
            # If it exceeds the limit, finalize the current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence
            current_tokens = len(sentence_tokens[0])
        else:
            # If it fits, add the sentence to the current chunk
            current_chunk += " " + sentence
            current_tokens += len(sentence_tokens[0])

    # Add the last chunk if any sentences remain
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to detect valid summary part
def is_in_enchant_dict(word):
    word_lower = word.lower()
    # Check in PyEnchant, capitalized form, WordNet, or custom list
    return (
        en_dict.check(word_lower) or
        en_dict.check(word_lower.capitalize()) or
        bool(wordnet.synsets(word_lower))
    )

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to convert nltk's POS tags to WordNet's format
def get_wordnet_pos(treebank_tag, token):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        # Adjust for words ending in -ing misclassified as nouns
        if token.endswith("ing"):
            return wordnet.VERB
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unsure

def lemmatize_token(token, pos_tag):
    # Get WordNet POS tag and lemmatize
    wordnet_pos = get_wordnet_pos(pos_tag, token)
    lemma = lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
    return lemma

def is_valid_word(valid_words, token, pos_tag):

    if re.match(r'^\d+(\.\d+)?$', token):
        return True
    
    # Check if the token is a Roman numeral
    if re.match(r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', token):
        return True

    """Validate if token is a meaningful word or a valid hyphenated term."""
    # Handle hyphenated words by checking each part separately
    if '-' in token:
        parts = token.split('-')
        part_tags = nltk.pos_tag(parts)  # Tag each part separately
        return all(is_valid_word(valid_words, part, tag) for part, tag in part_tags)

    lemma = lemmatize_token(token, pos_tag)
    # print(f"\n lemma: {lemma}")
    
    # Validate as a single word
    return lemma in valid_words or re.match(r'^[.,!?;_–-]$', token)

contractions = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "mightn't": "might not",
    "mustn't": "must not",
    "n't": " not",  # catch-all for remaining "n't" forms
}

def expand_contractions(text):
    """Replace contractions in text with their expanded forms."""
    for contraction, replacement in contractions.items():
        text = re.sub(r'\b' + contraction + r'\b', replacement, text)
    return text

def create_features_for_token(token, medical_terms):
    token_df = pd.DataFrame({'word': [token]})
    
    # Apply the same feature creation logic
    token_df['contains_number'] = token_df['word'].str.contains(r'\d').astype(int)
    token_df['num_dashes'] = token_df['word'].str.count('-')
    token_df['num_vowels'] = token_df['word'].str.count(r'[aeiouAEIOU]')
    token_df['contains_non_alphanum'] = token_df['word'].str.contains(r'[^a-zA-Z0-9-]').astype(int)
    token_df['contains_ine'] = token_df['word'].str.contains(r'ine', case=False).astype(int)

    pos_tag = nltk.pos_tag([token])[0]
    token_df['lemma'] = lemmatize_token(pos_tag[0], pos_tag[1])
    token_df['is_real_word_enchant'] = token_df['lemma'].apply(is_in_enchant_dict).astype(int)

    token_df['num_characters'] = token_df['word'].apply(len)
    token_df['starts_with_letter'] = token_df['word'].apply(lambda x: int(bool(re.match(r'^[A-Za-z]', x))))
    token_df['ends_with_letter'] = token_df['word'].apply(lambda x: int(bool(re.search(r'[A-Za-z]$', x))))

    # Check for medical terms
    token_df['contains_medical_term'] = token_df['word'].apply(lambda word: any(term in word for term in medical_terms)).astype(int)

    # Return the feature row as a dictionary
    return token_df.iloc[0].to_dict()

def is_valid_word_with_model(token, pipeline, medical_terms):
    # Extract features for the token
    features = create_features_for_token(token, medical_terms)

    # Convert the features to a DataFrame
    df = pd.DataFrame([features])
    
    # (Optional) If your pipeline expects exactly these 10 columns, 
    # subset the DataFrame to those columns. Just *do not* call .values
    df = df[
        [
            'contains_number', 
            'num_dashes', 
            'num_vowels', 
            'contains_non_alphanum', 
            'contains_ine', 
            'is_real_word_enchant', 
            'num_characters', 
            'starts_with_letter', 
            'ends_with_letter', 
            'contains_medical_term'
        ]
    ]

    # Now pass the *DataFrame* (not a NumPy array) directly to the pipeline
    prediction = pipeline.predict(df)[0]  # logistic regression outputs 0 or 1
    return prediction == 1

# clean summary function
tokenizer_nltk = TreebankWordTokenizer()

whitelist = {"pdd", "dsm-5", "dsm-iii", "ssri", "eg", "benzodiazepine", "benzodiazepines", "worldwide", "cochrane", "prisma-nma"} 

def clean_summary(summary, pipeline, medical_terms, valid_words):

    expanded_summary = expand_contractions(summary)
    # Tokenize the summary into words
    # tokens = nltk.word_tokenize(summary)
    tokens = tokenizer_nltk.tokenize(expanded_summary)
    
    # Get the POS tags for the tokens
    pos_tags = nltk.pos_tag(tokens)

    cleaned_tokens = []
    for token, pos_tag in pos_tags:
        # Check if the lemmatized form of the token is valid
        # print(f"\n {token}")
        if token.lower() in whitelist or is_valid_word(valid_words, token, pos_tag) or is_valid_word_with_model(token, pipeline, medical_terms):
            cleaned_tokens.append(token)
        else:
            # Once gibberish or non-valid words appear, stop processing
            break
    
    # Join the valid tokens back into a string
    cleaned_summary = " ".join(cleaned_tokens).strip()
    
    # Ensure the summary ends with a full sentence
    if cleaned_summary and cleaned_summary[-1] not in ".!?":
        cleaned_summary += "."
    
    # If the summary is empty, return a fallback message or just an empty string
    if not cleaned_summary:
        cleaned_summary = "Summary could not be generated properly."

    return cleaned_summary

def summarize_long_text(text, min_length, max_length, prompts=None, clean_chunks=True):
    """
    Splits the input text into chunks (by full sentences), then summarizes each chunk 
    individually. Depending on `clean_chunks`, it either cleans each summary (with 
    `clean_summary`) or leaves it as is.
    
    Args:
        text (str): The text to be summarized.
        min_length (int): The minimum length for each chunk's summary.
        max_length (int): The maximum length for each chunk's summary.
        prompts (optional): Additional prompts or context for summarization.
        clean_chunks (bool): If True, clean each chunk's summary; else use raw summaries.

    Returns:
        str: The combined summary of all chunks.
    """
    # Split the text into chunks of full sentences
    chunks = split_text_into_chunks(text)

    # Summarize each chunk individually and collect the results
    summaries = []
    for chunk in chunks:
        summary = summarize_text(chunk, min_length, max_length, prompts)
        
        if clean_chunks:
            summary = clean_summary(summary)

        summaries.append(summary)
    
    # Combine the individual summaries into a final summary
    combined_summary = " ".join(summaries)

    # Ensure the final summary ends with a full sentence or punctuation
    if combined_summary and combined_summary[-1] not in ".!?":
        combined_summary = combined_summary.rsplit(" ", 1)[0] + "."

    return combined_summary

import pdfplumber

def read_pdf_with_pdfplumber(file):
    """Read and extract text from a PDF file using pdfplumber with positional data."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # Use `extract_words` to get word positions and spacing information
            words = page.extract_words()
            page_text = ""

            # Reconstruct text based on the word positions to handle missing spaces
            for word in words:
                # Use a space before the word if it's not the first word on the line
                page_text += f" {word['text']}"
            
            text += page_text + "\n"  # Add newline to separate each page's content
    return text

def read_pdf_by_page(file):
    """Read and extract text from a PDF file using pdfplumber, handling proper spacing between words."""
    pages_text = []  # Store text for each page separately

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words()  # Extract words with positional data
            page_text = ""

            # Variables to track previous word's position for proper spacing
            prev_x1 = 0  # End x-coordinate of the previous word
            prev_top = 0  # y-coordinate of the previous word's top position

            for word in words:
                x0, y0, x1, y1 = word['x0'], word['top'], word['x1'], word['bottom']
                word_text = word['text']

                # If there's a gap between words on the same line, insert a space
                if prev_x1 > 0 and (x0 - prev_x1) > 1 and abs(y0 - prev_top) < 5:
                    page_text += " " + word_text
                else:
                    page_text += word_text

                # Update previous word's x1 and top position for spacing logic
                prev_x1 = x1
                prev_top = y0

            # Print text for each page as it's extracted (optional)
            print(f"Extracted text for Page {page_num + 1}:\n", page_text, "\n" + "-" * 80)

            # Append extracted text for each page separately
            pages_text.append(page_text.strip())  # Strip leading/trailing spaces for each page

    return pages_text

def read_txt(file):
    return file.read().decode("cp1252", errors='replace')

def extract_relevant_sections(text, keyword="lamotrigine"):
    """Extract paragraphs or sentences containing the keyword from the text."""
    relevant_sections = []
    for paragraph in text.split('\n'):
        if keyword.lower() in paragraph.lower():
            relevant_sections.append(paragraph)
    return " ".join(relevant_sections)

def count_tokens(text, tokenizer):
    tokenized_text = tokenizer.encode(text, return_tensors="pt", truncation=False)
    return tokenized_text.shape[1]  # Returns the number of tokens

def get_non_identified_words(valid_words, file_content):
    cleaned_text = preprocess_text(file_content)  # Preprocess the text
    words = nltk.word_tokenize(cleaned_text)
    words = [word.lower() for word in words]

    # Tag each word with its part of speech
    word_pos_tags = nltk.pos_tag(words)

    # Collect words that are considered invalid
    invalid_words = {word for word, pos in word_pos_tags if not is_valid_word(valid_words, word, pos)}

    return list(invalid_words)