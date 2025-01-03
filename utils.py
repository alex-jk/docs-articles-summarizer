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
def summarize_text(text, min_length, max_length, prompts=None, print_num_tokens=False):
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
def split_text_into_chunks(text, max_length=1024):
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
        return all(is_valid_word(part, tag) for part, tag in part_tags)

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
    
    # Convert the features to the expected model input
    feature_array = pd.DataFrame([features])[[
        'contains_number', 'num_dashes', 'num_vowels', 
        'contains_non_alphanum', 'contains_ine', 
        'is_real_word_enchant', 'num_characters', 
        'starts_with_letter', 'ends_with_letter',
        'contains_medical_term'
    ]].values
    
    # Predict using the logistic regression model
    return pipeline.predict(feature_array)[0] == 1  # Assuming 1 means valid