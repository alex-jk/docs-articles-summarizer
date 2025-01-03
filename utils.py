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