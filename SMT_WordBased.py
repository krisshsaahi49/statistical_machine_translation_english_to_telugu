# Import necessary libraries
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu

def align_sentences(english_sentences, telugu_sentences):
    """
    Aligns English and Telugu sentences using a simple word-by-word alignment approach.
    Args:
        english_sentences (list): List of English sentences
        telugu_sentences (list): List of Telugu sentences
    Returns:
        aligned_sentences (list): List of aligned sentences in the format [(english_sent, telugu_sent), ...]
    """
    aligned_sentences = []
    for i in range(len(english_sentences)):
        english_sent = english_sentences[i]
        telugu_sent = telugu_sentences[i]
        if len(english_sent.split()) == len(telugu_sent.split()):
            aligned_sentences.append((english_sent, telugu_sent))
    return aligned_sentences


def clean_aligned_sentences(aligned_sentences):
    """
    Cleans aligned sentences by removing empty alignments and special characters.
    Args:
        aligned_sentences (list): List of aligned sentences
    Returns:
        cleaned_aligned_sentences (list): List of cleaned aligned sentences
    """
    cleaned_aligned_sentences = []
    for english_sent, telugu_sent in aligned_sentences:
        cleaned_english_sent = ' '.join(english_sent.split())  # Remove extra spaces
        cleaned_telugu_sent = ' '.join(telugu_sent.split())  # Remove extra spaces
        cleaned_aligned_sentences.append((cleaned_english_sent, cleaned_telugu_sent))
    return cleaned_aligned_sentences


def train_translation_model(aligned_sentences, num_iterations=5):
    """
    Trains a statistical machine translation model using aligned sentences and IBM Model 4.
    Args:
        aligned_sentences (list): List of aligned sentences in the format [(english_sent, telugu_sent), ...]
        num_iterations (int): Number of iterations to run the training algorithm (default=5)
    Returns:
        translation_model (dict): Dictionary representing the translation model
    """
    # Initialize translation probabilities uniformly
    translation_model = {}
    for english_sent, telugu_sent in aligned_sentences:
        english_words = english_sent.split()
        telugu_words = telugu_sent.split()
        for english_word in english_words:
            if english_word not in translation_model:
                translation_model[english_word] = {}
            for telugu_word in telugu_words:
                if telugu_word not in translation_model[english_word]:
                    translation_model[english_word][telugu_word] = 1.0 / len(telugu_words)

    # Run IBM Model 4 training algorithm
    for iteration in range(num_iterations):
        count_e_f = {}
        count_e = {}
        count_j_i_l_m = {}
        count_i_l_m = {}

        for english_sent, telugu_sent in aligned_sentences:
            english_words = english_sent.split()
            telugu_words = telugu_sent.split()
            l = len(english_words)
            m = len(telugu_words)
            for j in range(m):
                for i in range(l):
                    # Calculate normalization factor
                    total = 0
                    for telugu_word in telugu_words:
                        total += translation_model[english_words[i]][telugu_word]

                    # Calculate delta values
                    delta = {}
                    for telugu_word in telugu_words:
                        delta[telugu_word] = translation_model[english_words[i]][telugu_word] / total

                    # Update counts
                    for telugu_word in telugu_words:
                        count_e_f[(english_words[i], telugu_word)] = count_e_f.get((english_words[i], telugu_word), 0) + delta[telugu_word]
                        count_e[english_words[i]] = count_e.get(english_words[i], 0) + delta[telugu_word]
                        count_j_i_l_m[(j, i, l, m)] = count_j_i_l_m.get((j, i, l, m), 0) + delta[telugu_word]
                        count_i_l_m[(i, l, m)] = count_i_l_m.get((i, l, m), 0) + delta[telugu_word]

        # Update translation probabilities
        for english_word in translation_model:
            for telugu_word in translation_model[english_word]:
                count = count_e_f.get((english_word, telugu_word), 0)
                total = count_e.get(english_word, 0)
                translation_model[english_word][telugu_word] = count / total

        # Update distortion probabilities
        for j, i, l, m in count_j_i_l_m:
            count = count_j_i_l_m.get((j, i, l, m), 0)
            total = count_i_l_m.get((i, l, m), 0)
            distortion_prob = count / total
            distortion_prob = max(distortion_prob, 1e-12) # prevent underflow
            distortion_table = {}
            distortion_table[(j, i, l, m)] = distortion_prob

    return translation_model


# def train_language_model(english_sentences):
#     """
#     Trains a language model using English sentences.
#     Args:
#         english_sentences (list): List of English sentences
#     Returns:
#         language_model (dict): Dictionary representing the language model
#     """
#     language_model = {}
#     total_words = 0
#     for sentence in english_sentences:
#         words = sentence.split()
#         for word in words:
#             if word not in language_model:
#                 language_model[word] = 0
#             language_model[word] += 1
#             total_words += 1
#     # Convert word counts to probabilities
#     for word in language_model:
#         language_model[word] /= total_words
#     return language_model

def train_language_model(telugu_sentences,n):
    """
    Trains a language model using Telugu sentences.
    Args:
        telugu_sentences (list): List of Telugu sentences
    Returns:
        language_model (dict): Dictionary representing the language model
    """
    language_model = {}
    total_ngrams = 0
    for sentence in telugu_sentences:
        words = sentence.split()
        for i in range(len(words)-n+1):
            ngram = ' '.join(words[i:i+n])
            if ngram not in language_model:
                language_model[ngram] = 0
            language_model[ngram] += 1
            total_ngrams += 1
    # Convert n-gram counts to probabilities
    for ngram in language_model:
        language_model[ngram] /= total_ngrams
    return language_model

def train_distortion_table(aligned_sentences):
    """
    Trains a distortion table using aligned sentences.
    Args:
        aligned_sentences (list): List of aligned sentences in the format [(english_sent, telugu_sent), ...]
    Returns:
        distortion_table (dict): Dictionary representing the distortion table
    """
    distortion_table = {}
    for english_sent, telugu_sent in aligned_sentences:
        english_words = english_sent.split()
        telugu_words = telugu_sent.split()
        for i in range(len(english_words)):
            english_word = english_words[i]
            telugu_word = telugu_words[i]
            if english_word not in distortion_table:
                distortion_table[english_word] = {}
            if telugu_word not in distortion_table[english_word]:
                distortion_table[english_word][telugu_word] = 0
            distortion_table[english_word][telugu_word] += 1
    return distortion_table

def train_phrase_table(aligned_sentences):
    """
    Trains a phrase table using aligned sentences.
    Args:
        aligned_sentences (list): List of aligned sentences in the format [(english_sent, telugu_sent), ...]
    Returns:
        phrase_table (dict): Dictionary representing the phrase table
    """
    phrase_table = {}
    for english_sent, telugu_sent in aligned_sentences:
        english_phrases = english_sent.split()
        telugu_phrases = telugu_sent.split()
        for i in range(len(english_phrases)):
            english_phrase = english_phrases[i]
            telugu_phrase = telugu_phrases[i]
            if english_phrase not in phrase_table:
                phrase_table[english_phrase] = {}
            if telugu_phrase not in phrase_table[english_phrase]:
                phrase_table[english_phrase][telugu_phrase] = 0
            phrase_table[english_phrase][telugu_phrase] += 1
    return phrase_table


def decode_sentences(input_sentences, phrase_table, translation_model, language_model, distortion_table):
    """
    Decodes input sentences using the statistical machine translation model.
    Args:
        input_sentences (list): List of input sentences
        phrase_table (dict): Dictionary representing the phrase table
        translation_model (dict): Dictionary representing the translation model
        language_model (dict): Dictionary representing the language model
        distortion_table (dict): Dictionary representing the distortion table
    Returns:
        decoded_sentences (list): List of decoded sentences
    """
    decoded_sentences = []
    for input_sent in input_sentences:
        input_words = input_sent.split()
        decoded_sent = []
        for i in range(len(input_words)):
            input_word = input_words[i]
            if input_word in phrase_table:
                phrase_scores = phrase_table[input_word]
                best_phrase = max(phrase_scores, key=phrase_scores.get)
                decoded_sent.append(best_phrase)
            else:
                if input_word in translation_model:
                    translation_scores = translation_model[input_word]
                    best_translation = max(translation_scores, key=translation_scores.get)
                    decoded_sent.append(best_translation)
                else:
                    if input_word in language_model:
                        language_scores = language_model[input_word]
                        best_word = max(language_scores, key=language_scores.get)
                        decoded_sent.append(best_word)
                    else:
                        # Use a small default value for unknown words
                        decoded_sent.append('<unk>') 
        decoded_sentences.append(' '.join(decoded_sent))
    return decoded_sentences


def tokenize_lower_remove_punctuation(sentences):
    """
    Tokenize, lowercase, and remove punctuation from sentences
    """
    tokenized_sentences = []
    for sentence in sentences:
        # Tokenize sentence
        tokens = nltk.word_tokenize(sentence)

        # Lowercase tokens
        tokens = [token.lower() for token in tokens]

        # Remove punctuation from tokens
        tokens = [token for token in tokens if token.isalnum()]

        # Join tokens to form sentence
        sentence = " ".join(tokens)
        tokenized_sentences.append(sentence)

    return tokenized_sentences

def calculate_bleu_score(actual_sentences, predicted_sentences):
    """
    Calculates the BLEU score between the actual and predicted sentences.
    Args:
        actual_sentences (list): List of actual sentences
        predicted_sentences (list): List of predicted sentences
    Returns:
        bleu_score (float): The BLEU score between the actual and predicted sentences
    """
    bleu_scores = []
    for predicted_sent in predicted_sentences:
        # Calculate BLEU score for n-grams up to 2
        weights = [(1.0 / n) for n in range(1, 2)]
        max_bleu = 0
        for actual_sent in actual_sentences:
            actual_sent = actual_sent.split()
            predicted_sent = ' '.join(predicted_sent)
            # predicted_sent = predicted_sent.split()
            bleu = sentence_bleu([actual_sent], predicted_sent, weights=weights)
            if bleu > max_bleu:
                max_bleu = bleu
        bleu_scores.append(max_bleu)
    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score


# Step 1: Corpus Preparation
# Read dataset
# df = pd.read_csv("pmindia.v1.te-en.csv")
df = pd.read_csv("BLEU_REFS.csv")

# Extract English and Telugu sentences
english_sentences = df["english"].tolist()
telugu_sentences = df["telugu"].tolist()

# Convert english_sentences and telugu_sentences to lists of strings if necessary
english_sentences = list(map(str, english_sentences))
telugu_sentences = list(map(str, telugu_sentences))

# Call the align_sentences function with the updated variables
aligned_sentences = align_sentences(english_sentences, telugu_sentences)

# Clean aligned sentences
cleaned_aligned_sentences = clean_aligned_sentences(aligned_sentences)

# Step 3: Training
# Train statistical machine translation models
translation_model = train_translation_model(cleaned_aligned_sentences)
language_model = train_language_model(telugu_sentences,2)
distortion_table = train_distortion_table(aligned_sentences)
phrase_table = train_phrase_table(aligned_sentences)

# Read the first 5 reference translations from BLEU_REFS.csv
df = pd.read_csv('BLEU_REFS.csv', usecols=['english', 'telugu'])
input_sentences = df['english'][:5].tolist()
ref_translations = df['telugu'][:5].tolist()

# Step 4: Decoding
# Implement decoding algorithm
decoded_telugu_sentences = decode_sentences(input_sentences, phrase_table, translation_model, language_model, distortion_table)

# Print BLEU scores for the first 5 translations
for i in range(5):
    bleu_score = calculate_bleu_score(ref_translations[i], decoded_telugu_sentences[i])
    print("Source sentences :", input_sentences[i])
    print("Translated sentences : ",decoded_telugu_sentences[i])
    print(f"BLEU score for sentence {i+1}: {bleu_score}")

