# Import necessary libraries
import pandas as pd
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
        cleaned_english_sent = ' '.join(
            english_sent.split())  # Remove extra spaces
        cleaned_telugu_sent = ' '.join(
            telugu_sent.split())  # Remove extra spaces
        cleaned_aligned_sentences.append(
            (cleaned_english_sent, cleaned_telugu_sent))
    return cleaned_aligned_sentences


def get_phrases(sentence, max_length):
    """
    Extracts phrases from a sentence up to a maximum length.
    Args:
        sentence (str): The input sentence
        max_length (int): The maximum length of phrases
    Returns:
        phrases (list): A list of phrases
    """
    words = sentence.split()
    phrases = []
    for i in range(len(words)):
        for j in range(i+1, min(len(words)+1, i+max_length+1)):
            phrase = ' '.join(words[i:j])
            phrases.append(phrase)
    return phrases


def align_phrases(english_phrases, telugu_phrases):
    """
    Aligns English and Telugu phrases using a simple phrase-by-phrase alignment approach.
    Args:
        english_phrases (list): List of English phrases
        telugu_phrases (list): List of Telugu phrases
    Returns:
        aligned_phrases (list): List of aligned phrases in the format [(english_phrase, telugu_phrase), ...]
    """
    aligned_phrases = []
    for i in range(len(english_phrases)):
        english_phrase = english_phrases[i]
        for j in range(len(telugu_phrases)):
            telugu_phrase = telugu_phrases[j]
            # if english_phrase == telugu_phrase:
            aligned_phrases.append((english_phrase, telugu_phrase))
    return aligned_phrases


def clean_aligned_phrases(aligned_phrases):
    """
    Cleans aligned phrases by removing empty alignments and special characters.
    Args:
        aligned_phrases (list): List of aligned phrases
    Returns:
        cleaned_aligned_phrases (list): List of cleaned aligned phrases
    """
    cleaned_aligned_phrases = []
    for english_phrase, telugu_phrase in aligned_phrases:
        cleaned_english_phrase = ' '.join(
            english_phrase.split())  # Remove extra spaces
        cleaned_telugu_phrase = ' '.join(
            telugu_phrase.split())  # Remove extra spaces
        cleaned_aligned_phrases.append(
            (cleaned_english_phrase, cleaned_telugu_phrase))
    return cleaned_aligned_phrases


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
    # translation_model[(0, 5)] = 0.0
    for english_sent, telugu_sent in aligned_sentences:
        english_phrases = get_phrases(english_sent, max_length=5)
        # print(english_phrases)
        telugu_phrases = get_phrases(telugu_sent, max_length=5)
        # print(telugu_phrases)
        aligned_phrases = align_phrases(english_phrases, telugu_phrases)
        for english_phrase, telugu_phrase in aligned_phrases:
            english_words = english_phrase.split()
            telugu_words = telugu_phrase.split()
            for english_index in range(len(english_words)):
                for telugu_index in range(len(telugu_words)):
                    translation_model[(english_index, telugu_index)] = 1.0 / \
                        (len(english_words) * len(telugu_words))

    # Run IBM Model 4 training algorithm
    for iteration in range(num_iterations):
        count_e_f = {}
        count_e = {}
        count_j_i_l_m = {}
        count_i_l_m = {}

        for english_sent, telugu_sent in aligned_sentences:
            aligned_phrases = align_phrases(english_phrases, telugu_phrases)

            for english_phrase, telugu_phrase in aligned_phrases:
                english_words = english_phrase.split()
                telugu_words = telugu_phrase.split()
                l = len(english_words)
                m = len(telugu_words)
                for j in range(m):
                    for i in range(l):
                        # Calculate normalization factor
                        total = 0
                        for telugu_index in range(len(telugu_words)):
                            total += translation_model[(i, telugu_index)]

                        # Calculate delta values
                        delta = {}
                        for telugu_index in range(len(telugu_words)):
                            delta[telugu_index] = translation_model[(
                                i, telugu_index)] / total

                        # Update counts
                        for telugu_index in range(len(telugu_words)):
                            count_e_f[(i, telugu_index)] = count_e_f.get(
                                (i, telugu_index), 0) + delta[telugu_index]
                            count_e[i] = count_e.get(
                                i, 0) + delta[telugu_index]
                            count_j_i_l_m[(j, i, l, m)] = count_j_i_l_m.get(
                                (j, i, l, m), 0) + delta[telugu_index]
                            count_i_l_m[(i, l, m)] = count_i_l_m.get(
                                (i, l, m), 0) + delta[telugu_index]

        # Update translation probabilities
        for english_index in translation_model:
                telugu_index = translation_model[english_index]
                english_phrases = get_phrases(english_sent, max_length=5)
                telugu_phrases = get_phrases(telugu_sent, max_length=5)
                # Align the phrases using the IBM Model 4 algorithm
        alignments = align_phrases(english_phrases, telugu_phrases)

        # Update the translation model using the phrase alignments
        for e_phrase, t_phrase in alignments:
            if e_phrase not in translation_model:
                translation_model[e_phrase] = {}
            if t_phrase not in translation_model[e_phrase]:
                translation_model[e_phrase][t_phrase] = 1
            else:
                translation_model[e_phrase][t_phrase] += 1

    return translation_model


def train_language_model(english_sentences):
    """
    Trains a language model using English sentences.
    Args:
        english_sentences (list): List of English sentences
    Returns:
        language_model (dict): Dictionary representing the language model
    """
    language_model = {}
    total_words = 0
    for sentence in english_sentences:
        words = sentence.split()
        for word in words:
            if word not in language_model:
                language_model[word] = 0
            language_model[word] += 1
            total_words += 1
    # Convert word counts to probabilities
    for word in language_model:
        language_model[word] /= total_words
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
                    best_translation = max(
                        translation_scores, key=translation_scores.get)
                    decoded_sent.append(best_translation)
                else:
                    if input_word in language_model:
                        language_scores = language_model[input_word]
                        best_word = max(language_scores,
                                        key=language_scores.get)
                        decoded_sent.append(best_word)
                    else:
                        # Use a small default value for unknown words
                        decoded_sent.append('<unk>')
        decoded_sentences.append(' '.join(decoded_sent))
    return decoded_sentences

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
            bleu = sentence_bleu(
                [actual_sent], predicted_sent, weights=weights)
            if bleu > max_bleu:
                max_bleu = bleu
        bleu_scores.append(max_bleu)
    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score


# Step 1: Corpus Preparation
# Read dataset
df = pd.read_csv("pmindia.v1.te-en.csv")
# df = pd.read_csv("BLEU_REFS.csv")

# Extract English and Telugu sentences
english_sentences = df["english"].tolist()
# english_sentences = tokenize_lower_remove_punctuation(english_sentences)
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
language_model = train_language_model(telugu_sentences)
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

