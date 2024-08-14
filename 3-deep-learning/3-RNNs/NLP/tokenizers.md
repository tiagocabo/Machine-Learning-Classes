# Tokenizers
Tokenizers are fundamental in Natural Language Processing (NLP) because they serve as the bridge between raw text data and the computational models that process this data. Here's why tokenizers are crucial:

## 1. Breaking Down Text into Manageable Units
**Text to Tokens:** Raw text is a continuous stream of characters, which needs to be divided into meaningful units for further processing. Tokenizers break down the text into smaller, manageable units called tokens. These tokens can be words, subwords, characters, or even sentences, depending on the type of tokenizer used.
**Granularity Control:** Tokenization allows for different levels of text representation. For instance, word-level tokenization is straightforward for many languages, but for morphologically rich languages or when handling unknown words, subword tokenization (like Byte Pair Encoding) is more effective.

## 2. Handling Vocabulary Size and Efficiency
- **Vocabulary Management:** Tokenizers help in defining the vocabulary used by an NLP model. By choosing an appropriate tokenization method, you can control the size of the vocabulary, which directly impacts the model's efficiency and memory usage.
- **OOV Handling:** Tokenizers are crucial for managing out-of-vocabulary (OOV) words. By breaking words into subwords or character-level tokens, tokenizers can reduce the occurrence of OOV words, enabling the model to handle rare or unseen words effectively.

## 3. Contextual Understanding
- **Capturing Meaning:** The way text is tokenized influences how well the model can capture the meaning and context of words or phrases. For example, tokenizers that recognize subwords can capture the meaning of prefixes or suffixes, enhancing the model’s ability to understand context.
- **Maintaining Semantic Integrity:** Poor tokenization can break down meaningful units inappropriately, leading to a loss of semantic integrity. Good tokenization preserves the meaning of text while making it easier for models to process.

## 4. Preprocessing for Consistency
- **Standardization:** Tokenizers often perform additional preprocessing tasks like lowercasing, removing punctuation, or handling special characters, ensuring that the text is in a consistent format before being fed into the model.
- **Normalization:** This is especially important for handling different forms of the same word (e.g., "playing" vs. "played") or different spellings (e.g., "color" vs. "colour").

## 5. Improving Model Training
- **Training Efficiency:** By reducing the size of the vocabulary and breaking text into tokens that are more meaningful, tokenizers can significantly improve the efficiency of model training. Models learn faster and generalize better when they are fed well-tokenized input.
- **Data Augmentation:** Tokenizers can also help in creating augmented versions of the dataset by breaking down text in various ways, providing the model with a richer set of training data.

## 6. Adapting to Different Languages
- **Language-Specific Needs:** Different languages have different tokenization requirements. For example, in languages like Chinese or Japanese, tokenization is more challenging because words are not always separated by spaces. A good tokenizer can adapt to the linguistic properties of the language being processed.
- **Multilingual Models:** For multilingual models, tokenizers need to be designed to handle multiple languages simultaneously, often requiring sophisticated algorithms that can tokenize text effectively across language boundaries.

## 7. Compatibility with Downstream Tasks
- **Task-Specific Tokenization:** Depending on the downstream NLP task (like translation, sentiment analysis, or question answering), the choice of tokenizer can vary. Some tasks might require finer granularity, while others might benefit from a coarser tokenization.
- **Alignment with Model Architecture:** Certain models, like BERT or GPT, require specific types of tokenization that are aligned with their architecture (e.g., WordPiece or Byte-Pair Encoding). Proper tokenization ensures compatibility and maximizes the model's performance.

## Conclusion
Tokenizers are essential in NLP because they prepare raw text for processing by breaking it into appropriate units, managing vocabulary size, preserving semantic meaning, and ensuring efficient model training. The quality of tokenization directly impacts the effectiveness of NLP models, making it a critical step in the NLP pipeline.

# BPE - Byte Pair encoding
## Introduction
Byte Pair Encoding (BPE) is a data compression technique originally used in file compression, but it has gained significant attention in the field of Natural Language Processing (NLP), particularly for tokenization in models like GPT and other transformer-based architectures.

## How BPE Works
1. **Start with the Character Level:**
BPE starts by treating each character in the input text as its own token. For example, the word "hello" would initially be represented as ['h', 'e', 'l', 'l', 'o'].

2. **Find the Most Frequent Pair:**
The algorithm then identifies the most frequent pair of consecutive tokens. In the word "hello", for example, the pair "l" and "l" would likely be identified as a frequent pair because they appear together often.

3. **Merge the Pair:**
Once the most frequent pair is found, the algorithm merges these two tokens into a new single token. So, "l" and "l" would be combined into "ll", transforming the sequence ['h', 'e', 'l', 'l', 'o'] into ['h', 'e', 'll', 'o'].

4. **Repeat the Process:**
The process of finding the most frequent pair and merging it is repeated. This is done until the desired vocabulary size is reached or until no more pairs are left to merge. For instance, if "he" became a frequent pair, the sequence might become ['he', 'll', 'o'].


# Application in NLP
In NLP, BPE is used for tokenizing text, particularly in situations where it's beneficial to break down words into subword units. This is especially important for handling rare words or different forms of the same word (like "playing", "played", "plays") efficiently.

For example:

**Handling Out-of-Vocabulary Words:** In traditional tokenization, a word not seen during tra- ining would be completely unknown to the model. BPE helps break down such words into familiar subword tokens. For instance, the word "unhappiness" might be tokenized into ["un", "happiness"].

- **Efficiency:** By using subword tokens, BPE can reduce the overall vocabulary size needed, which improves the efficiency and memory usage of models.

- **Generalization:** Since BPE captures common subwords, it allows models to generalize better to variations of words by recognizing the same subword in different contexts.

**Sample implementation**

```python
from collections import defaultdict, Counter

def get_vocab(corpus):
    """Creates a vocabulary from the corpus with each word split into characters."""
    vocab = defaultdict(int)
    for word in corpus:
        word = ' '.join(list(word)) + ' </w>'  # Add word boundary symbol
        vocab[word] += 1
    return vocab

def get_stats(vocab):
    """Calculates the frequency of each pair of characters in the vocabulary."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merges the most frequent pair of characters in the vocabulary."""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe(corpus, num_merges):
    """Performs BPE on the given corpus for a specified number of merges."""
    vocab = get_vocab(corpus)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f'Step {i+1}: {best}')
        print("Updated Vocabulary:", vocab)
    return vocab

# Example usage
corpus = ["low", "lower", "lowest"]
num_merges = 1

vocab = bpe(corpus, num_merges)
```

## Interesting links
- https://www.youtube.com/watch?v=HEikzVL-lZU&t=118s
- https://www.youtube.com/watch?v=zduSFxRajkE