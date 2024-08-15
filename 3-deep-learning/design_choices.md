# Design choices during model developemnt 

## 1. Choosing the model
The model choice depends on various factors like dataset size, complexity, and the nature of the problem. Here are some common options:

- **Random forest:** Random Forest is a popular ensemble learning method that can be applied to various types of classification tasks, including text classification.

- **Simple Dense Neural Network (DNN):** Suitable for relatively small and less complex datasets, especially when combined with feature extraction techniques like TF-IDF or Bag of Words (BoW). This approach is lightweight and fast.

- **Recurrent Neural Networks (RNNs), LSTMs, GRUs:** If your dataset is larger and the context of words (sequence) is important, RNN-based models are a good choice. LSTMs and GRUs help capture long-term dependencies in text.

- **Transformer Models (e.g., BERT, RoBERTa):** When you have a large dataset and want state-of-the-art performance, transformer models are highly effective. They understand context better by processing the entire sequence at once, allowing them to capture both long-range and bidirectional dependencies.

## 2. Choosing the Tokenizer
Tokenization is a critical step in text preprocessing. The choice depends on the model and the type of features you want to extract.

- **TF-IDF or Bag of Words (BoW):** Use a basic tokenizer that splits text into words and optionally applies stemming or lemmatization. The focus is on word frequency rather than sequence.

- **Subword Tokenizers (e.g., Byte-Pair Encoding, WordPiece):** Commonly used with transformer models like BERT. They handle out-of-vocabulary words better and help with languages that have many inflections.

- **Character-Level Tokenization:** Useful in cases where text contains a lot of misspellings or special characters. It’s more common in tasks like spelling correction or domain-specific tasks.

## 3. Choosing the Loss Metric

The choice of loss function depends on the nature of your labels and the output of your model.

- **Binary Classification (two classes):** Use nn.BCEWithLogitsLoss() if your model outputs logits for binary classification.

- **Multi-class Classification:** Use nn.CrossEntropyLoss(), which is suitable for problems with more than two classes. It expects raw logits as input and applies a softmax internally.

- **Regression Tasks:** Use nn.MSELoss() or nn.L1Loss() depending on whether you want to minimize squared or absolute errors.

## 4. Choosing the Optimizer

The optimizer controls how the model parameters are updated during training.

- **SGD (Stochastic Gradient Descent):** A basic optimizer that is often effective but may require careful tuning of the learning rate and momentum.

- **Adam (Adaptive Moment Estimation):** Generally a good default choice, as it adapts learning rates based on the first and second moments of the gradients. It’s widely used due to its robustness and efficiency.

- **RMSprop:** Effective in dealing with vanishing/exploding gradients and works well in RNNs.

- **AdamW:** A variation of Adam that decouples weight decay from the gradient update. It’s commonly used with transformer models.