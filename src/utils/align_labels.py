def align_labels(tokenizer, sentences, labels_list, label_all_tokens=True):
    """
    Aligns labels with tokenized inputs for token-level tasks like NER.
    
    Args:
        tokenizer: Tokenizer used for encoding the sentences.
        sentences (list of str): Input sentences.
        labels_list (list of list of int): Corresponding labels for each word in sentences.
        label_all_tokens (bool): Whether to label all tokens or only the first subword.
    
    Returns:
        torch.Tensor: Aligned labels tensor.
    """
    from transformers import PreTrainedTokenizer
    import torch

    aligned_labels = []
    for sentence, labels in zip(sentences, labels_list):
        # Tokenize sentence and get word_ids
        encoding = tokenizer(sentence, return_offsets_mapping=True, truncation=True)
        word_ids = encoding.word_ids()
        
        aligned_label = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_label.append(-100)
            elif word_idx != previous_word_idx:
                aligned_label.append(labels[word_idx])
            else:
                aligned_label.append(labels[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        aligned_labels.append(aligned_label)

    # Find the maximum sequence length
    max_len = max(len(labels) for labels in aligned_labels)

    # Pad labels to the maximum length
    padded_labels = []
    for labels in aligned_labels:
        padding_length = max_len - len(labels)
        padded_labels.append(labels + [-100] * padding_length)

    return torch.tensor(padded_labels)
