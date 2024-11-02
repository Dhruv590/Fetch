
## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- **Docker**: (Optional) For containerized environments
- **CUDA**: (Optional) For GPU acceleration

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/dhruv590/multi-task-sentence-transformer.git
    cd multi-task-sentence-transformer
    ```

2. **Set Up a Virtual Environment**

    It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

### Data Preparation

1. **Navigate to the Data Directory**

    ```bash
    cd data
    ```

2. **Prepare Your Dataset**

    - **Ensure your `your_data.csv`** follows the structure outlined in `data/README.md`.
    - **Example:**
    
        ```csv
        Sentence,Sentiment,Category
        "I absolutely love this product!",Positive,Positive Sentiment
        "The battery life is too short.",Negative,Technical Issues
        ...
        ```

3. **Run the Data Preparation Script**

    Navigate back to the project root and execute the script:

    ```bash
    cd ..
    python prepare_dataset.py
    ```

    **Output:**

    ```
    Data Loaded Successfully
    Labels Mapped to Integers
    Sentiment Mapping: {'Negative': 0, 'Positive': 1}
    Category Mapping: {'Technical Issues': 0, 'Positive Sentiment': 1, 'Negative Sentiment': 2, 'Purchase or Value Concerns': 3}
    Data Split into Training and Testing Sets
    Training Set Size: 800
    Testing Set Size: 200
    Processed data saved to processed_data
    ```

    **Note:** Adjust the dataset size as needed. Ensure that `processed_data/` contains `train.csv`, `test.csv`, and `label_mappings.json`.

### Training the Model

1. **Navigate to the Training Directory**

    ```bash
    cd src/training
    ```

2. **Run the Training Script**

    ```bash
    python train.py
    ```

    **Monitoring Training:**

    You'll see output similar to:

    ```
    Epoch 1/3
    ----------
    Training: 100%|██████████| 1000/1000 [01:40<00:00,  9.67it/s]
    Average Training Loss: 0.6931
    Validation: 100%|██████████| 200/200 [00:30<00:00,  6.56it/s]
    Average Validation Loss: 0.6928

    ...
    Model saved to ../models/multitask_mpnet
    ```

    **Note:** With a small synthetic dataset, loss values may not reflect meaningful learning. Use a larger and more diverse dataset for better results.

### Inference and Exploration

1. **Navigate to the Notebooks Directory**

    ```bash
    cd ../../notebooks
    ```

2. **Open and Run the Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

    Open `exploration.ipynb` and execute the cells to perform inference using the trained model.

    **Sample Output:**

    ```
    Sentence: "I absolutely love this product!"
    Category: Positive Sentiment
    Sentiment: Positive

    Sentence: "The battery life is too short."
    Category: Technical Issues
    Sentiment: Negative

    ...
    ```

### Docker Setup (Optional)

To ensure a reproducible environment, you can run the project inside a Docker container.

1. **Build the Docker Image**

    ```bash
    docker build -t multi-task-transformer .
    ```

2. **Run the Docker Container**

    ```bash
    docker run -it --rm -p 8888:8888 multi-task-transformer
    ```

    **Access Jupyter Notebook:**

    - Open a browser and navigate to `http://localhost:8888`.
    - Use the token provided in the terminal to access the notebook.

### Layer-wise Learning Rate Optimization (Bonus Task)

The `train.py` script already incorporates layer-wise learning rates, allowing different parts of the model to learn at different speeds. This approach helps in preserving pre-trained knowledge in the transformer backbone while allowing task-specific heads to adapt quickly.

**Rationale:**

- **Early Layers:** Capture general language features. Assigning lower learning rates ensures these features remain stable.
- **Later Layers and Heads:** More task-specific. Assigning higher learning rates allows them to adapt to new tasks effectively.

---

## **Additional Scripts and Utilities**

### **a. `src/utils/align_labels.py`**

Already provided above.

### **b. `src/models/multitask_mpnet.py`**

Already provided above.

### **c. `src/training/train.py`**

Already provided above.

---

## **13. Troubleshooting**

### **a. `NameError: name 'sentiment_mapping' is not defined`**

**Cause:**

- The `save_data` function was attempting to access `sentiment_mapping` and `category_mapping` without receiving them as parameters.

**Solution:**

- Ensure that the `save_data` function accepts `sentiment_mapping` and `category_mapping` as parameters.
- Update the `main` function to pass these mappings to `save_data`.

**Reference:**

See the corrected `prepare_dataset.py` script provided above.

### **b. Other Common Issues**

1. **No GPU Detected**

    - **Symptom:** Slow training and inference.
    - **Solution:** Ensure that PyTorch is installed with CUDA support and that GPU drivers are correctly set up.

    ```python
    import torch
    print(torch.cuda.is_available())  # Should return True if GPU is available
    ```

2. **Out of Memory Errors**

    - **Symptom:** Training script fails due to insufficient GPU memory.
    - **Solutions:**
      - Reduce the `BATCH_SIZE` in `train.py`.
      - Use gradient accumulation to simulate larger batches.
      - Optimize the model by using mixed precision training with `torch.cuda.amp`.

3. **Incorrect Label Alignment**

    - **Symptom:** NER predictions are misaligned with tokens.
    - **Solutions:**
      - Verify that the `align_labels` function correctly aligns labels with tokenized inputs.
      - Ensure that labels are provided as lists matching the number of words in each sentence.
      - Check if `label_all_tokens` parameter in `align_labels` is set appropriately.

4. **Training Not Converging**

    - **Symptom:** Loss does not decrease over epochs.
    - **Solutions:**
      - Check learning rates. Ensure that layer-wise learning rates are set correctly.
      - Increase the number of training epochs.
      - Use a larger and more diverse training dataset.
      - Implement learning rate schedulers for better convergence.

5. **Unexpected Inference Results**

    - **Symptom:** Sentences are not classified correctly, or NER entities are misidentified.
    - **Solutions:**
      - Ensure that the model has been trained sufficiently.
      - Use a more substantial dataset for training.
      - Fine-tune hyperparameters like learning rates, batch sizes, and number of epochs.

---

## **14. Final Notes**

- **Data Security:** Ensure that any sensitive data is handled appropriately and not included in the repository.
- **Model Saving:** The training script saves the trained model and tokenizer for future use.
- **Extensibility:** The architecture allows for adding more tasks by introducing additional classification heads.
- **Hyperparameter Tuning:** Adjust learning rates, batch sizes, and other hyperparameters based on your specific dataset and requirements.
- **Evaluation Metrics:** Implement evaluation metrics like accuracy for Task A and F1-score for Task B to assess model performance effectively.

---

If you encounter any further issues or need additional assistance with specific parts of the project, feel free to ask!
