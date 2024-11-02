
## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- **Docker**: (Optional) For containerized environments
- **CUDA/Metal**: (Optional) For GPU acceleration or apple metal gpu acceleration

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

    - **Ensure your `data.csv`** follows the structure outlined in `data/README.md`.
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
    cd processing_data
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
    Processed data saved to processing_data
    ```

    **Note:** Adjust the dataset size as needed. Ensure that `processing_data/` contains `train.csv`, `test.csv`, and `label_mappings.json`.

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
