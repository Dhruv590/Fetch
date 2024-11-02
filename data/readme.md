# Data Directory

This directory contains the datasets required for training and evaluating the multi-task sentence transformer.

## Dataset Structure

- `your_data.csv`: A CSV file containing sentences along with labels for Sentiment and Category.

## CSV File Format

The CSV file should have the following columns:

- `Sentence`: The input sentence.
- `Sentiment`: Label for Sentiment Analysis (e.g., Positive, Negative).
- `Category`: Label for Sentence Classification (e.g., Technical Issues, Positive Sentiment).

### Example

```csv
Sentence,Sentiment,Category
"So there is no way for me to plug it in here in the US unless I go by a converter.",Negative,Technical Issues
"Good case, Excellent value.",Positive,Positive Sentiment
...
