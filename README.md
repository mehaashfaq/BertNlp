# Twitter Sentiment Analysis using BERT and PyQt5

## 📌 Project Overview

This project is a desktop-based Twitter Sentiment Analysis application developed using **BERT (Bidirectional Encoder Representations from Transformers)** and **PyQt5**. The application classifies tweets into four sentiment categories:

- Positive
- Negative
- Neutral
- Irrelevant

The project demonstrates the complete NLP pipeline, including data preprocessing, model fine-tuning, evaluation, and deployment through an interactive graphical user interface (GUI).

---

## Features

- Fine-tuned BERT model for sentiment classification
- Interactive PyQt5 desktop application
- Load a trained BERT model
- Load Twitter dataset from CSV
- Predict sentiment of selected tweets
- Manual text sentiment prediction
- Confidence score and probability distribution
- Clean and user-friendly interface

---

## Technologies Used

- Python 3.x
- PyTorch
- Hugging Face Transformers
- Datasets
- Scikit-learn
- Pandas
- NumPy
- PyQt5
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset

Dataset used:

**Twitter Entity Sentiment Analysis**

File:

```
dataset/twitter_training.csv
```

---

## Project Structure

```
Twitter-Sentiment-Analysis-BERT/
│
├── app.py
├── train_bert.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── twitter_training.csv
│
├── saved_bert_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│   └── label_encoder.pkl
│
└── screenshots/
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Twitter-Sentiment-Analysis-BERT.git
```

Move into the project directory

```bash
cd Twitter-Sentiment-Analysis-BERT
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Open the notebook:

```
train_bert.ipynb
```

The notebook performs:

- Data Loading
- Data Cleaning
- Label Encoding
- Train/Test Split
- Tokenization
- Fine-tuning BERT
- Model Evaluation
- Saving the trained model

The trained model is saved inside:

```
saved_bert_model/
```

---

## Running the Application

Run the desktop application using:

```bash
python app.py
```

---

## How to Use

1. Launch the application.
2. Click **Load Model** and select the `saved_bert_model` folder.
3. Click **Load Dataset** and select `twitter_training.csv`.
4. Select any tweet from the dataset or type your own text.
5. Click **Predict**.
6. View the predicted sentiment and confidence score.

---

## Model Output

The model predicts one of the following classes:

- Positive
- Negative
- Neutral
- Irrelevant

---

## Dependencies

Main libraries used:

- torch
- transformers
- datasets
- evaluate
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pyqt5
- joblib

---

## Results

The fine-tuned BERT model successfully classifies tweets into four sentiment categories using contextual language understanding provided by transformer-based architectures.

---

## Future Improvements

- Deploy as a web application using Streamlit or Flask.
- Add drag-and-drop dataset support.
- Visualize prediction statistics.
- Support batch predictions.
- Add model comparison with RoBERTa and DistilBERT.

---

## Author

**Meha Ashfaq**

BS Computer Science

---

## License

This project is developed for educational purposes.
