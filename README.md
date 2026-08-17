News Article NLP Analyzer

Project Overview

This project is a Natural Language Processing (NLP) based News Article Analyzer developed as part of an Information Retrieval (IR) project.

The system processes news articles and predicts their category using NLP techniques and machine learning.

Dataset

The project uses a news article dataset containing:

- Article title
- Article body
- Article category

The dataset is included in this repository as a ZIP file.

Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression

NLP Techniques

The project performs the following preprocessing steps:

1. Text cleaning
2. Tokenization
3. Stop-word removal
4. Stemming
5. TF-IDF feature extraction

Machine Learning

The processed text is converted into TF-IDF features and classified using a Logistic Regression model.

The dataset is divided into:

- 80% training data
- 20% testing data

The model is evaluated using accuracy and a classification report.

Query Analysis

The system allows the user to enter a news-related query and performs:

- Tokenization
- Stop-word identification
- Stop-word removal
- Stemming
- TF-IDF analysis
- News category prediction
- Prediction confidence

Project Files

- "news_article_nlp_analyzer.py" — Main Python source code
- "new article category.zip" — News article dataset
- "output.pdf" — Project output and results

How to Run

Install the required Python libraries:

pip install pandas nltk scikit-learn

Then run:

python news_article_nlp_analyzer.py

The program will process the dataset, train the model, and allow the user to enter news-related queries for classification.

Output

The project generates NLP preprocessing results, TF-IDF analysis, model accuracy, classification results, and predicted news categories.
