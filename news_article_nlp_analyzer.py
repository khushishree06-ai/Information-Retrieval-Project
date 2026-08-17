# ============================================================
# NEWS ARTICLE NLP ANALYZER
# ============================================================

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ============================================================
# DOWNLOAD NLTK DATA
# ============================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("       LOADING NEWS DATASET")
print("==========================================")

df = pd.read_csv(
    "/content/news-article-categories.csv",
    engine="python",
    on_bad_lines="skip"
)

print("\nDataset loaded successfully!")
print("Total articles:", len(df))


# ============================================================
# SHOW COLUMNS
# ============================================================

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df["title"] = df["title"].fillna("")
df["body"] = df["body"].fillna("")
df["category"] = df["category"].fillna("Unknown")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=["title", "body"]
).reset_index(drop=True)

after = len(df)

print("\nDuplicate articles removed:", before - after)


# ============================================================
# COMBINE TITLE + BODY
# ============================================================

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["body"].astype(str)
)


# ============================================================
# SHOW CATEGORY DISTRIBUTION
# ============================================================

print("\n==========================================")
print("      NEWS CATEGORY DISTRIBUTION")
print("==========================================")

print(df["category"].value_counts())


# ============================================================
# NLP TOOLS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove numbers and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Tokenization
    tokens = word_tokenize(text)

    # Stop word removal
    filtered = [
        word
        for word in tokens
        if word not in stop_words
        and word.isalpha()
    ]

    # Stemming
    stemmed = [
        stemmer.stem(word)
        for word in filtered
    ]

    return " ".join(stemmed)


# ============================================================
# PREPROCESS DATASET
# ============================================================

print("\n==========================================")
print("        NLP PREPROCESSING")
print("==========================================")

print("Tokenization + Stop Word Removal + Stemming")

df["processed_text"] = df["text"].apply(preprocess)

print("Preprocessing completed!")


# ============================================================
# SHOW PREPROCESSING EXAMPLE
# ============================================================

print("\n==========================================")
print("       PREPROCESSING EXAMPLE")
print("==========================================")

print("\nORIGINAL ARTICLE:")

print(df["text"].iloc[0][:1000])

print("\nPROCESSED ARTICLE:")

print(df["processed_text"].iloc[0][:1000])


# ============================================================
# INPUT AND OUTPUT
# ============================================================

X = df["processed_text"]

y = df["category"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\n==========================================")
print("         TRAIN TEST SPLIT")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))


# ============================================================
# TF-IDF
# ============================================================

print("\n==========================================")
print("              TF-IDF")
print("==========================================")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF completed!")

print(
    "Number of TF-IDF features:",
    len(tfidf.get_feature_names_out())
)


# ============================================================
# TF-IDF TABLE
# ============================================================

print("\n==========================================")
print("           TF-IDF SAMPLE TABLE")
print("==========================================")

features = tfidf.get_feature_names_out()

tfidf_table = pd.DataFrame(
    X_train_tfidf[:5].toarray(),
    columns=features
)

print("Showing first 20 TF-IDF features:")

display(
    tfidf_table.iloc[:, :20].round(4)
)


# ============================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n==========================================")
print("          MODEL TRAINING")
print("==========================================")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed!")


# ============================================================
# TEST MODEL
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================================")
print("           MODEL ACCURACY")
print("==========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("       CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# COMPLETE QUERY ANALYSIS FUNCTION
# ============================================================

def analyze_query(query):

    print("\n")
    print("==========================================")
    print("              NLP ANALYSIS")
    print("==========================================")

    # 1. ORIGINAL TEXT

    print("\n1. ORIGINAL TEXT")
    print("------------------------------------------")

    print(query)


    # 2. CLEAN TEXT

    clean_text = str(query).lower()

    clean_text = re.sub(
        r"<.*?>",
        " ",
        clean_text
    )

    clean_text = re.sub(
        r"http\S+|www\S+",
        " ",
        clean_text
    )

    clean_text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        clean_text
    )


    # 3. TOKENIZATION

    tokens = word_tokenize(
        clean_text
    )

    print("\n2. TOKENIZATION")
    print("------------------------------------------")

    print(tokens)


    # 4. STOP WORDS FOUND

    found_stopwords = [
        word
        for word in tokens
        if word in stop_words
    ]

    print("\n3. STOP WORDS FOUND")
    print("------------------------------------------")

    if len(found_stopwords) > 0:
        print(found_stopwords)
    else:
        print("No stop words found.")


    # 5. AFTER STOP WORD REMOVAL

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
        and word.isalpha()
    ]

    print("\n4. AFTER STOP WORD REMOVAL")
    print("------------------------------------------")

    print(filtered_tokens)


    # 6. STEMMING

    stemmed_tokens = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    print("\n5. STEMMING")
    print("------------------------------------------")

    stemming_table = pd.DataFrame({
        "Original Word": filtered_tokens,
        "Stemmed Word": stemmed_tokens
    })

    display(stemming_table)


    # 7. FINAL PROCESSED TEXT

    processed = " ".join(
        stemmed_tokens
    )

    print("\n6. FINAL PROCESSED TEXT")
    print("------------------------------------------")

    print(processed)


    # 8. TF-IDF FOR QUERY

    vector = tfidf.transform(
        [processed]
    )

    values = vector.toarray()[0]

    words = tfidf.get_feature_names_out()

    tfidf_query = pd.DataFrame({
        "Word": words,
        "TF-IDF": values
    })

    tfidf_query = tfidf_query[
        tfidf_query["TF-IDF"] > 0
    ]

    tfidf_query = tfidf_query.sort_values(
        by="TF-IDF",
        ascending=False
    )

    print("\n7. TF-IDF VALUES FOR QUERY")
    print("------------------------------------------")

    if len(tfidf_query) > 0:

        display(
            tfidf_query.round(4)
        )

    else:

        print(
            "No words from this query were "
            "found in the TF-IDF vocabulary."
        )


    # 9. CATEGORY PREDICTION

    prediction = model.predict(
        vector
    )[0]

    probability = model.predict_proba(
        vector
    )[0]

    confidence = max(
        probability
    ) * 100


    # 10. TOP 3 PREDICTIONS

    class_names = model.classes_

    top_indices = probability.argsort()[
        ::-1
    ][:3]

    top_predictions = pd.DataFrame({

        "Category":
        class_names[top_indices],

        "Probability (%)":
        probability[top_indices] * 100

    })

    print("\n8. TOP 3 PREDICTED CATEGORIES")
    print("------------------------------------------")

    display(
        top_predictions.round(2)
    )


    # 11. FINAL RESULT

    print("\n9. FINAL PREDICTION")
    print("------------------------------------------")

    print(
        "Predicted Category:",
        prediction
    )

    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )

    print("\n==========================================")
    print("           ANALYSIS COMPLETE")
    print("==========================================")


# ============================================================
# USER QUERY LOOP
# ============================================================

print("\n==========================================")
print("       NEWS NLP ANALYZER")
print("==========================================")

print("\nMODEL IS READY!")

print("\nEnter any news-related query.")

print("\nThe program automatically performs:")

print("Original Text")
print("↓")
print("Tokenization")
print("↓")
print("Stop Word Identification")
print("↓")
print("Stop Word Removal")
print("↓")
print("Stemming")
print("↓")
print("TF-IDF")
print("↓")
print("News Category Prediction")

print("\nType 'exit' to stop.")


# ============================================================
# CONTINUOUS QUERY LOOP
# ============================================================

while True:

    query = input(
        "\nEnter your query: "
    )

    if query.lower().strip() == "exit":

        print("\nProgram stopped.")

        break

    if query.strip() == "":

        print("Please enter some text.")

        continue

    analyze_query(query)
