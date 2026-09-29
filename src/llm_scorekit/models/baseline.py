from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data import TextDataset


# Fit word TF-IDF and logistic regression on the supplied training partition.
def fit_tfidf_classifier(training_data: TextDataset) -> Pipeline:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1_000,
    )

    model = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
    model.fit(training_data.texts, training_data.labels)

    return model
