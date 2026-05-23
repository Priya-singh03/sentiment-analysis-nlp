import re
import math
from collections import Counter

# ===== PREPROCESSING =====

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'just', 'don', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
}

def preprocess(text):
    """Tokenize, lowercase, remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

def tfidf_vectorize(corpus):
    """Compute TF-IDF vectors for a list of texts."""
    tokenized = [preprocess(doc) for doc in corpus]
    
    # Build vocabulary
    vocab = sorted(set(word for doc in tokenized for word in doc))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    N = len(corpus)
    
    # Document frequency
    df = Counter()
    for doc in tokenized:
        df.update(set(doc))
    
    # TF-IDF matrix
    vectors = []
    for doc in tokenized:
        tf = Counter(doc)
        total = len(doc) if doc else 1
        vec = [0.0] * len(vocab)
        for word, idx in word_to_idx.items():
            if tf[word] > 0:
                tf_val = tf[word] / total
                idf_val = math.log((N + 1) / (df[word] + 1)) + 1
                vec[idx] = tf_val * idf_val
        vectors.append(vec)
    
    return vectors, vocab


# ===== NAIVE BAYES CLASSIFIER =====

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.classes = []
        self.vocab = set()

    def train(self, texts, labels):
        self.classes = list(set(labels))
        N = len(texts)

        for cls in self.classes:
            cls_texts = [texts[i] for i in range(N) if labels[i] == cls]
            self.class_probs[cls] = math.log(len(cls_texts) / N)

            all_words = []
            for text in cls_texts:
                all_words.extend(preprocess(text))
            self.vocab.update(all_words)

            word_count = Counter(all_words)
            total_words = sum(word_count.values()) + len(self.vocab)
            self.word_probs[cls] = {
                word: math.log((count + 1) / total_words)
                for word, count in word_count.items()
            }
            self.word_probs[cls]['__unk__'] = math.log(1 / total_words)

    def predict(self, text):
        tokens = preprocess(text)
        scores = {}
        for cls in self.classes:
            score = self.class_probs[cls]
            for token in tokens:
                score += self.word_probs[cls].get(
                    token, self.word_probs[cls]['__unk__']
                )
            scores[cls] = score

        predicted = max(scores, key=lambda c: scores[c])
        # Softmax for confidence
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total = sum(exp_scores.values())
        confidence = {c: round(v / total, 4) for c, v in exp_scores.items()}
        return predicted, confidence


# ===== SVM-INSPIRED CLASSIFIER (Linear) =====

class LinearSVMClassifier:
    def __init__(self, learning_rate=0.01, epochs=100, C=1.0):
        self.lr = learning_rate
        self.epochs = epochs
        self.C = C
        self.weights = {}
        self.bias = {}
        self.classes = []

    def _dot(self, vec1, vec2):
        return sum(a * b for a, b in zip(vec1, vec2))

    def train(self, vectors, labels):
        self.classes = list(set(labels))
        n_features = len(vectors[0]) if vectors else 0

        for cls in self.classes:
            self.weights[cls] = [0.0] * n_features
            self.bias[cls] = 0.0

        for _ in range(self.epochs):
            for i, (vec, label) in enumerate(zip(vectors, labels)):
                for cls in self.classes:
                    y = 1 if label == cls else -1
                    score = self._dot(self.weights[cls], vec) + self.bias[cls]
                    if y * score < 1:
                        # Update weights
                        self.weights[cls] = [
                            w - self.lr * (w - self.C * y * x)
                            for w, x in zip(self.weights[cls], vec)
                        ]
                        self.bias[cls] += self.lr * self.C * y
                    else:
                        self.weights[cls] = [
                            w - self.lr * w for w in self.weights[cls]
                        ]

    def predict(self, vector):
        scores = {
            cls: self._dot(self.weights[cls], vector) + self.bias[cls]
            for cls in self.classes
        }
        predicted = max(scores, key=lambda c: scores[c])
        max_s = max(scores.values())
        exp_scores = {c: math.exp(s - max_s) for c, s in scores.items()}
        total = sum(exp_scores.values())
        confidence = {c: round(v / total, 4) for c, v in exp_scores.items()}
        return predicted, confidence


# ===== TRAINING DATA =====

TRAINING_DATA = [
    # Positive
    ("This is absolutely wonderful and amazing!", "positive"),
    ("I love this product, it works perfectly!", "positive"),
    ("Excellent quality, highly recommend!", "positive"),
    ("Great experience, very happy with the results.", "positive"),
    ("Outstanding performance, exceeded expectations.", "positive"),
    ("Fantastic service, will definitely come back!", "positive"),
    ("Best purchase I have ever made, incredible value.", "positive"),
    ("Super impressed, works like a charm!", "positive"),
    ("Really enjoy using this, very satisfying.", "positive"),
    ("Brilliant design, easy to use, love it.", "positive"),
    ("Amazing results, completely satisfied.", "positive"),
    ("Top notch quality, very pleased.", "positive"),
    ("Perfect exactly what I needed, great product.", "positive"),
    ("Wonderful experience from start to finish.", "positive"),
    ("Superb craftsmanship, highly impressed.", "positive"),

    # Negative
    ("This is terrible, completely useless product.", "negative"),
    ("Worst purchase ever, total waste of money.", "negative"),
    ("Horrible quality, broke after one day.", "negative"),
    ("Very disappointed, does not work as described.", "negative"),
    ("Awful experience, customer service is terrible.", "negative"),
    ("Complete garbage, save your money.", "negative"),
    ("Defective product, very frustrating experience.", "negative"),
    ("Extremely poor quality, not worth the price.", "negative"),
    ("Disgusting product, will never buy again.", "negative"),
    ("Terrible performance, constantly crashing.", "negative"),
    ("Very bad, nothing works as expected.", "negative"),
    ("Dreadful experience, regret buying this.", "negative"),
    ("Pathetic quality, falls apart immediately.", "negative"),
    ("Shocking how bad this product is.", "negative"),
    ("Appalling, worst thing I have ever bought.", "negative"),

    # Neutral
    ("The product arrived on time.", "neutral"),
    ("It does what it is supposed to do.", "neutral"),
    ("Average quality, neither good nor bad.", "neutral"),
    ("Okay product, nothing special.", "neutral"),
    ("Meets basic requirements, nothing more.", "neutral"),
    ("Standard product, works as expected.", "neutral"),
    ("Reasonable price for what you get.", "neutral"),
    ("Typical product, no surprises.", "neutral"),
    ("Acceptable performance overall.", "neutral"),
    ("Product is as described, nothing extra.", "neutral"),
    ("Works fine, does the job.", "neutral"),
    ("Not bad not great, just average.", "neutral"),
    ("Mediocre experience, could be better.", "neutral"),
    ("Fairly standard, gets the job done.", "neutral"),
    ("Normal product, expected results.", "neutral"),
]


# ===== MAIN SENTIMENT ANALYZER =====

class SentimentAnalyzer:
    def __init__(self):
        texts = [d[0] for d in TRAINING_DATA]
        labels = [d[1] for d in TRAINING_DATA]

        # Train Naive Bayes
        self.nb = NaiveBayesClassifier()
        self.nb.train(texts, labels)

        # Train SVM with TF-IDF
        vectors, self.vocab = tfidf_vectorize(texts)
        self.svm = LinearSVMClassifier(learning_rate=0.05, epochs=200)
        self.svm.train(vectors, labels)
        self.train_texts = texts

    def _text_to_vector(self, text):
        tokens = preprocess(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        N = len(self.train_texts)

        # Compute DF from training
        df = Counter()
        for doc in self.train_texts:
            df.update(set(preprocess(doc)))

        vec = []
        for word in self.vocab:
            if tf[word] > 0:
                tf_val = tf[word] / total
                idf_val = math.log((N + 1) / (df[word] + 1)) + 1
                vec.append(tf_val * idf_val)
            else:
                vec.append(0.0)
        return vec

    def analyze(self, text, model="Naive Bayes"):
        if model == "Naive Bayes":
            prediction, confidence = self.nb.predict(text)
        else:
            vec = self._text_to_vector(text)
            prediction, confidence = self.svm.predict(vec)

        # Get key words
        tokens = preprocess(text)
        word_freq = Counter(tokens)
        top_words = word_freq.most_common(5)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "top_words": top_words
        }

    def batch_analyze(self, texts, model="Naive Bayes"):
        return [self.analyze(text, model) for text in texts]

    def evaluate(self, test_texts, test_labels, model="Naive Bayes"):
        results = self.batch_analyze(test_texts, model)
        predictions = [r["prediction"] for r in results]

        correct = sum(p == l for p, l in zip(predictions, test_labels))
        accuracy = correct / len(test_labels)

        classes = ["positive", "negative", "neutral"]
        metrics = {}
        for cls in classes:
            tp = sum(p == cls and l == cls for p, l in zip(predictions, test_labels))
            fp = sum(p == cls and l != cls for p, l in zip(predictions, test_labels))
            fn = sum(p != cls and l == cls for p, l in zip(predictions, test_labels))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics[cls] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

        return accuracy, metrics
