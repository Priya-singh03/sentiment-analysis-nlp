import streamlit as st
from sentiment import SentimentAnalyzer

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Sentiment Analysis NLP",
    page_icon="🎭",
    layout="wide"
)

# ===== CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0a0f;
        color: #f0f0ff;
    }

    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f472b6, #a78bfa, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        color: #8888aa;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .result-positive {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .result-negative {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .result-neutral {
        background: rgba(156, 163, 175, 0.1);
        border: 2px solid rgba(156, 163, 175, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    .sentiment-label {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .confidence-bar {
        background: #111118;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #5b21b6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }

    .stTextArea textarea {
        background: #111118 !important;
        color: #f0f0ff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }

    div[data-testid="metric-container"] {
        background: #111118;
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 10px;
        padding: 1rem;
    }

    .word-chip {
        display: inline-block;
        background: rgba(124, 58, 237, 0.2);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.85rem;
        color: #c4b5fd;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
@st.cache_resource
def load_analyzer():
    return SentimentAnalyzer()

analyzer = load_analyzer()

# ===== HEADER =====
st.markdown('<div class="main-title">🎭 Sentiment Analysis NLP</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Positive / Negative / Neutral Classification using Naive Bayes & SVM | TF-IDF Vectorization</div>', unsafe_allow_html=True)
st.divider()

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox("ML Algorithm", ["Naive Bayes", "SVM (Linear)"])

    st.divider()
    st.markdown("### 📊 About")
    st.markdown("""
    **Dataset:** 45 labeled samples  
    **Classes:** Positive, Negative, Neutral  
    **Features:** TF-IDF Vectorization  
    **Models:** Naive Bayes, Linear SVM  
    **Metrics:** Accuracy, Precision, Recall, F1
    """)

    st.divider()
    st.markdown("### 🧪 Corpora Used")
    st.markdown("""
    - Product Reviews
    - Service Feedback  
    - General Opinions
    """)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis", "📈 Model Evaluation"])

# ===== TAB 1: SINGLE ANALYSIS =====
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### ✍️ Input Text")

        SAMPLE_REVIEWS = {
            "Select a sample...": "",
            "Positive Review": "This product is absolutely amazing! Best purchase I've ever made. The quality is outstanding and it works perfectly.",
            "Negative Review": "Terrible product, complete waste of money. Broke after one day. Very disappointed with the quality.",
            "Neutral Review": "The product arrived on time. It does what it is supposed to do. Nothing special but works fine.",
            "Mixed Review": "The product looks good but the delivery was slow. Quality seems decent but customer service could be better.",
        }

        sample = st.selectbox("Try a sample:", list(SAMPLE_REVIEWS.keys()), key="single_sample")
        default = SAMPLE_REVIEWS[sample] if sample != "Select a sample..." else ""

        user_text = st.text_area(
            "Enter text to analyze:",
            value=default,
            height=180,
            placeholder="Type a review, comment, or any text...",
            key="single_input"
        )

        word_count = len(user_text.split()) if user_text else 0
        st.caption(f"📊 Word count: {word_count}")

        analyze_btn = st.button("🔍 Analyze Sentiment", key="single_btn")

    with col2:
        st.markdown("### 🎯 Result")

        if analyze_btn:
            if not user_text or word_count < 3:
                st.error("⚠️ Please enter at least 3 words!")
            else:
                result = analyzer.analyze(user_text, model_choice)
                pred = result["prediction"]
                conf = result["confidence"]
                top_words = result["top_words"]

                # Emoji & color
                emoji = "😊" if pred == "positive" else "😞" if pred == "negative" else "😐"
                css_class = f"result-{pred}"
                color = "#10b981" if pred == "positive" else "#ef4444" if pred == "negative" else "#9ca3af"

                st.markdown(f"""
                <div class="{css_class}">
                    <div style="font-size: 3rem">{emoji}</div>
                    <div class="sentiment-label" style="color: {color}">{pred.upper()}</div>
                    <div style="color: #8888aa; font-size: 0.9rem">Confidence: {round(conf.get(pred, 0) * 100, 1)}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Class Probabilities")
                for cls, prob in sorted(conf.items(), key=lambda x: x[1], reverse=True):
                    bar_color = "#10b981" if cls == "positive" else "#ef4444" if cls == "negative" else "#9ca3af"
                    st.markdown(f"**{cls.capitalize()}**")
                    st.progress(prob)
                    st.caption(f"{round(prob * 100, 1)}%")

                if top_words:
                    st.markdown("#### 🔑 Key Words")
                    chips = " ".join([f'<span class="word-chip">{w}</span>' for w, _ in top_words])
                    st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("👈 Enter text and click **Analyze Sentiment**")

# ===== TAB 2: BATCH ANALYSIS =====
with tab2:
    st.markdown("### 📋 Analyze Multiple Texts at Once")

    DEFAULT_BATCH = """This product is amazing, I love it!
Terrible quality, complete waste of money.
The item arrived on time and works fine.
Absolutely outstanding, highly recommend!
Very disappointed, does not work as described.
Average product, nothing special.
Best purchase ever, incredible value!
Horrible experience, will never buy again."""

    batch_input = st.text_area(
        "Enter one text per line:",
        value=DEFAULT_BATCH,
        height=250,
        key="batch_input"
    )

    if st.button("🚀 Analyze All", key="batch_btn"):
        texts = [t.strip() for t in batch_input.strip().split('\n') if t.strip()]
        if not texts:
            st.error("Please enter at least one text!")
        else:
            results = analyzer.batch_analyze(texts, model_choice)

            pos = sum(1 for r in results if r["prediction"] == "positive")
            neg = sum(1 for r in results if r["prediction"] == "negative")
            neu = sum(1 for r in results if r["prediction"] == "neutral")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(results))
            c2.metric("😊 Positive", pos)
            c3.metric("😞 Negative", neg)
            c4.metric("😐 Neutral", neu)

            st.divider()

            for i, (text, result) in enumerate(zip(texts, results)):
                pred = result["prediction"]
                conf = result["confidence"]
                emoji = "😊" if pred == "positive" else "😞" if pred == "negative" else "😐"
                color = "#10b981" if pred == "positive" else "#ef4444" if pred == "negative" else "#9ca3af"

                with st.expander(f"{emoji} Text {i+1}: {text[:60]}..."):
                    st.markdown(f"**Prediction:** <span style='color:{color}'>{pred.upper()}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** {round(conf.get(pred, 0) * 100, 1)}%")
                    st.markdown(f"**Text:** {text}")

# ===== TAB 3: MODEL EVALUATION =====
with tab3:
    st.markdown("### 📈 Model Performance Evaluation")

    TEST_DATA = [
        ("I absolutely love this, fantastic product!", "positive"),
        ("This is the worst thing I have ever bought.", "negative"),
        ("Works as expected, nothing special.", "neutral"),
        ("Incredible quality, highly recommend to everyone!", "positive"),
        ("Very bad experience, completely useless.", "negative"),
        ("Okay product, does the job.", "neutral"),
        ("Amazing value for money, very impressed!", "positive"),
        ("Defective and frustrating, very disappointed.", "negative"),
        ("Standard product, average performance.", "neutral"),
        ("Superb quality, exceeded all expectations!", "positive"),
        ("Dreadful product, total waste of time.", "negative"),
        ("Reasonable product, meets basic needs.", "neutral"),
    ]

    test_texts = [d[0] for d in TEST_DATA]
    test_labels = [d[1] for d in TEST_DATA]

    if st.button("🧪 Run Evaluation", key="eval_btn"):
        with st.spinner("Evaluating models..."):
            nb_acc, nb_metrics = analyzer.evaluate(test_texts, test_labels, "Naive Bayes")
            svm_acc, svm_metrics = analyzer.evaluate(test_texts, test_labels, "SVM (Linear)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔵 Naive Bayes")
            st.metric("Accuracy", f"{round(nb_acc * 100, 1)}%")
            for cls, m in nb_metrics.items():
                with st.expander(f"{cls.capitalize()}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision", m["precision"])
                    c2.metric("Recall", m["recall"])
                    c3.metric("F1 Score", m["f1"])

        with col2:
            st.markdown("#### 🟣 SVM (Linear)")
            st.metric("Accuracy", f"{round(svm_acc * 100, 1)}%")
            for cls, m in svm_metrics.items():
                with st.expander(f"{cls.capitalize()}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precision", m["precision"])
                    c2.metric("Recall", m["recall"])
                    c3.metric("F1 Score", m["f1"])

        st.divider()
        st.markdown("#### 📊 Test Samples Used")
        for text, label in TEST_DATA:
            emoji = "😊" if label == "positive" else "😞" if label == "negative" else "😐"
            st.caption(f"{emoji} **{label.upper()}** — {text}")
