import streamlit as st 
import pandas as pd
import torch
import torch._dynamo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px

# Suppress torch.compile() errors
torch._dynamo.config.suppress_errors = True

# Must be called first
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Load DistilBERT with performance enhancements
@st.cache_resource
def load_model():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Optional performance boost with compile
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print("⚠️ torch.compile not applied:", e)

    return tokenizer, model, device

tokenizer, model, device = load_model()

# Category mapping
category_keywords = {
    "General Feedback": ["overall experience", "general feedback", "honest opinion", "constructive criticism", "testimonial"],
    "Delivery Issues": ["late delivery", "delayed shipping", "wrong address", "tracking issue", "courier problem", "package lost", "damaged package"],
    "Product Quality": ["defective item", "low durability", "poor material", "manufacturing defect", "scratched", "not as described", "faulty product"],
    "Pricing Complaints": ["overpriced", "hidden charges", "price hike", "billing issue", "unreasonable cost", "expensive for quality", "discount problem"],
    "Customer Service": ["unresponsive support", "rude staff", "slow response", "poor assistance", "helpdesk issue", "lack of follow-up", "unhelpful agent"],
    "Usability & Features": ["confusing interface", "hard to use", "missing feature", "complicated navigation", "not user-friendly", "glitchy experience", "improvement needed"],
    "Positive Experience": ["highly recommend", "exceptional service", "exceeded expectations", "best experience", "impressed", "fantastic quality", "outstanding product"],
    "Negative Experience": ["worst purchase", "frustrating", "huge disappointment", "terrible service", "never buying again", "horrible experience", "waste of money"],
    "Recommendation & Loyalty": ["loyal customer", "repeat purchase", "brand trust", "long-term user", "customer retention", "always choose this", "favorite brand"],
    "Neutral Remarks": ["just okay", "average quality", "not bad", "decent enough", "fine product", "acceptable", "nothing special"]
}

# Sentiment analysis
def analyze_sentiments(data, batch_size=8):
    sentiments, sentiment_labels = [], []
    labels_map = {0: "negative", 1: "positive"}
    texts = data["text"].tolist()
    my_bar = st.progress(0)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_scores = probs.cpu().numpy()
            sentiment_class = np.argmax(sentiment_scores, axis=1)

        sentiment_labels.extend([labels_map[int(x)] for x in sentiment_class])
        sentiments.extend(sentiment_scores[:, 1])  # positive score

        my_bar.progress(min((i + batch_size) / len(texts), 1.0))
        torch.cuda.empty_cache()

    data["sentiment"] = sentiments
    data["sentiment_label"] = sentiment_labels
    return data

# Assign clusters
def dynamic_category_assignment(data, num_clusters):
    categories = list(category_keywords.keys())
    assigned = [categories[i % len(categories)] for i in range(num_clusters)]
    data["category"] = data["cluster"].apply(lambda x: assigned[x])
    return data

# K-means-like clustering based on sentiment
def numpy_kmeans(data, n_clusters=10, max_iter=100):
    np.random.seed(42)
    scores = data["sentiment"].values
    centroids = np.random.choice(scores, n_clusters)

    for _ in range(max_iter):
        clusters = np.array([np.abs(scores - c) for c in centroids]).argmin(axis=0)
        new_centroids = np.array([scores[clusters == k].mean() if np.any(clusters == k) else centroids[k] for k in range(n_clusters)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    data["cluster"] = clusters
    return dynamic_category_assignment(data, n_clusters), centroids

# Plot results
def plot_sentiment_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data, x="sentiment", hue="category", bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Category Counts")
    st.bar_chart(data["category"].value_counts())

def plot_sentiment_vs_cluster(data):
    fig = px.scatter(data, x="sentiment", y="cluster", color="category", title="Sentiment vs Cluster", labels={"sentiment": "Sentiment Score", "cluster": "Cluster"})
    st.plotly_chart(fig)

# Streamlit UI
def main():
    st.title("⚡ Super-Fast Sentiment Analysis (DistilBERT + Optimized)")

    option = st.sidebar.radio("Select Stage", ["Upload Data", "Run Sentiment Analysis", "Cluster Data", "View Results"])

    if option == "Upload Data":
        uploaded_file = st.file_uploader("📤 Upload CSV with a 'text' column", type=["csv"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if "text" not in df.columns:
                    st.error("❌ Missing 'text' column!")
                else:
                    st.session_state["raw_data"] = df
                    st.session_state["analyzed_data"] = None
                    st.session_state["categorized_data"] = None
                    st.success("✅ File Uploaded!")
                    st.write(df.head())
            except Exception as e:
                st.error("❌ Failed to load file")
                st.error(str(e))

    elif option == "Run Sentiment Analysis":
        if "raw_data" in st.session_state:
            if st.session_state.get("analyzed_data") is None:
                with st.spinner("🔍 Analyzing... please wait"):
                    analyzed = analyze_sentiments(st.session_state["raw_data"])
                    st.session_state["analyzed_data"] = analyzed
                    st.success("✔️ Analysis Done")
                    st.write(analyzed.head())
            else:
                st.info("✅ Already analyzed. No need to run again.")
                st.write(st.session_state["analyzed_data"].head())
        else:
            st.warning("📤 Upload your file first.")

    elif option == "Cluster Data":
        if "analyzed_data" in st.session_state:
            if st.session_state.get("categorized_data") is None:
                with st.spinner("🔄 Clustering..."):
                    clustered, _ = numpy_kmeans(st.session_state["analyzed_data"], n_clusters=10)
                    st.session_state["categorized_data"] = clustered
                    st.success("✔️ Clustering Done")
                    st.write(clustered.head())
            else:
                st.info("✅ Already clustered.")
                st.write(st.session_state["categorized_data"].head())
        else:
            st.warning("📊 Run sentiment analysis first.")

    elif option == "View Results":
        if "categorized_data" in st.session_state:
            st.subheader("📈 Sentiment Distribution by Category")
            plot_sentiment_distribution(st.session_state["categorized_data"])

            st.subheader("📌 Sentiment vs Clusters")
            plot_sentiment_vs_cluster(st.session_state["categorized_data"])

        else:
            st.warning("🔍 Please finish clustering.")

if __name__ == "__main__":
    main()
