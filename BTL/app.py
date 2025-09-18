# app.py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set_style("whitegrid")

# ======================
# 1. Tiền xử lý dữ liệu
# ======================
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)     # URL
    s = re.sub(r"\S+@\S+", " ", s)                     # Email
    s = re.sub(r"\+?\d[\d\s\-]{5,}\d", " ", s)         # Số điện thoại
    s = re.sub(r"[^a-z0-9\s]", " ", s)                 # Ký tự đặc biệt
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_and_prepare(path):
    df = pd.read_csv(path, encoding="utf-8")
    df['text'] = df['sms']  # đồng nhất cột dùng cho pipeline
    df['text_clean'] = df['sms'].apply(clean_text)
    df['len_char'] = df['sms'].apply(lambda x: len(str(x)))
    df['len_word'] = df['sms'].apply(lambda x: len(str(x).split()))
    df['has_number'] = df['sms'].str.contains(r"\d").astype(int)
    df['has_special'] = df['sms'].str.contains(r"[^a-zA-Z0-9\s]").astype(int)
    return df

# ======================
# 2. EDA (hiển thị trong Streamlit)
# ======================
def exploratory_data_analysis_streamlit(df):
    # Phân bố nhãn
    label_counts = df['label'].value_counts().sort_index()
    label_percent = (label_counts / label_counts.sum() * 100).round(2)

    st.subheader("📊 Phân bố nhãn (Count & Percent)")
    label_df = pd.DataFrame({
        "Label": label_counts.index.astype(str),
        "Count": label_counts.values,
        "Percent (%)": label_percent.values
    }).set_index("Label")
    st.table(label_df)

    # Biểu đồ cột (Count)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, palette="viridis", ax=ax)
    ax.set_title("Số lượng theo nhãn")
    ax.set_xlabel("Label (0 = Ham, 1 = Spam)")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)

    # Biểu đồ bánh (Percent)
    fig2, ax2 = plt.subplots(figsize=(4,3))
    colors = ["#4CAF50", "#F44336"] if len(label_percent) == 2 else None
    ax2.pie(label_percent.values, labels=label_percent.index.astype(str),
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title("Tỷ lệ phần trăm theo nhãn")
    ax2.axis('equal')
    st.pyplot(fig2)

    # Độ dài tin nhắn
    df['msg_len'] = df['text'].apply(len)
    st.subheader("📈 Phân phối độ dài tin nhắn")
    fig3, ax3 = plt.subplots(figsize=(4,3))
    sns.histplot(df['msg_len'], bins=50, kde=True, ax=ax3)
    ax3.set_xlabel("Độ dài tin nhắn")
    ax3.set_ylabel("Tần suất")
    st.pyplot(fig3)

    # Boxplot
    st.subheader("📦 Boxplot độ dài theo nhãn")
    fig4, ax4 = plt.subplots(figsize=(4,3))
    sns.boxplot(x='label', y='msg_len', data=df, palette="Set2", ax=ax4)
    ax4.set_xlabel("Label")
    ax4.set_ylabel("Độ dài tin nhắn")
    st.pyplot(fig4)

    # Top từ
    st.subheader("🔤 Top 20 từ phổ biến (chưa loại stopwords)")
    all_words = " ".join(df['text_clean']).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(20)
    if common_words:
        words, counts = zip(*common_words)
        fig5, ax5 = plt.subplots(figsize=(5,3.5))
        sns.barplot(x=list(counts), y=list(words), palette="mako", ax=ax5)
        ax5.set_xlabel("Tần suất")
        ax5.set_ylabel("Từ")
        st.pyplot(fig5)

    # Tỷ lệ chứa số / ký tự đặc biệt
    st.subheader("🔎 Tỷ lệ chứa số / ký tự đặc biệt theo nhãn")
    st.write(df.groupby('label')[['has_number', 'has_special']].mean().round(4))

    # Báo cáo tự động ngắn
    st.subheader("📑 Báo cáo EDA tự động (tóm tắt)")
    report = []
    if label_percent.min() < 30:
        report.append("⚠️ Dữ liệu bị mất cân bằng nhãn, cân nhắc oversampling/undersampling.")
    else:
        report.append("✅ Dữ liệu phân bố khá cân bằng giữa các nhãn.")
    report.append("📈 Tin nhắn thường ngắn (<200 ký tự).")
    report.append("📦 Spam có xu hướng dài hơn ham.")
    report.append("🔎 Spam có xu hướng chứa nhiều số/ký tự đặc biệt hơn ham.")
    for r in report:
        st.markdown("- " + r)

# ======================
# 3. Huấn luyện & đánh giá
# ======================
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    cms = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({"Mô hình": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})
        cms[name] = confusion_matrix(y_test, y_pred)
    return pd.DataFrame(results), cms

# ======================
# Main pipeline (chạy khi start app)
# ======================
file_path = "BTL/data/train.csv" #"E:/DangVanThanh/train.csv"   # sửa nếu cần
df = load_and_prepare(file_path)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.9, ngram_range=(1,2))
X = tfidf.fit_transform(df['text_clean'])
y = df['label'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}
results_df, cms = train_and_evaluate(models, X_train, X_test, y_train, y_test)

# Lưu TF-IDF và best model
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
best_model_name = results_df.sort_values(by="F1-score", ascending=False).iloc[0]["Mô hình"]
best_model = models[best_model_name]
joblib.dump(best_model, "best_model.pkl")

# ======================
# Streamlit UI: 3 tab
# ======================
st.set_page_config(page_title="Spam SMS Detector", layout="centered")
tab1, tab2, tab3 = st.tabs(["📊 Khám phá dữ liệu", "🤖 Kết quả mô hình", "📩 Dự đoán Spam/Ham"])

with tab1:
    st.header("📊 Khám phá dữ liệu (EDA)")
    exploratory_data_analysis_streamlit(df)

with tab2:
    st.header("🤖 Kết quả mô hình")
    st.dataframe(results_df.sort_values(by="F1-score", ascending=False).reset_index(drop=True))
    for name, cm in cms.items():
        st.subheader(f"Confusion Matrix - {name}")
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Ham (0)", "Spam (1)"],
                    yticklabels=["Ham (0)", "Spam (1)"],
                    ax=ax_cm)
        st.pyplot(fig_cm)

with tab3:
    st.header("📩 Dự đoán Spam/Ham")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("best_model.pkl")

    sms = st.text_area("Nhập tin nhắn:", height=150)
    if st.button("Dự đoán"):
        if sms.strip() == "":
            st.warning("Bạn chưa nhập tin nhắn!")
        else:
            sms_clean = clean_text(sms)
            X_new = tfidf.transform([sms_clean])
            y_pred = model.predict(X_new)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_new).max()
            if int(y_pred) == 1:
                st.error(f"🚨 Spam ({'%.2f'% (prob*100) + '%' if prob is not None else ''})")
            else:
                st.success(f"✅ Ham ({'%.2f'% (prob*100) + '%' if prob is not None else ''})")
