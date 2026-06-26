import os
import re
import time
import uuid
from datetime import datetime
from groq import Groq
import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, session
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


# =========================
# ENV LOAD
# =========================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(32))

# =========================
# SESSION STORE
# =========================
session_store: dict[str, dict] = {}

# =========================
# TRENDYOL API
# =========================
API_URL = (
    "https://apigw.trendyol.com/"
    "discovery-storefront-trproductgw-service/"
    "api/review-read/product-reviews/detailed"
)

# =========================
# LABEL MAPPING
# =========================
LABEL_MAP = {
    "LABEL_0": "neutral",
    "LABEL_1": "positive",
    "LABEL_2": "negative",
}

# =========================
# SENTIMENT PIPELINE
# =========================
_sentiment_pipeline = None


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        _sentiment_pipeline = pipeline(
            task="text-classification",
            model="saribasmetehan/bert-base-turkish-sentiment-analysis",
            tokenizer="saribasmetehan/bert-base-turkish-sentiment-analysis",
            device=device,
        )
    return _sentiment_pipeline


# =========================
# TEXT HELPERS
# =========================
def metin_temizle(metin: str) -> str:
    if not isinstance(metin, str):
        return ""
    metin = metin.replace("İ", "i").replace("I", "ı")
    temiz = re.sub(r"[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]", "", metin)
    temiz = temiz.lower()
    return re.sub(r"\s+", " ", temiz).strip()


def extract_content_id(url: str) -> str:
    pid = re.search(r"-p-(\d+)", url or "")
    if not pid:
        qpid = re.search(r"[?&]contentId=(\d+)", url or "")
        if qpid:
            return qpid.group(1)
        raise ValueError(
            "Geçerli bir Trendyol ürün URL'si giriniz. "
            "URL içinde -p-ürünId formatı bulunmalı."
        )
    return pid.group(1)


def get_turkish_stopwords() -> list[str]:
    try:
        return stopwords.words("turkish")
    except LookupError:
        import nltk
        nltk.download("stopwords")
        return stopwords.words("turkish")


# =========================
# YILDIZ + MODEL FÜZYONU
# =========================
def fuse_sentiment(label: str, score: float, star: int) -> str:
    """
    Model tahmini ile yıldız skorunu birleştir.
    Yıldız 1-2 → kesinlikle negatif
    Yıldız 4-5 → modelin negatif dediği ama düşük güvenli yorumları pozitife çek
    Yıldız 3   → modele güven
    """
    if star <= 2:
        return "negative"
    if star >= 4:
        if label == "negative" and score < 0.82:
            return "positive"
        return label if label != "neutral" else "positive"
    # Yıldız 3: modele güven ama nötrü koru
    return label


# =========================
# DATA FETCHING
# =========================
def fetch_reviews(content_id: str, max_pages: int = 25) -> pd.DataFrame:
    all_reviews = []
    for page in range(max_pages):
        params = {
            "contentId": content_id,
            "page": page,
            "pageSize": 20,
            "order": "DESC",
            "orderBy": "Score",
            "channelId": 1,
        }
        resp = requests.get(API_URL, params=params, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Trendyol API hatası: {resp.status_code}")

        reviews = resp.json().get("result", {}).get("reviews", [])
        if not reviews:
            break

        for r in reviews:
            try:
                tarih = datetime.fromtimestamp(r.get("createdAt", 0) / 1000).strftime("%d.%m.%Y")
            except Exception:
                tarih = ""
            all_reviews.append({
                "Kullanıcı": r.get("userFullName", ""),
                "Yorum": r.get("comment", ""),
                "Yıldız": r.get("rate", 0),
                "Tarih": tarih,
                "Beğeni": r.get("likesCount", 0),
                "Satıcı": r.get("seller", {}).get("name", ""),
            })
        time.sleep(0.2)

    if not all_reviews:
        raise RuntimeError("Bu ürün için yorum bulunamadı.")
    return pd.DataFrame(all_reviews)




# =========================
# TOKEN-EFFICIENT PROMPT BUILDER
# =========================
TRENDYOL_STOPWORDS_EXTRA = [
    "ürün", "aldım", "geldi", "tavsiye", "ederim", "trendyol",
    "teşekkürler", "kesinlikle", "bence", "çok", "daha", "var",
    "bir", "bu", "ile", "için", "gibi", "kadar", "da", "de",
    "ki", "mi", "ne", "ama", "iyi", "güzel", "kötü", "tam",
]


def extract_keywords(reviews: list[str], top_n: int = 15) -> list[str]:
    clean = [metin_temizle(r) for r in reviews if isinstance(r, str) and len(r.strip()) > 5]
    if not clean:
        return []
    try:
        stop = list(set(get_turkish_stopwords() + TRENDYOL_STOPWORDS_EXTRA))
        vec = TfidfVectorizer(stop_words=stop, max_features=top_n, ngram_range=(1, 2))
        vec.fit(clean)
        return list(vec.get_feature_names_out())
    except Exception:
        return []


def pick_representative_reviews(reviews: list[str], n: int = 5) -> list[str]:
    filtered = [
        r for r in reviews
        if isinstance(r, str) and 15 <= len(r.strip()) <= 400
    ]
    if len(filtered) < n:
        filtered = [
            r for r in reviews
            if isinstance(r, str) and 15 <= len(r.strip()) <= 700
        ]
    if not filtered:
        return []

    deduped = []
    seen_tokens: list[set] = []
    for r in filtered:
        tokens = set(metin_temizle(r).split())
        if not any(
            len(tokens & s) / max(len(tokens | s), 1) > 0.8
            for s in seen_tokens
        ):
            deduped.append(r)
            seen_tokens.append(tokens)

    if len(deduped) <= n:
        return deduped

    clean_map = [(r, metin_temizle(r)) for r in deduped]
    originals, clean = zip(*clean_map)
    try:
        stop = list(set(get_turkish_stopwords() + TRENDYOL_STOPWORDS_EXTRA))
        vec = TfidfVectorizer(stop_words=stop, max_features=120)
        mat = vec.fit_transform(clean)
        scores = np.asarray(mat.mean(axis=1)).flatten()
        idx = scores.argsort()[-n:][::-1]
        return [originals[i] for i in idx]
    except Exception:
        return list(deduped[:n])


def get_confident_reviews(df: pd.DataFrame, label: str, min_confidence: float = 0.82) -> list[str]:
    """Sadece modelin emin olduğu yorumları döndür."""
    filtered = df[
        (df["Duygu Analizi"] == label) &
        (df["Güven Skoru"] >= min_confidence)
    ]
    return filtered.sort_values("Beğeni", ascending=False)["Yorum"].head(45).tolist()


def build_prompt(df: pd.DataFrame, summary: dict) -> str:
    pos_reviews_raw = get_confident_reviews(df, "positive")
    neg_reviews_raw = get_confident_reviews(df, "negative")

    pos_keywords = extract_keywords(pos_reviews_raw, top_n=12)
    neg_keywords = extract_keywords(neg_reviews_raw, top_n=12)

    pos_samples = pick_representative_reviews(pos_reviews_raw, n=5)
    neg_samples = pick_representative_reviews(neg_reviews_raw, n=5)

    s = summary["sentiment"]
    stats_line = (
        f"Toplam {summary['total_reviews']} yorum | "
        f"Ort puan {summary['average_rating']:.1f}/5 | "
        f"Olumlu %{s['positive']['percent']:.0f} ({s['positive']['count']}) | "
        f"Olumsuz %{s['negative']['percent']:.0f} ({s['negative']['count']}) | "
        f"Nötr %{s['neutral']['percent']:.0f} ({s['neutral']['count']})"
    )

    pos_kw_line = ", ".join(pos_keywords) if pos_keywords else "—"
    neg_kw_line = ", ".join(neg_keywords) if neg_keywords else "—"
    pos_sample_block = "\n".join(f"• {r}" for r in pos_samples) if pos_samples else "Yeterli veri yok."
    neg_sample_block = "\n".join(f"• {r}" for r in neg_samples) if neg_samples else "Yeterli veri yok."

    return f"""Sen e-ticaret yorum analizi yapan tarafsız bir asistansın.
SADECE aşağıdaki veriye dayan. Dışarıdan bilgi, tahmin veya uydurma örnek ekleme.
Veri yoksa "Bu konuda yorumlarda yeterli bilgi yok." de.
Cevabın 4-6 cümleyi geçmesin. Türkçe, sade, net yaz.

[İSTATİSTİKLER]
{stats_line}

[OLUMLU ANAHTAR KELİMELER]
{pos_kw_line}

[OLUMSUZ ANAHTAR KELİMELER]
{neg_kw_line}

[TEMSİLCİ OLUMLU YORUMLAR]
{pos_sample_block}

[TEMSİLCİ OLUMSUZ YORUMLAR]
{neg_sample_block}

Kullanıcı satın alma sorarsa genel eğilime göre temkinli öner.
Çoğunluk belirginse net söyle. Azınlık görüşler için "bazı kullanıcılar" de."""


# =========================
# CORE ANALYSIS
# =========================
def analyze_product(url: str) -> dict:
    content_id = extract_content_id(url)
    df = fetch_reviews(content_id)

    # Sentiment analysis
    clean_texts = [metin_temizle(y) or "nötr" for y in df["Yorum"]]
    pipe = get_sentiment_pipeline()
    results = pipe(clean_texts, batch_size=32, truncation=True, max_length=128)

    # Label mapping + güven skoru
    df["Duygu Analizi"] = [LABEL_MAP.get(r["label"], r["label"]) for r in results]
    df["Güven Skoru"] = [r["score"] for r in results]

    # Yıldız füzyonu
    df["Duygu Analizi"] = df.apply(
        lambda row: fuse_sentiment(row["Duygu Analizi"], row["Güven Skoru"], row["Yıldız"]),
        axis=1,
    )

    # Stats
    counts = df["Duygu Analizi"].value_counts()
    ratios = df["Duygu Analizi"].value_counts(normalize=True) * 100
    pos_count = int(counts.get("positive", 0))
    neg_count = int(counts.get("negative", 0))
    neu_count = int(counts.get("neutral", 0))
    total = len(df)

    review_weight = min(1.0, np.log10(total + 1) / 2)
    sentiment_clarity = abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
    model_confidence = float(df["Güven Skoru"].mean())
    confidence_score = (model_confidence * 0.70 + review_weight * 0.15 + sentiment_clarity * 0.15) * 100

    pos_reviews = get_confident_reviews(df, "positive")
    neg_reviews = get_confident_reviews(df, "negative")

    summary = {
        "content_id": content_id,
        "total_reviews": total,
        "average_rating": float(df["Yıldız"].mean()),
        "confidence_score": round(confidence_score, 2),
        "sentiment": {
            "positive": {"count": pos_count, "percent": float(ratios.get("positive", 0))},
            "negative": {"count": neg_count, "percent": float(ratios.get("negative", 0))},
            "neutral":  {"count": neu_count, "percent": float(ratios.get("neutral", 0))},
        },
        "featured_positive": pick_representative_reviews(pos_reviews, 4),
        "featured_negative": pick_representative_reviews(neg_reviews, 4),



    }
    # =========================
    # EXCEL EXPORT (AUTO SAVE)
    # =========================
    try:
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)

        file_name = f"yorum_analiz_{content_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        file_path = os.path.join(export_dir, file_name)

        df.to_excel(file_path, index=False)

        print(f"[EXCEL] Kaydedildi -> {file_path}")

    except Exception as e:
        print(f"[EXCEL HATA] {e}")

    prompt = build_prompt(df, summary)
    return {"summary": summary, "prompt": prompt}




# =========================
# HTML TEMPLATE
# =========================
HTML = """<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trendyol Yorum Analizi</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2330;
    --border: #30363d;
    --border2: #21262d;
    --ink: #e6edf3;
    --ink2: #8b949e;
    --ink3: #6e7681;
    --orange: #f27a1a;
    --orange-dim: rgba(242,122,26,.12);
    --orange-glow: rgba(242,122,26,.25);
    --green: #3fb950;
    --green-dim: rgba(63,185,80,.12);
    --red: #f85149;
    --red-dim: rgba(248,81,73,.12);
    --blue: #58a6ff;
    --radius: 10px;
    --mono: 'JetBrains Mono', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: Inter, system-ui, sans-serif;
    background: var(--bg);
    color: var(--ink);
    min-height: 100vh;
    line-height: 1.5;
  }

  /* ── LAYOUT ── */
  .shell { max-width: 1200px; margin: 0 auto; padding: 28px 20px 60px; }

  /* ── TOPBAR ── */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; margin-bottom: 28px;
  }
  .brand { display: flex; align-items: center; gap: 14px; }
  .brand-icon {
    width: 44px; height: 44px; border-radius: 10px;
    background: linear-gradient(135deg, #f27a1a, #e05a00);
    display: grid; place-items: center;
    font-weight: 800; font-size: 15px; color: #fff;
    box-shadow: 0 0 20px var(--orange-glow);
  }
  .brand-text h1 { font-size: 20px; font-weight: 700; letter-spacing: -.3px; }
  .brand-text p { font-size: 13px; color: var(--ink2); margin-top: 2px; }

  .status-chip {
    display: flex; align-items: center; gap: 7px;
    font-size: 12px; color: var(--ink2);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 999px; padding: 6px 12px;
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--ink3); transition: background .3s;
  }
  .status-dot.active { background: var(--orange); box-shadow: 0 0 8px var(--orange); animation: pulse 1.4s infinite; }
  .status-dot.done { background: var(--green); box-shadow: 0 0 8px var(--green); animation: none; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── INPUT ── */
  .input-row {
    display: flex; gap: 10px; margin-bottom: 20px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 10px;
  }
  .url-input {
    flex: 1; background: var(--bg); border: 1px solid var(--border2);
    border-radius: 8px; padding: 0 14px; height: 44px; color: var(--ink);
    font: inherit; font-size: 14px; outline: none; transition: border-color .2s;
  }
  .url-input:focus { border-color: var(--orange); box-shadow: 0 0 0 3px var(--orange-dim); }
  .url-input::placeholder { color: var(--ink3); }

  .btn-primary {
    background: var(--orange); color: #fff; border: none;
    border-radius: 8px; padding: 0 20px; height: 44px;
    font: inherit; font-size: 14px; font-weight: 700;
    cursor: pointer; white-space: nowrap;
    transition: filter .15s, transform .1s;
    box-shadow: 0 0 16px var(--orange-glow);
  }
  .btn-primary:hover { filter: brightness(1.1); }
  .btn-primary:active { transform: scale(.98); }
  .btn-primary:disabled { opacity: .5; cursor: wait; }

  /* ── PROGRESS ── */
  .progress-bar-wrap {
    display: none; height: 3px; background: var(--surface2);
    border-radius: 99px; margin-bottom: 20px; overflow: hidden;
  }
  .progress-bar-wrap.active { display: block; }
  .progress-bar {
    height: 100%; width: 0%; background: var(--orange);
    border-radius: inherit; transition: width .4s ease;
    animation: indeterminate 1.6s ease-in-out infinite;
  }
  @keyframes indeterminate {
    0%{transform:translateX(-100%) scaleX(.5)}
    100%{transform:translateX(300%) scaleX(.5)}
  }

  /* ── NOTICE ── */
  .notice {
    display: none; padding: 12px 16px; border-radius: 8px;
    background: rgba(248,81,73,.1); border: 1px solid rgba(248,81,73,.3);
    color: #f85149; font-size: 13px; margin-bottom: 20px;
  }

  /* ── EMPTY STATE ── */
  .empty-state {
    border: 1px dashed var(--border); border-radius: var(--radius);
    padding: 60px 32px; text-align: center; color: var(--ink3);
  }
  .empty-state-icon { font-size: 36px; margin-bottom: 12px; }
  .empty-state p { font-size: 14px; max-width: 360px; margin: 0 auto; }

  /* ── DASHBOARD ── */
  .dashboard { display: none; }
  .dashboard.ready { display: block; }

  /* ── METRICS ── */
  .metrics {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px; margin-bottom: 20px;
  }
  .metric {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: var(--radius); padding: 16px 18px;
    transition: border-color .2s;
  }
  .metric:hover { border-color: var(--orange); }
  .metric-label { font-size: 12px; color: var(--ink2); margin-bottom: 10px; text-transform: uppercase; letter-spacing: .5px; }
  .metric-value { font-family: var(--mono); font-size: 28px; font-weight: 500; line-height: 1; }
  .metric-value.positive { color: var(--green); }
  .metric-value.negative { color: var(--red); }
  .metric-value.orange { color: var(--orange); }

  /* ── MAIN GRID ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 16px;
  }

  /* ── PANEL ── */
  .panel {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: var(--radius); overflow: hidden;
  }
  .panel-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; border-bottom: 1px solid var(--border2);
  }
  .panel-head h2 { font-size: 14px; font-weight: 600; }
  .panel-head .label { font-size: 12px; color: var(--ink2); font-family: var(--mono); }
  .panel-body { padding: 18px; }

  /* ── GAUGE ── */
  .gauge-wrap { display: flex; flex-direction: column; align-items: center; margin-bottom: 20px; }
  .gauge-svg { overflow: visible; display: block; }
  .gauge-legend {
    display: flex; gap: 20px; margin-top: 10px; font-size: 13px;
  }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: middle; }

  /* ── FEATURED REVIEWS ── */
  .review-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .review-col {
    background: var(--surface2); border: 1px solid var(--border2);
    border-radius: 8px; padding: 14px;
    display: flex; flex-direction: column;
  }
  .review-col h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 10px; flex-shrink: 0; }
  .review-col h3.pos { color: var(--green); }
  .review-col h3.neg { color: var(--red); }
  .review-list {
    list-style: none; display: flex; flex-direction: column; gap: 8px;
    max-height: 280px; overflow-y: auto;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
  .review-item {
    font-size: 12px; color: var(--ink2); padding: 10px 12px;
    background: var(--bg); border-radius: 6px; border-left: 2px solid var(--border);
    line-height: 1.55; white-space: normal; word-break: break-word;
    cursor: default;
    /* Show full text, no truncation */
  }
  .review-item.pos { border-left-color: var(--green); }
  .review-item.neg { border-left-color: var(--red); }
  .review-item:hover { background: var(--surface2); color: var(--ink); }

  /* ── CHAT PANEL ── */
  .chat-panel {
    display: flex; flex-direction: column;
    min-height: 540px; max-height: 700px;
  }
  .chat-head-info { display: flex; align-items: center; gap: 10px; }
  .ai-avatar {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, var(--orange), #c05a00);
    display: grid; place-items: center; font-size: 12px; font-weight: 800;
    box-shadow: 0 0 12px var(--orange-glow);
  }
  .online-badge {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 8px var(--green);
  }

  .chat-log {
    flex: 1; overflow-y: auto; padding: 14px;
    display: flex; flex-direction: column; gap: 10px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }

  .msg {
    max-width: 88%; padding: 10px 13px; border-radius: 10px;
    font-size: 13px; line-height: 1.5; white-space: pre-wrap;
  }
  .msg.bot {
    align-self: flex-start; background: var(--surface2);
    border: 1px solid var(--border2); color: var(--ink);
  }
  .msg.user {
    align-self: flex-end; background: var(--orange);
    color: #fff; border: none;
  }
  .msg.typing { color: var(--ink3); font-style: italic; }

  .chat-footer {
    padding: 12px; border-top: 1px solid var(--border2);
    display: flex; gap: 8px;
  }
  .chat-input {
    flex: 1; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0 12px; height: 40px;
    color: var(--ink); font: inherit; font-size: 13px; outline: none;
    transition: border-color .2s;
  }
  .chat-input:focus { border-color: var(--orange); }
  .chat-input::placeholder { color: var(--ink3); }
  .chat-input:disabled { opacity: .4; }

  .btn-send {
    background: var(--orange); color: #fff; border: none;
    border-radius: 8px; padding: 0 16px; height: 40px;
    font: inherit; font-size: 13px; font-weight: 700;
    cursor: pointer; transition: filter .15s;
  }
  .btn-send:hover { filter: brightness(1.1); }
  .btn-send:disabled { opacity: .4; cursor: not-allowed; }

  /* ── RESPONSIVE ── */
  @media (max-width: 900px) {
    .main-grid { grid-template-columns: 1fr; }
    .metrics { grid-template-columns: repeat(3, 1fr); }
    .chat-panel { min-height: 420px; }
  }
  @media (max-width: 600px) {
    .topbar { flex-direction: column; align-items: flex-start; }
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .review-cols { grid-template-columns: 1fr; }
    .input-row { flex-direction: column; }
  }
</style>
</head>
<body>
<div class="shell">

  <!-- TOPBAR -->
  <header class="topbar">
    <div class="brand">
      <div class="brand-icon">TY</div>
      <div class="brand-text">
        <h1>Trendyol Yorum Analizi</h1>
        <p>Yorumları analiz eder, özetler ve sorularınızı yanıtlar</p>
      </div>
    </div>
    <div class="status-chip">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Bekleniyor</span>
    </div>
  </header>

  <!-- INPUT -->
  <div class="input-row">
    <input class="url-input" id="productUrl" type="url" placeholder="Trendyol ürün URL'sini yapıştırın…">
    <button class="btn-primary" id="analyzeBtn">Analiz Et</button>
  </div>

  <div class="progress-bar-wrap" id="progressWrap"><div class="progress-bar"></div></div>
  <div class="notice" id="notice"></div>

  <!-- EMPTY STATE -->
  <div class="empty-state" id="emptyState">
    <div class="empty-state-icon">🔍</div>
    <p>Bir ürün URL'si girin — yorum dağılımı, öne çıkan müşteri görüşleri ve AI sohbet alanı burada açılacak.</p>
  </div>

  <!-- DASHBOARD -->
  <div class="dashboard" id="dashboard">

    <!-- METRICS -->
    <div class="metrics">
      <div class="metric">
        <div class="metric-label">Toplam Yorum</div>
        <div class="metric-value" id="mTotal">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Ort. Puan</div>
        <div class="metric-value orange" id="mRating">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Olumlu</div>
        <div class="metric-value positive" id="mPos">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Olumsuz</div>
        <div class="metric-value negative" id="mNeg">—</div>
      </div>
      <div class="metric">
        <div class="metric-label">Güven</div>
        <div class="metric-value orange" id="mConf">—</div>
      </div>
    </div>

    <!-- MAIN GRID -->
    <div class="main-grid">

      <!-- LEFT: analysis panel -->
      <div class="panel">
        <div class="panel-head">
          <h2>Duygu Analizi Raporu</h2>
          <span class="label" id="productIdLabel"></span>
        </div>
        <div class="panel-body">

          <!-- GAUGE -->
          <div class="gauge-wrap">
            <svg class="gauge-svg" width="260" height="155" viewBox="0 0 260 155">
              <defs>
                <linearGradient id="posGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" style="stop-color:#2ea043"/>
                  <stop offset="100%" style="stop-color:#56d364"/>
                </linearGradient>
                <linearGradient id="negGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" style="stop-color:#f85149"/>
                  <stop offset="100%" style="stop-color:#da3633"/>
                </linearGradient>
              </defs>
              <!-- Full track -->
              <path fill="none" stroke="#1c2330" stroke-width="16" stroke-linecap="butt"
                    d="M25,135 A105,105 0 0,1 235,135"/>
              <!-- Positive arc (left->top) -->
              <path id="gaugePos" fill="none" stroke="url(#posGrad)" stroke-width="16" stroke-linecap="round"
                    d="M25,135 A105,105 0 0,1 235,135"
                    style="stroke-dasharray:330; stroke-dashoffset:330; transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)"/>
              <!-- Negative arc (right->top, reversed) -->
              <path id="gaugeNeg" fill="none" stroke="url(#negGrad)" stroke-width="16" stroke-linecap="round"
                    d="M235,135 A105,105 0 0,0 25,135"
                    style="stroke-dasharray:330; stroke-dashoffset:330; transition:stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1) .15s"/>
              <!-- Needle -->
              <line id="gaugeNeedle" x1="130" y1="135" x2="130" y2="42"
                    stroke="#c9d1d9" stroke-width="2.5" stroke-linecap="round"
                    style="transform-origin:130px 135px; transform:rotate(-90deg); transition:transform 1.2s cubic-bezier(.4,0,.2,1) .1s"/>
              <circle cx="130" cy="135" r="5" fill="#c9d1d9"/>
              <!-- Center label -->
              <text id="gaugeCenter" x="130" y="116" text-anchor="middle"
                    font-size="20" font-weight="700" fill="#e6edf3"
                    font-family="JetBrains Mono,monospace">—</text>
              <text id="gaugeSub" x="130" y="131" text-anchor="middle"
                    font-size="10" fill="#8b949e" font-family="Inter,sans-serif">olumlu</text>
              <!-- Side labels -->
              <text x="8" y="151" font-size="18" fill="#8b949e">😊</text>
              <text x="228" y="151" font-size="18" fill="#8b949e">😕</text>
            </svg>
            <div class="gauge-legend">
              <span><span class="legend-dot" style="background:var(--green)"></span><span id="legendPos">Olumlu —</span></span>
              <span><span class="legend-dot" style="background:var(--red)"></span><span id="legendNeg">Olumsuz —</span></span>
            </div>
          </div>

          <!-- FEATURED REVIEWS -->
          <div class="review-cols">
            <div class="review-col">
              <h3 class="pos">Öne Çıkan Olumlu</h3>
              <ul class="review-list" id="posReviews"></ul>
            </div>
            <div class="review-col">
              <h3 class="neg">Öne Çıkan Olumsuz</h3>
              <ul class="review-list" id="negReviews"></ul>
            </div>
          </div>

        </div>
      </div>

      <!-- RIGHT: chat -->
      <div class="panel chat-panel">
        <div class="panel-head">
          <div class="chat-head-info">
            <div class="ai-avatar">AI</div>
            <div>
              <h2>Yorumlarla Sohbet</h2>
            </div>
          </div>
          <div class="online-badge" id="onlineBadge" style="opacity:.3"></div>
        </div>
        <div class="chat-log" id="chatLog">
          <div class="msg bot">Analiz tamamlandıktan sonra ürün yorumları hakkında soru sorabilirsin.</div>
        </div>
        <div class="chat-footer">
          <input class="chat-input" id="chatInput" placeholder="Örn: Bu ürün alınır mı?" disabled>
          <button class="btn-send" id="sendBtn" disabled>Gönder</button>
        </div>
      </div>

    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const analyzeBtn = $('analyzeBtn');
const productUrl = $('productUrl');
const progressWrap = $('progressWrap');
const notice = $('notice');
const emptyState = $('emptyState');
const dashboard = $('dashboard');
const chatLog = $('chatLog');
const chatInput = $('chatInput');
const sendBtn = $('sendBtn');

function setStatus(state) {
  const dot = $('statusDot'), txt = $('statusText');
  dot.className = 'status-dot';
  if (state === 'loading') { dot.classList.add('active'); txt.textContent = 'Analiz ediliyor…'; }
  else if (state === 'done') { dot.classList.add('done'); txt.textContent = 'Analiz tamamlandı'; }
  else { txt.textContent = 'Bekleniyor'; }
}

function showNotice(msg) {
  notice.textContent = msg;
  notice.style.display = msg ? 'block' : 'none';
}

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = `msg ${role}`;
  d.textContent = text;
  chatLog.appendChild(d);
  chatLog.scrollTop = chatLog.scrollHeight;
  return d;
}

function setReviewList(id, items, cls) {
  const ul = $(id); ul.innerHTML = '';
  const data = (items && items.length) ? items : ['Yeterli veri bulunamadı.'];
  data.forEach(t => {
    const li = document.createElement('li');
    li.className = `review-item ${cls}`;
    li.textContent = t;
    ul.appendChild(li);
  });
}

function animateGauge(posPercent, negPercent) {
  const ARC_LEN = 330;

  // Positive: left side fills proportionally
  const posOffset = ARC_LEN - (posPercent / 100) * ARC_LEN;
  // Negative: right side fills proportionally
  const negOffset = ARC_LEN - (negPercent / 100) * ARC_LEN;

  setTimeout(() => {
    document.getElementById('gaugePos').style.strokeDashoffset = posOffset;
    document.getElementById('gaugeNeg').style.strokeDashoffset = negOffset;

    // Needle: -90deg = full left (100% pos), +90deg = full right (100% neg)
    // Map posPercent 0-100 to angle -90 to +90
    const angle = ((negPercent - posPercent) / 100) * 90;
    document.getElementById('gaugeNeedle').style.transform = `rotate(${angle}deg)`;

    // Center label
    document.getElementById('gaugeCenter').textContent = `%${posPercent.toFixed(0)}`;
  }, 100);

  $('legendPos').textContent = `Olumlu %${posPercent.toFixed(1)}`;
  $('legendNeg').textContent = `Olumsuz %${negPercent.toFixed(1)}`;
}

function renderDashboard(data) {
  const s = data.summary;
  const sent = s.sentiment;

  $('mTotal').textContent = s.total_reviews.toLocaleString('tr');
  $('mRating').textContent = s.average_rating.toFixed(2);
  $('mPos').textContent = `%${sent.positive.percent.toFixed(1)}`;
  $('mNeg').textContent = `%${sent.negative.percent.toFixed(1)}`;
  $('mConf').textContent = `%${s.confidence_score.toFixed(1)}`;
  $('productIdLabel').textContent = `ID: ${s.content_id}`;

  animateGauge(sent.positive.percent, sent.negative.percent);
  setReviewList('posReviews', s.featured_positive, 'pos');
  setReviewList('negReviews', s.featured_negative, 'neg');

  // unlock chat
  chatLog.innerHTML = '';
  addMsg('bot', 'Analiz tamamlandı! Bu ürünün yorumlarına dayanarak soru sorabilirsin. 💬');
  chatInput.disabled = false;
  sendBtn.disabled = false;
  $('onlineBadge').style.opacity = '1';

  emptyState.style.display = 'none';
  dashboard.classList.add('ready');
  setStatus('done');
}

// ── ANALYZE ──
analyzeBtn.addEventListener('click', async () => {
  const url = productUrl.value.trim();
  if (!url) return;

  showNotice('');
  analyzeBtn.disabled = true;
  progressWrap.classList.add('active');
  setStatus('loading');
  chatInput.disabled = true;
  sendBtn.disabled = true;
  $('onlineBadge').style.opacity = '.3';

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Analiz başarısız.');
    renderDashboard(data);
  } catch (e) {
    showNotice(e.message);
    setStatus('idle');
  } finally {
    analyzeBtn.disabled = false;
    progressWrap.classList.remove('active');
  }
});

productUrl.addEventListener('keydown', e => { if (e.key === 'Enter') analyzeBtn.click(); });

// ── CHAT ──
async function sendMessage() {
  const q = chatInput.value.trim();
  if (!q) return;
  addMsg('user', q);
  chatInput.value = '';
  sendBtn.disabled = true;
  const pending = addMsg('bot typing', 'Yanıt hazırlanıyor…');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Yanıt alınamadı.');
    pending.className = 'msg bot';
    pending.textContent = data.answer;
  } catch (e) {
    pending.className = 'msg bot';
    pending.textContent = '⚠️ ' + e.message;
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
    chatLog.scrollTop = chatLog.scrollHeight;
  }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });
</script>
</body>
</html>"""


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return render_template_string(HTML)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]

    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL boş olamaz."}), 400

    try:
        result = analyze_product(url)
        session_store[sid] = result
        return jsonify({"summary": result["summary"]})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if "sid" not in session:
        return jsonify({"error": "Önce bir ürün analizi yapmalısınız."}), 400

    sid = session["sid"]
    analysis = session_store.get(sid)
    if not analysis:
        return jsonify({"error": "Önce bir ürün analizi yapmalısınız."}), 400

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Lütfen bir soru yazın."}), 400

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": analysis["prompt"]},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return jsonify({"answer": res.choices[0].message.content})
    except Exception as exc:
        import traceback

        print("\n" + "=" * 80)
        traceback.print_exc()
        print("=" * 80)
        print("Exception type:", type(exc))
        print("Exception repr:", repr(exc))
        print("=" * 80)

        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    # Pre-load sentiment pipeline at startup to avoid first-request delay
    print("Sentiment modeli yükleniyor…")
    get_sentiment_pipeline()
    print("Model hazır.")
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False)
