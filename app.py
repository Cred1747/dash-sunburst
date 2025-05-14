import os
import re
import glob
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)
data_dir = "data"

# Detect available targets
available_targets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Detect available k values
def extract_k_values():
    k_set = set()
    for target in available_targets:
        files = os.listdir(os.path.join(data_dir, target))
        for f in files:
            match = re.search(r'k=(\d+)', f)
            if match:
                k_set.add(int(match.group(1)))
    return sorted(k_set)

available_k = extract_k_values()

# === Layout ===
app.layout = html.Div([
    html.H2("BERTopic Sunburst Explorer"),

    html.Div([
        dcc.Dropdown(available_targets, id="target", value=available_targets[0], clearable=False),
        dcc.Dropdown([str(k) for k in available_k], id="k", value=str(available_k[0]), clearable=False),
        dcc.Dropdown(["Both", "Positive", "Negative"], id="sentiment", value="Both", clearable=False),
    ], style={"width": "30%", "display": "inline-block", "margin": "10px"}),

    dcc.Graph(id="sunburst")
])

# === Color palette ===
cb_palette = [
    "#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288",
    "#AA4499", "#44AA99", "#999933", "#882255", "#661100",
    "#6699CC", "#888888", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]

# === Load function (uses wildcards to match real filenames) ===
def load_data(base_path, sentiment, target, k):
    doc_pattern = f"{sentiment.lower()}_{target}_*k={k}*document_info.csv"
    rep_pattern = f"{sentiment.lower()}_{target}_*k={k}*topic_representation.csv"
    doc_files = glob.glob(os.path.join(base_path, doc_pattern))
    rep_files = glob.glob(os.path.join(base_path, rep_pattern))

    if not doc_files or not rep_files:
        return pd.DataFrame(), pd.DataFrame()

    docs = pd.read_csv(doc_files[0])
    reps = pd.read_csv(rep_files[0])
    docs["Sentiment"] = sentiment
    reps["Sentiment"] = sentiment
    reps["Topic"] = pd.to_numeric(reps["Topic"], errors="coerce")
    return docs, reps

# === Dash Callback ===
@app.callback(
    Output("sunburst", "figure"),
    Input("target", "value"),
    Input("k", "value"),
    Input("sentiment", "value")
)
def update_sunburst(target, k, selected_sentiment):
    base_path = os.path.join(data_dir, target)
    sentiments = ["Positive", "Negative"] if selected_sentiment == "Both" else [selected_sentiment]

    all_docs, all_topics = [], []

    for sentiment in sentiments:
        docs, topics = load_data(base_path, sentiment, target, k)
        if not docs.empty and not topics.empty:
            all_docs.append(docs)
            all_topics.append(topics)

    if not all_docs or not all_topics:
        return go.Figure().update_layout(title="No data found.")

    docs_df = pd.concat(all_docs)
    topics_df = pd.concat(all_topics)

    # Label cleaning
    def clean_label(label, top_n=4):
        try:
            parts = label.strip("[]").replace("'", "").split(", ")
            return ", ".join(parts[:top_n])
        except:
            return "Unknown"

    topics_df["Clean_Label"] = topics_df["Representation"].astype(str).apply(clean_label)
    topics_df["Topic_ID"] = topics_df["Sentiment"] + "_" + topics_df["Topic"].astype(str)

    topic_map = dict(zip(zip(topics_df["Sentiment"], topics_df["Topic"]), topics_df["Clean_Label"]))
    topic_id_map = dict(zip(zip(topics_df["Sentiment"], topics_df["Topic"]), topics_df["Topic_ID"]))

    docs_df["Topic"] = pd.to_numeric(docs_df["Topic"], errors="coerce")
    docs_df["Topic_Label"] = docs_df.apply(lambda row: topic_map.get((row["Sentiment"], row["Topic"]), "Unknown"), axis=1)
    docs_df["Topic_ID"] = docs_df.apply(lambda row: topic_id_map.get((row["Sentiment"], row["Topic"]), "Unknown"), axis=1)

    topic_counts = docs_df.groupby(["Sentiment", "Topic_ID", "Topic_Label"]).size().reset_index(name="Doc_Count")
    sentiment_counts = docs_df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Doc_Count"]

    # Build sunburst
    labels, ids, parents, values, colors = [], [], [], [], []
    color_map = {}
    color_idx = 0

    for _, row in sentiment_counts.iterrows():
        sentiment = row["Sentiment"]
        labels.append(sentiment)
        ids.append(sentiment)
        parents.append("")
        values.append(row["Doc_Count"])
        colors.append("lightblue" if sentiment == "Negative" else "salmon")

    for _, row in topic_counts.iterrows():
        label = row["Topic_Label"]
        tid = row["Topic_ID"]
        sentiment = row["Sentiment"]
        doc_count = row["Doc_Count"]

        labels.append(label)
        ids.append(tid)
        parents.append(sentiment)
        values.append(doc_count)

        if label not in color_map:
            color_map[label] = cb_palette[color_idx % len(cb_palette)]
            color_idx += 1
        colors.append(color_map[label])

    fig = go.Figure(go.Sunburst(
        labels=labels,
        ids=ids,
        parents=parents,
        values=values,
        marker=dict(colors=colors),
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Posts: %{value}<extra></extra>'
    ))

    fig.update_layout(
        margin=dict(t=40, l=40, r=40, b=40),
        title=f"Sunburst: {target} | k = {k} | Sentiment = {selected_sentiment}"
    )

    return fig

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8000, debug=True)




