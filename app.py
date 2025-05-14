import os
import re
import glob
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

app = Dash(__name__)
data_dir = "data"
cached_docs_df = pd.DataFrame()
cached_topic_labels = {}

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

    dcc.Graph(id="sunburst"),
    html.Div(id="custom_legend", style={"padding": "20px"}),

    html.Div([
        html.Label("Number of Tweets to Display:"),
        dcc.Dropdown(
            options=[
                {"label": str(n), "value": n} for n in [5, 10, 20, 50, 100, "All"]
            ],
            value=10,
            id="tweet_count_dropdown",
            clearable=False
        )
    ], style={"width": "30%", "margin": "10px"}),

    html.Div(id="tweet_output", style={"padding": "20px", "maxHeight": "500px", "overflowY": "scroll", "borderTop": "1px solid #ccc"})
])

# === Color palette ===
cb_palette = [
    "#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288",
    "#AA4499", "#44AA99", "#999933", "#882255", "#661100",
    "#6699CC", "#888888", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]

# === Load function ===
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

@app.callback(
    Output("sunburst", "figure"),
    Output("custom_legend", "children"),
    Input("target", "value"),
    Input("k", "value"),
    Input("sentiment", "value")
)
def update_sunburst(target, k, selected_sentiment):
    global cached_docs_df, cached_topic_labels
    base_path = os.path.join(data_dir, target)
    sentiments = ["Positive", "Negative"] if selected_sentiment == "Both" else [selected_sentiment]

    all_docs, all_topics = [], []

    for sentiment in sentiments:
        docs, topics = load_data(base_path, sentiment, target, k)
        if not docs.empty and not topics.empty:
            all_docs.append(docs)
            all_topics.append(topics)

    if not all_docs or not all_topics:
        return go.Figure().update_layout(title="No data found."), ""

    docs_df = pd.concat(all_docs)
    topics_df = pd.concat(all_topics)
    cached_docs_df = docs_df.copy()

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
    cached_topic_labels = topic_map.copy()

    docs_df["Topic"] = pd.to_numeric(docs_df["Topic"], errors="coerce")
    docs_df["Topic_ID"] = docs_df.apply(lambda row: topic_id_map.get((row["Sentiment"], row["Topic"]), "Unknown"), axis=1)

    topic_counts = docs_df.groupby(["Sentiment", "Topic_ID", "Topic"]).size().reset_index(name="Doc_Count")
    sentiment_counts = docs_df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Doc_Count"]

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
        topic_number = row["Topic"]
        label = topic_map.get((row["Sentiment"], topic_number), str(topic_number))
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
        hovertemplate='<b>%{label}</b><br>Posts: %{value}<extra></extra>',
        textinfo='label+percent entry+value'
    ))

    fig.update_layout(
        margin=dict(t=40, l=40, r=40, b=40),
        title=f"Sunburst: {target} | k = {k} | Sentiment = {selected_sentiment}"
    )

    legend_items = [
        html.Div([
            html.Span(style={"display": "inline-block", "width": "15px", "height": "15px", "backgroundColor": color_map[label], "marginRight": "8px"}),
            label
        ], style={"marginBottom": "5px"}) for label in color_map
    ]

    return fig, legend_items

@app.callback(
    Output("tweet_output", "children"),
    Input("sunburst", "clickData"),
    Input("tweet_count_dropdown", "value")
)
def display_tweets(clickData, tweet_count):
    if not clickData or "label" not in clickData["points"][0]:
        return "Click a topic to view tweets."

    label = clickData["points"][0]["label"]

    if 'cached_docs_df' not in globals() or cached_docs_df.empty:
        return "No data loaded."

    topic_number = None
    for (sentiment, topic), clean in cached_topic_labels.items():
        if clean == label:
            topic_number = topic
            break

    if topic_number is None:
        try:
            topic_number = int(label)
        except ValueError:
            return f"'{label}' is not a numeric topic."

    df = cached_docs_df[cached_docs_df["Topic"] == topic_number]
    if df.empty:
        return f"No tweets found for topic {topic_number}"

    tweet_col = "original_text" if "original_text" in df.columns else df.columns[0]
    tweets = df[tweet_col].dropna().astype(str)

    if tweet_count != "All":
        tweets = tweets.head(int(tweet_count))

    return html.Div([
        html.H4(f"Top Tweets for Topic {topic_number}"),
        html.Ul([html.Li(tweet) for tweet in tweets])
    ])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)







