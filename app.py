import os
import re
import glob
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import ast

app = Dash(__name__)
data_dir = "data"
cached_docs_df = pd.DataFrame()
cached_topic_labels = {}

# === Detect targets ===
available_targets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# === Detect available k values ===
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

# === App Layout ===
app.layout = html.Div([
    html.H2("BERTopic Sunburst Explorer"),

    html.Div([
        html.Div([
            html.Label("Dataset"),
            dcc.Dropdown(available_targets, id="target", value=available_targets[0], clearable=False),
            html.Label("Positive k"),
            dcc.Dropdown([str(k) for k in available_k], id="k_positive", value=str(available_k[0]), clearable=False),
            html.Label("Negative k"),
            dcc.Dropdown([str(k) for k in available_k], id="k_negative", value=str(available_k[0]), clearable=False),
        ], style={"width": "30%", "display": "inline-block", "margin": "10px"}),

        html.Div([
            html.H4("Prime k values"),
            html.Ul([
                html.Li("BT (Brian Thompson): Positive k = 9, Negative k = 7"),
                html.Li("LM (Luigi Mangione): Positive k = 8, Negative k = 8")
            ])
        ], style={"width": "60%", "display": "inline-block", "verticalAlign": "top", "margin": "10px"})
    ]),

    dcc.Graph(id="sunburst"),
    html.Div(id="custom_legend", style={"padding": "20px"}),

    html.Div([
        html.Label("Number of Tweets to Display:"),
        dcc.Dropdown(
            options=[{"label": str(n), "value": n} for n in [5, 10, 20, 50, 100, "All"]],
            value=10,
            id="tweet_count_dropdown",
            clearable=False
        )
    ], style={"width": "30%", "margin": "10px"}),

    html.Div(id="tweet_output", style={"padding": "20px", "maxHeight": "500px", "overflowY": "scroll", "borderTop": "1px solid #ccc"})
])

# === Color Palette ===
cb_palette = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288",
              "#AA4499", "#44AA99", "#999933", "#882255", "#661100",
              "#6699CC", "#888888", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# === Load Merged CSV ===
def load_data(base_path, sentiment, target, k):
    doc_pattern = f"{sentiment.lower()}_{target}_*k={k}*document_info.csv"
    rep_pattern = f"{sentiment.lower()}_{target}_*k={k}*topic_representation.csv"

    doc_files = glob.glob(os.path.join(base_path, doc_pattern))
    rep_files = glob.glob(os.path.join(base_path, rep_pattern))

    if not doc_files or not rep_files:
        print("❌ Missing files for", sentiment, target, k)
        return pd.DataFrame()

    try:
        docs = pd.read_csv(doc_files[0])
        reps = pd.read_csv(rep_files[0])
    except pd.errors.EmptyDataError:
        print("❌ Empty file detected")
        return pd.DataFrame()

    docs["Sentiment"] = sentiment
    reps["Sentiment"] = sentiment

    docs["Topic"] = pd.to_numeric(docs["Topic"], errors="coerce")
    reps["Topic"] = pd.to_numeric(reps["Topic"], errors="coerce")

    reps["Representation"] = reps["Representation"].apply(
    lambda x: ", ".join(ast.literal_eval(x)) if isinstance(x, str) and x.startswith("[") else str(x)
)


    merged = pd.merge(docs, reps[["Topic", "Representation"]], on="Topic", how="left")
    return merged

# === Sunburst Callback ===
@app.callback(
    Output("sunburst", "figure"),
    Output("custom_legend", "children"),
    Input("target", "value"),
    Input("k_positive", "value"),
    Input("k_negative", "value")
)
def update_sunburst(target, k_positive, k_negative):
    global cached_docs_df, cached_topic_labels
    base_path = os.path.join(data_dir, target)
    sentiments = ["Positive", "Negative"]
    all_docs = []

    for sentiment in sentiments:
        k_val = k_positive if sentiment == "Positive" else k_negative
        docs = load_data(base_path, sentiment, target, k_val)
        if not docs.empty:
            all_docs.append(docs)

    if not all_docs:
        return go.Figure().update_layout(title="No data found."), ""

    docs_df = pd.concat(all_docs)
    cached_docs_df = docs_df.copy()
    docs_df["Topic"] = pd.to_numeric(docs_df["Topic"], errors="coerce")
    docs_df["Topic_ID"] = docs_df["Sentiment"] + "_" + docs_df["Topic"].astype(str)

    if "Representation_y" in docs_df.columns:
        topic_labels = docs_df[["Sentiment", "Topic", "Representation_y"]].drop_duplicates()
        topic_labels = topic_labels.rename(columns={"Representation_y": "Representation"})
    else:
        topic_labels = docs_df[["Sentiment", "Topic", "Representation"]].drop_duplicates()

    cached_topic_labels.clear()
    cached_topic_labels.update({
        (row.Sentiment, row.Topic): row.Representation for _, row in topic_labels.iterrows()
    })


    docs_df["Clean_Label"] = docs_df.apply(lambda row: cached_topic_labels.get((row["Sentiment"], row["Topic"]), f"Topic {row['Topic']}"), axis=1)

    topic_counts = docs_df.groupby(["Sentiment", "Topic_ID", "Topic", "Clean_Label"]).size().reset_index(name="Doc_Count")
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
        label = row["Clean_Label"]
        tid = row["Sentiment"] + "_" + str(row["Topic"])
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
        title=f"Sunburst: {target} | k+ = {k_positive}, k- = {k_negative}"
    )

    legend_items = [
        html.Div([
            html.Span(style={"display": "inline-block", "width": "15px", "height": "15px", "backgroundColor": color_map[label], "marginRight": "8px"}),
            label
        ], style={"marginBottom": "5px"}) for label in color_map
    ]

    return fig, legend_items

# === Click Tweet Output ===
@app.callback(
    Output("tweet_output", "children"),
    Input("sunburst", "clickData"),
    State("tweet_count_dropdown", "value")
)
def display_tweets(clickData, tweet_count):
    if not clickData or "label" not in clickData["points"][0]:
        return ""

    label = clickData["points"][0]["label"]
    sentiment = clickData["points"][0]["parent"]

    def normalize(text):
        return str(text).strip().lower()

    label_norm = normalize(label)

    matching_topics = [
        k for k, v in cached_topic_labels.items()
        if normalize(v) == label_norm and k[0] == sentiment
    ]

    if not matching_topics:
        return html.Div(["\u274C No topic match found."], style={"color": "red", "fontSize": "20px", "fontWeight": "bold"})

    topic_num = matching_topics[0][1]
    df = cached_docs_df[(cached_docs_df["Topic"] == topic_num) & (cached_docs_df["Sentiment"] == sentiment)]

    for col in ["original_text", "Document", "text", "content"]:
        if col in df.columns:
            text_column = col
            break
    else:
        return html.Div(["\u274C No valid text column found."], style={"color": "red"})

    if tweet_count != "All":
        df = df.head(int(tweet_count))

    tweet_list = [html.Li(tweet) for tweet in df[text_column]]
    return html.Div([
        html.H5(f"Top {tweet_count} Tweets for Topic {topic_num}"),
        html.Ul(tweet_list)
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)







