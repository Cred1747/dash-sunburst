# BERTopic Sunburst Explorer

An interactive Dash app for exploring stance-based BERTopic results by sentiment and topic label.  
Click on sunburst segments to view representative tweets by topic and sentiment.

🚀 **Live App**:  
👉 [https://dash-sunburst-production.up.railway.app/](https://dash-sunburst-production.up.railway.app/)

---

## Features

- Select dataset (`BT`, `LM`, etc.)
- Customize number of topics (`k`) for Positive and Negative sentiment
- Clickable sunburst chart with custom color palette
- Tweet preview panel filtered by topic and sentiment
- Dynamic legend and hover tooltips

...

## Setup Locally

```bash
git clone https://github.com/YOUR_USERNAME/bertopic-sunburst.git
cd bertopic-sunburst
pip install -r requirements.txt
python app.py
