import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

import time
import random
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import numpy as np
from serpapi import GoogleSearch
import plotly.graph_objects as go
from textblob import TextBlob
import plotly.express as px
import warnings
import requests
    
import http.client
import urllib.parse
import json
import plotly.express as px

API_HOST = "semrush-magic-tool.p.rapidapi.com"
API_KEY = "4908184efbmsh6b629e2c8488a44p184b49jsn9cfd845aacfd"

warnings.filterwarnings("ignore", category=FutureWarning)
SERPAPI_KEY = "9c24c825ae5ccf7ef6e534d6d0f28c7dc13f72171b31c5d7a73b548b00774467"  # <-- Replace with your SerpApi key

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]








    
def plot_keyword_cluster_graph(keyword, related_df):
    if related_df is None or related_df.empty or "Keyword" not in related_df.columns:
        return None

    # Prepare nodes and edges
    nodes = [keyword] + related_df["Keyword"].tolist()
    edges = [(keyword, k) for k in related_df["Keyword"]]

    # Assign positions in a circle
    import math
    n = len(nodes)
    angle_step = 2 * math.pi / max(n-1, 1)
    positions = {keyword: (0, 0)}
    for i, k in enumerate(related_df["Keyword"]):
        angle = i * angle_step
        positions[k] = (math.cos(angle), math.sin(angle))

    # Create edges for Plotly
    edge_x = []
    edge_y = []
    for src, dst in edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Create nodes for Plotly
    node_x = []
    node_y = []
    node_text = []
    for k in nodes:
        x, y = positions[k]
        node_x.append(x)
        node_y.append(y)
        node_text.append(k)

    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=30, color='skyblue'),
        text=node_text,
        textposition="bottom center",
        hoverinfo='text'
    ))

    fig.update_layout(
        showlegend=False,
        title="Keyword Cluster Graph",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
def create_pytrends_session():
    session = TrendReq(hl='en-US', tz=330)
    return session


def random_user_agent():
    return random.choice(user_agents)
def fetch_trends_data_pytrends(keyword):
    pytrends = create_pytrends_session()
    time.sleep(random.uniform(1, 2))
    timeframes = ['today 12-m', 'today 5-y', 'today 3-m', 'today 1-m']
    time_df = pd.DataFrame()
    for timeframe in timeframes:
        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            time_df = pytrends.interest_over_time()
            if not time_df.empty:
                break
        except Exception:
            continue
    if time_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if 'isPartial' in time_df.columns:
        time_df = time_df.drop('isPartial', axis=1)
    # Country-Level Interest
    country_df = pd.DataFrame()
    try:
        pytrends.build_payload([keyword], timeframe=timeframe)
        country_data = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
        if not country_data.empty:
            country_df = country_data.sort_values(by=keyword, ascending=False).head(10).reset_index()
            country_df = country_df.rename(columns={"geoName": "Country", keyword: "Interest"})
            if 'Country' not in country_df.columns:
                country_df["Country"] = country_df.index
        else:
            country_df = pd.DataFrame([{"Country": "No data available", "Interest": 0}])
    except Exception:
        country_df = pd.DataFrame([{"Country": "Error fetching data", "Interest": 0}])
    return time_df, country_df

def fetch_related_queries_serpapi(keyword):
    params = {
        "engine": "google_trends",
        "q": keyword,
        "api_key": SERPAPI_KEY,
        "data_type": "RELATED_QUERIES"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    queries = results.get('related_queries', [])
    # If queries is a dict, try to extract the values
    if isinstance(queries, dict):
        for key in ['top', 'rising']:
            if key in queries and isinstance(queries[key], list) and queries[key]:
                queries = queries[key]
                break
        else:
            queries = []
    if not isinstance(queries, list):
        queries = []
    filtered = []
    for q in queries:
        if 'query' in q and 'value' in q:
            filtered.append({'Keyword': q['query'], 'Popularity': q['value']})
    if not filtered:
        filtered = [{"Keyword": "No related queries found", "Popularity": ""}]
    return pd.DataFrame(filtered)

def fetch_news_serpapi(keyword, num_headlines=5):
    # Tech context mapping for ambiguous terms
    tech_context = {
        "python": "python programming",
        "java": "java programming",
        "apple": "apple technology",
        "amazon": "amazon web services",
        "tesla": "tesla technology",
        "facebook": "facebook social media",
        "google": "google technology",
        "windows": "microsoft windows",
        "cloud": "cloud computing",
        "aws": "amazon web services",
        "azure": "microsoft azure",
        "oracle": "oracle database",
        "android": "android os",
        "ios": "ios apple",
        "linux": "linux os",
        "openai": "openai artificial intelligence",
        "chatgpt": "chatgpt ai",
        "microsoft": "microsoft technology",
        "meta": "meta facebook",
        "blockchain": "blockchain technology",
        "bitcoin": "bitcoin cryptocurrency",
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "data": "data science",
        "sql": "sql database",
        "docker": "docker container",
        "kubernetes": "kubernetes container",
        "node": "node.js",
        "react": "react js",
        "vue": "vue js",
        "angular": "angular js",
        "flutter": "flutter app",
        "swift": "swift programming",
        "go": "go programming",
        "ruby": "ruby programming",
        "php": "php programming",
        "c++": "c++ programming",
        "c#": "c# programming",
        "typescript": "typescript programming",
        "javascript": "javascript programming",
    }
    search_term = tech_context.get(keyword.lower(), f"{keyword} technology")
    params = {
        "engine": "google_news",
        "q": search_term,
        "api_key": SERPAPI_KEY,
        "num": num_headlines
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    headlines = []
    tech_words = [
        "tech", "software", "hardware", "programming", "developer", "engineer", "app", "AI", "ML", "cloud",
        "data", "database", "web", "platform", "startup", "robot", "cyber", "digital", "IT", "comput", "code",
        "open source", "release", "update", "launch", "security", "network", "server", "API", "framework",
        "tool", "system", "OS", "operating system", "device", "smart", "gadget", "blockchain", "crypto",
        "virtual", "augmented", "machine learning", "artificial intelligence", "python", "java", "javascript",
        "typescript", "c++", "c#", "php", "ruby", "swift", "go", "node", "react", "vue", "angular", "flutter"
    ]
    for article in results.get('news_results', []):
        title = article.get('title', '')
        if any(word.lower() in title.lower() for word in tech_words):
            headlines.append(title)
        if len(headlines) >= num_headlines:
            break
    if not headlines:
        headlines = [f"No tech news found for '{keyword}'."]
    return pd.DataFrame({"Headline": headlines})



def get_builtwith_tech_stack(domain):
    try:
        domain = domain.strip().replace("https://", "").replace("http://", "").replace("www.", "")
        if not domain or '.' not in domain:
            if not domain:
                return pd.DataFrame([{"Technology": "❌ Invalid domain", "Category": "Error"}])
            domain = f"{domain}.com"
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--user-agent={random.choice(user_agents)}")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)
        url = f"https://builtwith.com/{domain}"
        driver.get(url)
        time.sleep(random.uniform(3, 7))
        tech_stack = []
        selectors = [
            "div.card",
            ".tech-item", 
            ".technology-item",
            "[data-tech]",
            "div.row.technology-row",
            "div.tech-card"
        ]
        for selector in selectors:
            try:
                cards = driver.find_elements(By.CSS_SELECTOR, selector)
                if cards:
                    for card in cards:
                        try:
                            category = "Unknown"
                            try:
                                category_elem = card.find_element(By.CSS_SELECTOR, "h5, h4, h3, .category-title")
                                category = category_elem.text.strip()
                                items = card.find_elements(By.CSS_SELECTOR, "ul li, .tech-name, .technology-name")
                                for item in items:
                                    tech_name = item.text.strip()
                                    if tech_name and 2 < len(tech_name) < 50:
                                        tech_name = tech_name.replace("View Global Trends", "").strip()
                                        tech_name = tech_name.replace("Usage Statistics", "").strip()
                                        tech_name = tech_name.replace("Download List", "").strip()
                                        tech_name = tech_name.split(" - ")[0].strip()
                                        tech_name = tech_name.split(" View")[0].strip()
                                        if tech_name and len(tech_name) > 2:
                                            tech_stack.append({"Technology": tech_name, "Category": category})
                            except:
                                try:
                                    links = card.find_elements(By.CSS_SELECTOR, "a")
                                    for link in links:
                                        tech_name = link.text.strip()
                                        if tech_name and 2 < len(tech_name) < 30:
                                            tech_name = tech_name.split(" View")[0].strip()
                                            tech_name = tech_name.split(" -")[0].strip()
                                            excluded_terms = ['Learn More', 'View Details', 'Usage Statistics', 'Global Trends', 'Download']
                                            if not any(term in tech_name for term in excluded_terms):
                                                tech_stack.append({"Technology": tech_name, "Category": "Web Technology"})
                                except:
                                    tech_text = card.text.strip()
                                    if tech_text and 5 < len(tech_text) < 30:
                                        tech_text = tech_text.split('\n')[0].strip()
                                        tech_text = tech_text.split(' View')[0].strip()
                                        tech_text = tech_text.split(' -')[0].strip()
                                        excluded_terms = ['Usage Statistics', 'Global Trends', 'Download', 'View', 'Statistics']
                                        if not any(term in tech_text for term in excluded_terms):
                                            tech_stack.append({"Technology": tech_text, "Category": "General"})
                        except Exception:
                            continue
                    if tech_stack:
                        break
            except Exception:
                continue
        driver.quit()
        if not tech_stack:
            tech_stack.append({"Technology": f"❌ No technologies found for {domain}", "Category": "Not Found"})
        return pd.DataFrame(tech_stack)
    except Exception as e:
        error_msg = f"❌ Error fetching tech stack: {str(e)}"
        return pd.DataFrame([{"Technology": "Error", "Category": error_msg}])

def plot_country_heatmap(country_df, keyword="Keyword"):
    if country_df is None or country_df.empty or "Country" not in country_df.columns:
        # Fallback: show a dummy heatmap with a few countries and zero interest
        dummy_data = pd.DataFrame({
            "Country": ["United States", "India", "United Kingdom", "Germany", "France"],
            "Interest": [0, 0, 0, 0, 0]
        })
        fig = px.choropleth(
            dummy_data,
            locations="Country",
            locationmode="country names",
            color="Interest",
            color_continuous_scale="Viridis",
            title=f"No country data available for '{keyword}'"
        )
        return fig
    fig = px.choropleth(
        country_df,
        locations="Country",
        locationmode="country names",
        color="Interest",
        color_continuous_scale="Viridis",
        title="Interest by Country"
    )
    return fig

def forecast_trend(time_df, keyword, periods=12):
    if time_df is None or time_df.empty or keyword not in time_df.columns:
        return None
    y = time_df[keyword].values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    forecast = model.predict(future_X)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_df.index, y, label="Historical")
    future_dates = pd.date_range(time_df.index[-1], periods=periods+1, freq='W')[1:]
    ax.plot(future_dates, forecast, label="Forecast", linestyle="--")
    ax.set_title(f"Trend Forecast for '{keyword}'")
    ax.legend()
    plt.tight_layout()
    return fig
    

        

def generate_summary(keyword, time_df, country_df, news_headlines=None):
    try:
        if time_df is None or time_df.empty:
            return "⚠ Insufficient data to generate summary."
        peak = time_df[keyword].max()
        peak_date = time_df[keyword].idxmax().strftime("%B %d, %Y")
        top_country = (
            country_df.iloc[0]["Country"]
            if country_df is not None and not country_df.empty and "Country" in country_df.columns
            else "Unknown"
        )
        news_str = ""
        if news_headlines is not None and not news_headlines.empty:
            news_str = "\n\nRecent News Headlines:\n" + "\n".join([f"- {h}" for h in news_headlines['Headline']])
        avg_interest = time_df[keyword].mean()
        trend_direction = "increasing" if time_df[keyword].iloc[-1] > time_df[keyword].iloc[0] else "decreasing"
        return f"""📊 Analysis Summary for '{keyword}'

Peak Performance: Highest interest level of {peak} recorded on {peak_date}

Geographic Leader: {top_country} shows the highest interest

Trend Pattern: The search interest appears to be {trend_direction} over the analyzed period with an average interest level of {avg_interest:.1f}

Key Insights: This keyword shows {'strong' if peak > 70 else 'moderate' if peak > 30 else 'low'} search volume, suggesting {'high' if peak > 70 else 'moderate' if peak > 30 else 'limited'} public interest in this topic.

{news_str}"""
    except Exception as e:
        return f"⚠ Error generating summary: {str(e)}"

def get_keyword_trend_summary(keywords, timeframe="today 12-m"):
    pytrends = create_pytrends_session()
    trend_data = []

    for kw in keywords:  # Limit to 15 to avoid blocking
        try:
            pytrends.build_payload([kw], timeframe=timeframe)
            df = pytrends.interest_over_time()
            if not df.empty and kw in df.columns:
                vals = df[kw].dropna().values
                if len(vals) >= 2:
                    pct_change = ((vals[-1] - vals[0]) / max(vals[0], 1)) * 100
                    trend_emoji = "📈 Rising" if pct_change > 5 else "📉 Falling" if pct_change < -5 else "➖ Stable"
                    trend_data.append({
                        "Keyword": kw,
                        "Trend": trend_emoji,
                        "% Change": f"{pct_change:.1f}%",
                        "Volume Trend": "High" if pct_change > 25 else "Medium" if pct_change > 5 else "Low",
                    })
        except Exception as e:
            trend_data.append({
                "Keyword": kw,
                "Trend": "❌ Error",
                "% Change": "N/A",
                "Volume Trend": str(e)
            })

    return pd.DataFrame(trend_data)
def fetch_keyword_data(keyword):
    conn = http.client.HTTPSConnection(API_HOST)
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': API_HOST
    }
    params = urllib.parse.urlencode({"keyword": keyword, "country": "us"})
    conn.request("GET", f"/keyword-research?{params}", headers=headers)
    res = conn.getresponse()
    data = res.read()
    print("Raw response:", data.decode('utf-8'))  # For debugging
    if res.status != 200:
        return f"❌ Error {res.status}: {data.decode('utf-8')}", None

    try:
        data = json.loads(data)
        print("Parsed JSON:", data)  # For debugging

        # Handle both dict and list responses
        if isinstance(data, dict):
            results = data.get("result", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []

        if not results:
            return "⚠ No data found for the keyword.", None

        top = results[0]
        print("Top result:", top)  # For debugging

        keyword_text = top.get("keyword", "N/A")
        volume = top.get("avg_monthly_searches", "N/A")
        cpc = top.get("High CPC", "N/A")
        competition = top.get("competition_value", "N/A")

        # Build trend DataFrame
        trend_data = top.get("monthly_search_volumes", [])
        if trend_data:
            months = [f"{item['month']} {item['year']}" for item in trend_data]
            searches = [item['searches'] for item in trend_data]
            df = pd.DataFrame({"Month": months, "Search Volume": searches})
            fig = px.line(df, x="Month", y="Search Volume", title=f"📊 Trend for '{keyword_text}'")
        else:
            fig = None

        info = f"""
        ### 🔍 Keyword: {keyword_text}
        - *Search Volume*: {volume}
        - *High CPC*: {cpc}
        - *Competition*: {competition}
        """

        return info, fig
    except Exception as e:
        return f"⚠ Failed to parse response: {str(e)}", None
        
def fetch_all(keyword, want_summary):
    if not keyword or not keyword.strip():
        return (None, None, "❌ Please enter a keyword.", None, None, None, None, None, None)
    keyword = keyword.strip()
    time_df, country_df = fetch_trends_data_pytrends(keyword)
    related_df = fetch_related_queries_serpapi(keyword)
    related_trend_table = get_keyword_trend_summary(related_df['Keyword'].tolist())

    news_df = fetch_news_serpapi(keyword)
    tech_df = get_builtwith_tech_stack(keyword if '.' in keyword else f"{keyword}.com")
    trends_plot = None
    if not time_df.empty and keyword in time_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        time_df[keyword].plot(ax=ax, legend=True, linewidth=2)
        ax.set_title(f"Google Trends Over Time for '{keyword}'", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Interest Level", fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        trends_plot = fig
    country_heatmap = plot_country_heatmap(country_df, keyword)
    forecast_fig = forecast_trend(time_df, keyword)
    summary = generate_summary(keyword, time_df, country_df, news_df) if want_summary else ""
    cluster_fig = plot_keyword_cluster_graph(keyword, related_df) 
    merged_df = pd.merge(related_df, related_trend_table, on="Keyword", how="left")
    semrush_info, semrush_plot = fetch_keyword_data(keyword)
    return (
        trends_plot, country_df, summary, merged_df, tech_df,
        country_heatmap, forecast_fig, news_df, cluster_fig,
        semrush_info, semrush_plot
            )
        




with gr.Blocks() as demo:
    with gr.Tab("Google Trends & Tech Stack"):
        gr.Markdown("# 📊 Google Trends Analyzer + AI Summary + Tech Stack (pytrends + SerpApi Version)")
        gr.Markdown("Enter a keyword to analyze Google Trends data, get the technology stack of related websites, see news, heatmaps, and trend forecasting.")

        with gr.Row():
            keyword_input = gr.Textbox(
                label="Enter a Keyword", 
                placeholder="e.g., amazon, python, tesla",
                value="python"
            )
            summary_toggle = gr.Checkbox(label="Generate AI Summary", value=True)

        submit_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Row():
            trends_plot = gr.Plot(label="📈 Trends Over Time")

        with gr.Row():
            country_table = gr.Dataframe(label="🌍 Top Countries by Interest")

        summary_output = gr.Textbox(label="🤖 AI Summary", lines=6, max_lines=10)
        
        with gr.Row():
            merged_table = gr.Dataframe(label="🔗 Related Queries + Trend Summary")
            tech_stack_table = gr.Dataframe(label="⚙ Technology Stack")

        with gr.Row():
            country_heatmap = gr.Plot(label="🌍 Country Heatmap")
            forecast_plot = gr.Plot(label="🔮 Trend Forecast")
            news_output = gr.Dataframe(label="📰 Recent News Headlines")
        with gr.Row():
            cluster_graph = gr.Plot(label="🔗 Keyword Cluster Graph")
    
        gr.Markdown("## 🔍 SEMrush Keyword Insights (via RapidAPI)")
        with gr.Row():
            semrush_info = gr.Markdown()
            semrush_plot = gr.Plot()
            submit_btn.click(
        fn=fetch_all,
        inputs=[keyword_input, summary_toggle],
        outputs=[
            trends_plot, country_table, summary_output,
            merged_table, tech_stack_table,
            country_heatmap, forecast_plot, news_output,
            cluster_graph,
            semrush_info, semrush_plot
        ]
    )

    
if _name_ == "_main_":
    demo.launch(share=True, debug=True)