import os
import time
import json
import urllib.request
import urllib.parse
from xml.etree import ElementTree as ET
from datetime import datetime

# ==============================================================================
# 🔑 THE ULTIMATE KEYWORD MATRIX
# ==============================================================================
KEYWORDS = [
    # 1. Market Microstructure & LOB
    "Limit Order Book Deep Learning",
    "Order Flow Imbalance",
    "Volume Synchronized Probability of Informed Trading",
    "Hawkes Process Market Making",
    "Queue Reactive Model",
    "Cumulative Volume Delta Crypto",
    "Adverse Selection Market Making",
    "Spoofing Detection Machine Learning",
    "Limit Order Placement Reinforcement Learning",
    "Easley-O'Hara Model",
    "Glosten-Milgrom Model",
    "Kyle's Lambda Market Depth",
    "Hasbrouck Information Share",
    "Tick Size Microstructure",

    # 2. Algorithmic Strategies & Alpha
    "Avellaneda-Stoikov Reinforcement Learning",
    "Statistical Arbitrage Cointegration",
    "Principal Component Analysis Statistical Arbitrage",
    "Funding Rate Arbitrage Perpetual Futures",
    "Liquidation Cascades Momentum",
    "Volatility Regime Detection Hidden Markov Model",
    "Hurst Exponent Mean Reversion",
    "Trend Following CTA Bullish",
    "Cross-Sectional Momentum Alpha",
    "Kalman Filter Pairs Trading",
    "Cross-Exchange Arbitrage Crypto",
    "Options Market Making Delta Neutral",
    "Triangular Arbitrage High Frequency",
    "Pairs Trading Copula",
    "Index Arbitrage Execution",
    "Basis Trading Cash and Carry Arbitrage",
    "Latency Arbitrage Microwave Networks",
    "Calendar Spread Arbitrage Roll Yield",

    # 3. Options Greeks, Derivatives, Commodities
    "Volatility Smile Skew",
    "Local Volatility Models Calibration",
    "SABR Model High Frequency",
    "Heston Model Machine Learning",
    "Dispersion Trading Index Options",
    "VIX Arbitrage Futures",
    "Gamma Squeeze Options Flow",
    "Gamma Scalping Dynamic Hedging",
    "Delta Hedging Transaction Costs",
    "Theta Decay Short Premium",
    "Vega Convexity Vanna Volga",
    "Iron Condor Machine Learning",
    "Variance Swaps Pricing",
    "Implied Volatility Surface Deep Learning",
    "Commodities Microstructure Crude Oil",
    "Crack Spreads Energy Arbitrage",
    "Gold Tick Data High Frequency",
    "CME Globex Order Book",
    "ICE Futures Microstructure",
    "Eurex Market Making",

    # 4. Alternative Data, On-Chain & DeFi
    "On-chain Analytics Whale Tracking",
    "Mempool Sniping MEV",
    "Flash Loan Arbitrage Smart Contract",
    "DEX Arbitrage",
    "Automated Market Maker Impermanent Loss",
    "Social Sentiment Twitter Bitcoin",
    "Sandwich Attacks Slippage",
    "JIT Liquidity Uniswap",
    "Cross-Chain Bridges Arbitrage",
    "TWAMM Time-Weighted Average Market Maker",
    "Zero-Knowledge Proofs Dark Pools",

    # 5. Advanced Machine Learning, AI & Econophysics
    "Transformer Limit Order Book",
    "Time Series Foundation Model",
    "Financial Sentiment Analysis FinBERT",
    "Proximal Policy Optimization Trading",
    "Soft Actor-Critic Inventory Management",
    "Generative Adversarial Networks Market Simulation",
    "Diffusion Models Financial Time Series",
    "Graph Neural Networks Cryptocurrency",
    "Spiking Neural Networks High Frequency",
    "Federated Learning Financial Data",
    "Quantum Machine Learning Finance",
    "Normalizing Flows Time Series",
    "Econophysics Agent-Based Market Models",
    "Power Laws Order Book",

    # 6. Latency Engineering & Infrastructure
    "High Frequency Trading DPDK",
    "Solarflare ef_vi OpenOnload",
    "FPGA Order Book Tick-to-Trade",
    "C++20 Lock-free Queue",
    "Kernel Bypass Networking",
    "SIMD AVX-512 Financial Mathematics",
    "Cache-line Alignment False Sharing",
    "RDMA Remote Direct Memory Access Trading",
    "PTP Precision Time Protocol Latency Jitter"
]

def fetch_semantic_scholar(keyword):
    """Scrapes Semantic Scholar (Alternative to arXiv) for top 3 papers."""
    query = urllib.parse.quote(keyword)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=20&fields=title,url,year"
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for item in data.get('data', []):
                title = item.get('title', '').replace('\n', ' ')
                paper_url = item.get('url', '')
                year = item.get('year', 'N/A')
                results.append((title, paper_url, year))
        time.sleep(1)
    except Exception as e:
        pass
    return results

def fetch_arxiv(keyword):
    """Scrapes arXiv for top 3 papers matching the keyword."""
    query = urllib.parse.quote(keyword)
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=20&sortBy=relevance&sortOrder=descending"
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
                pdf_url = ""
                for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                    if link.attrib.get('title') == 'pdf':
                        pdf_url = link.attrib.get('href')
                results.append((title, pdf_url))
        time.sleep(1) # Respect rate limit
    except Exception as e:
        pass
    return results

def fetch_github(keyword):
    """Scrapes GitHub REST API for top 3 repos matching keyword."""
    query = urllib.parse.quote(keyword)
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=20"
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HFT-Researcher-Bot'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for item in data.get('items', []):
                name = item.get('full_name')
                html_url = item.get('html_url')
                desc = item.get('description', '')
                stars = item.get('stargazers_count', 0)
                results.append((name, html_url, desc, stars))
        time.sleep(2) # GitHub rate limits are strict without auth
    except Exception as e:
        pass
    return results

def fetch_huggingface(keyword):
    """Scrapes Hugging Face for top 3 models."""
    query = urllib.parse.quote(keyword)
    url = f"https://huggingface.co/api/models?search={query}&limit=20"
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for item in data:
                model_id = item.get('id')
                downloads = item.get('downloads', 0)
                url = f"https://huggingface.co/{model_id}"
                results.append((model_id, url, downloads))
        time.sleep(0.5)
    except Exception as e:
        pass
    return results

def fetch_kaggle(keyword):
    """Generates direct Kaggle Dataset search URLs."""
    # Since Kaggle requires an authenticated JSON token to use its API natively, 
    # we generate a highly targeted search URL for the user to instantly view datasets.
    query = urllib.parse.quote(keyword)
    url = f"https://www.kaggle.com/search?q={query}+in%3Adatasets"
    return url

def generate_report():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'research')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'research_dump_{date_str}.md')

    print(f"Starting Autonomous Research Pipeline...")
    print(f"Scanning {len(KEYWORDS)} advanced topics across Semantic Scholar, arXiv, GitHub, Hugging Face, and Kaggle.")
    print(f"This will take approximately {len(KEYWORDS) * 5} seconds due to API rate limits.")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"# Master HFT Research Dump ({date_str})\n\n")
        f.write("> **Auto-generated by HFT Autonomous Researcher**\n\n")

        for idx, kw in enumerate(KEYWORDS):
            print(f"[{idx+1}/{len(KEYWORDS)}] Searching: {kw}")
            f.write(f"## Topic: `{kw}`\n\n")

            # 1. Semantic Scholar (Alternative to arXiv)
            scholar = fetch_semantic_scholar(kw)
            f.write("### Academic Papers (Semantic Scholar)\n")
            if scholar:
                for title, url, year in scholar:
                    f.write(f"* **[{title}]({url})** (Year: {year})\n")
            else:
                f.write("* *No highly relevant Semantic Scholar papers found.*\n")
            f.write("\n")

            # 2. arXiv
            papers = fetch_arxiv(kw)
            f.write("### Academic Papers (arXiv)\n")
            if papers:
                for title, url in papers:
                    f.write(f"* **[{title}]({url})**\n")
            else:
                f.write("* *No highly relevant arXiv papers found.*\n")
            f.write("\n")

            # 3. GitHub
            repos = fetch_github(kw)
            f.write("### Open Source Code (GitHub)\n")
            if repos:
                for name, url, desc, stars in repos:
                    clean_desc = (desc or 'No description').replace('\n', ' ')
                    f.write(f"* **[{name}]({url})** (Stars: {stars}): {clean_desc}\n")
            else:
                f.write("* *No highly relevant repositories found (or rate limited).*\n")
            f.write("\n")

            # 4. Hugging Face
            models = fetch_huggingface(kw)
            f.write("### ML Models (Hugging Face)\n")
            if models:
                for mid, url, dls in models:
                    f.write(f"* **[{mid}]({url})** (Downloads: {dls})\n")
            else:
                f.write("* *No highly relevant models found.*\n")
            f.write("\n")
            
            # 5. Kaggle
            kaggle_url = fetch_kaggle(kw)
            f.write("### Datasets (Kaggle)\n")
            f.write(f"* **[Search Kaggle Datasets for '{kw}']({kaggle_url})**\n")
            
            f.write("\n---\n\n")

    print(f"Research complete! Massive report saved to: {out_file}")

if __name__ == "__main__":
    generate_report()
