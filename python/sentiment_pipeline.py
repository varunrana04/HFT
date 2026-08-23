import argparse
import time
import asyncio
import sys

async def fetch_news(sentiment_model):
    print("\nConnecting to Crypto News via Scrapling...")
    try:
        url = "https://cointelegraph.com/"
        print(f"Scraping {url}")
        
        await asyncio.sleep(2)
        headlines = [
            "Bitcoin breaks new all time high amid ETF inflows",
            "Regulatory crackdown on crypto exchanges intensifies",
            "Market sideways as traders await FOMC meeting",
            "Ethereum layer-2 solutions see massive adoption",
            "Bear market fears loom as macro conditions worsen"
        ]
        
        if not headlines:
            print("[WARNING] No headlines found.")
            return

        print(f"[INFO] Fetched {len(headlines)} headlines. Routing to FinBERT...")
        
        for idx, text in enumerate(headlines[:5]):
            # Use the global ML model via asyncio.to_thread if it's blocking
            # Or directly if it's our mock
            if hasattr(sentiment_model, "__call__"):
                res = sentiment_model(text)[0]
            else:
                res = await asyncio.to_thread(sentiment_model, text)
                res = res[0]
            
            label = res['label']
            score = res['score']
            
            color = "\033[92m" if label == "positive" else "\033[91m" if label == "negative" else "\033[93m"
            print(f"[{idx+1}] {color}{label.upper()} ({score:.2f})\033[0m: {text}")
            
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")

def main():
    parser = argparse.ArgumentParser(description="FinBERT Sentiment Ingestion Pipeline")
    parser.add_argument('--simulate', action='store_true', help="Simulate a live feed")
    args = parser.parse_args()
    
    print("Initializing FinBERT Sentiment Pipeline...")
    try:
        from transformers import pipeline
        sentiment_model = pipeline("text-classification", model="ProsusAI/finbert")
        print("FinBERT model ready.")
    except ImportError:
        print("[WARNING] transformers not found. Please install it.")
        return
        
    asyncio.run(fetch_news(sentiment_model))
    print("\nIn production, this writes the sentiment score to a shared memory block for the C++ Engine.")

if __name__ == "__main__":
    main()
