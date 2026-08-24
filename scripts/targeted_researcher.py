import os
import re
import json
import time
import urllib.request
import urllib.error
from datetime import datetime

# ==============================================================================
# 🎯 TARGET MATRIX
# ==============================================================================

GITHUB_ORGS_AND_USERS = [
    "HKUDS",
    "AI4Finance-Foundation",
    "microsoft", # Will be too large, but we can filter by 'qlib' or just get top 10 repos by stars if needed. Let's restrict to top 15 by stars to avoid massive dumps.
    "Open-Finance-Lab",
    "nautechsystems",
    "aruvins"
]

AWESOME_LISTS = [
    "paperswithbacktest/awesome-systematic-trading",
    "wilsonfreitas/awesome-quant",
    "PyPatel/Quant-Finance-Resources",
    "Open-Finance-Lab/Awesome-MFFMs"
]

SPECIFIC_REPOS = [
    "shiyu-coder/Kronos",
    "vincent05r/FinCast-fts",
    "skfolio/skfolio",
    "d4vinci/Scrapling"
]

CUSTOM_URLS = [
    ("4 Quant Finance Projects (Google Doc)", "https://docs.google.com/document/d/1ypO2ORyXqGNcnmMxIMgOIj88b4IvdM6nrD3kXaMahaQ/mobilebasic"),
    ("London Strategic Edge API", "https://londonstrategicedge.com/api-documentation/")
]

def fetch_github_org_repos(org_name):
    """Fetches top repositories for a given GitHub user or organization."""
    url = f"https://api.github.com/users/{org_name}/repos?sort=updated&per_page=15"
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HFT-Targeted-Researcher'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            for item in data:
                name = item.get('name')
                html_url = item.get('html_url')
                desc = item.get('description', '')
                stars = item.get('stargazers_count', 0)
                results.append((name, html_url, desc, stars))
        time.sleep(1)
    except urllib.error.HTTPError as e:
        print(f"Failed to fetch {org_name}: {e}")
    except Exception as e:
        pass
    return sorted(results, key=lambda x: x[3], reverse=True) # Sort by stars

def fetch_specific_repo(repo_full_name):
    """Fetches details for a specific repository."""
    url = f"https://api.github.com/repos/{repo_full_name}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'HFT-Targeted-Researcher'})
        with urllib.request.urlopen(req, timeout=10) as response:
            item = json.loads(response.read().decode())
            name = item.get('full_name')
            html_url = item.get('html_url')
            desc = item.get('description', '')
            stars = item.get('stargazers_count', 0)
            return (name, html_url, desc, stars)
    except Exception as e:
        return None

def fetch_awesome_list_links(repo_full_name):
    """Downloads the README of an awesome list and extracts all Markdown links."""
    branch = "main"
    url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/README.md"
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            readme_text = response.read().decode('utf-8')
            # Extract markdown links: [Title](URL)
            pattern = re.compile(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)')
            matches = pattern.findall(readme_text)
            for title, link in matches:
                if 'github.com' in link or 'arxiv.org' in link or 'papers' in link.lower():
                    results.append((title.strip(), link.strip()))
        time.sleep(1)
    except urllib.error.HTTPError:
        # Fallback to master branch
        try:
            url_master = f"https://raw.githubusercontent.com/{repo_full_name}/master/README.md"
            req = urllib.request.Request(url_master, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                readme_text = response.read().decode('utf-8')
                pattern = re.compile(r'\[([^\]]+)\]\((http[s]?://[^\)]+)\)')
                matches = pattern.findall(readme_text)
                for title, link in matches:
                    if 'github.com' in link or 'arxiv.org' in link or 'papers' in link.lower():
                        results.append((title.strip(), link.strip()))
        except Exception:
            pass
    return list(dict.fromkeys(results)) # Remove duplicates

def fetch_custom_url_links(url):
    """Scrapes a URL for embedded links."""
    results = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            # Very basic extraction of hrefs
            pattern = re.compile(r'href=[\'"]?(http[s]?://[^\'" >]+)')
            matches = pattern.findall(html)
            for link in set(matches):
                if link != url:
                    results.append(link)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
    return results

def generate_targeted_report():
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'research')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'targeted_dump_{date_str}.md')

    print("Starting Targeted Elite Alpha Acquisition...")
    print(f"Targeting {len(GITHUB_ORGS_AND_USERS)} Orgs, {len(AWESOME_LISTS)} Awesome Lists, and specific models.")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"# Targeted Alpha Acquisition Dump ({date_str})\n\n")
        f.write("> **Sniper Extraction Protocol for High-Value Quant Targets**\n\n")

        # 1. SPECIFIC FOUNDATION MODELS
        print("Extracting specific foundation models...")
        f.write("## 1. High-Value Foundation Models & Engines\n\n")
        for repo in SPECIFIC_REPOS:
            data = fetch_specific_repo(repo)
            if data:
                name, url, desc, stars = data
                clean_desc = (desc or '').replace('\n', ' ')
                f.write(f"* **[{name}]({url})** (Stars: {stars}): {clean_desc}\n")
        f.write("\n---\n\n")

        # 2. ELITE LABS AND ORGANIZATIONS
        print("Extracting Org repositories...")
        f.write("## 2. Elite Quant/ML Labs (Top Repositories)\n\n")
        for org in GITHUB_ORGS_AND_USERS:
            f.write(f"### Organization: `{org}`\n")
            repos = fetch_github_org_repos(org)
            if repos:
                for name, url, desc, stars in repos:
                    # Filter out massive junk from Microsoft, only keep high signal
                    if org == 'microsoft' and 'qlib' not in name.lower() and stars < 5000:
                        continue
                    clean_desc = (desc or '').replace('\n', ' ')
                    f.write(f"* **[{name}]({url})** (Stars: {stars}): {clean_desc}\n")
            else:
                f.write("* *No repositories found or rate limited.*\n")
            f.write("\n")
        f.write("---\n\n")

        # 3. CURATED AWESOME LISTS (DEEP LINK EXTRACTION)
        print("Extracting Awesome List hyperlinks...")
        f.write("## 3. Curated Resource Lists (Extracted Links)\n\n")
        for lst in AWESOME_LISTS:
            f.write(f"### Repository: `{lst}`\n")
            f.write(f"*Source: https://github.com/{lst}*\n\n")
            links = fetch_awesome_list_links(lst)
            if links:
                f.write(f"Found {len(links)} external links (GitHub/arXiv focus):\n")
                # To prevent the file from becoming 10,000 lines, limit to top 150 links per list
                for title, link in links[:150]:
                    f.write(f"* [{title}]({link})\n")
                if len(links) > 150:
                    f.write(f"* ...and {len(links) - 150} more. (See raw repo for full list).\n")
            else:
                f.write("* *Could not parse README or no links found.*\n")
            f.write("\n")
        f.write("---\n\n")

        # 4. CUSTOM URLS
        print("Scraping custom web targets...")
        f.write("## 4. Custom Web Targets\n\n")
        for name, url in CUSTOM_URLS:
            f.write(f"### {name}\n")
            f.write(f"*Source: {url}*\n\n")
            extracted = fetch_custom_url_links(url)
            if extracted:
                f.write("Extracted Links:\n")
                for link in extracted[:50]: # Limit to 50
                    f.write(f"* {link}\n")
            else:
                f.write("* *Could not scrape embedded links.*\n")
            f.write("\n")

    print(f"Targeted acquisition complete! Dump saved to: {out_file}")

if __name__ == "__main__":
    generate_targeted_report()
