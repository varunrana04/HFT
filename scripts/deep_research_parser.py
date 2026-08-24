import os
import re
from collections import Counter
from datetime import datetime

DUMP_FILES = [
    "research/research_dump_20260823_231550.md",
    "research/targeted_dump_20260823_231232.md"
]

KEYWORDS = {
    "Deep Learning": [r"deep learning", r"cnn", r"lstm", r"transformer", r"neural network"],
    "Reinforcement Learning": [r"rl", r"ppo", r"ddpg", r"reinforcement learning", r"q-learning", r"mdp"],
    "Latency/Hardware": [r"dpdk", r"fpga", r"kernel bypass", r"ef_vi", r"onload", r"rdma", r"simd"],
    "Market Microstructure": [r"hawkes", r"avellaneda", r"glosten", r"vpin", r"ofi", r"queue reactive"],
    "Advanced Math": [r"diffusion", r"markov", r"stochastic", r"copula", r"kalman"]
}

def parse_dumps():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    github_repos = []
    arxiv_papers = []
    
    keyword_counts = {category: Counter() for category in KEYWORDS}

    # Regex patterns
    # Format: * **[Title](Link)** (Stars: 123): Description
    # Or just: * [Title](Link)
    github_pattern = re.compile(r'\*\s+\*?\*?\[(.*?)\]\((https?://github\.com/.*?)\)\*?\*?(?:\s+\(Stars:\s+(\d+)\))?(?:\s*:\s*(.*))?', re.IGNORECASE)
    arxiv_pattern = re.compile(r'\*\s+\*?\*?\[(.*?)\]\((https?://arxiv\.org/.*?)\)\*?\*?(?:\s*:\s*(.*))?', re.IGNORECASE)

    total_lines = 0

    for dump in DUMP_FILES:
        filepath = os.path.join(base_dir, dump)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                
                # Check for GitHub
                gh_match = github_pattern.search(line)
                if gh_match:
                    title, link, stars_str, desc = gh_match.groups()
                    stars = int(stars_str) if stars_str else 0
                    desc = desc if desc else ""
                    github_repos.append({"title": title, "link": link, "stars": stars, "desc": desc.strip()})
                    _count_keywords(title + " " + desc, keyword_counts)
                    continue
                
                # Check for arXiv
                arx_match = arxiv_pattern.search(line)
                if arx_match:
                    title, link, desc = arx_match.groups()
                    desc = desc if desc else ""
                    arxiv_papers.append({"title": title, "link": link, "desc": desc.strip()})
                    _count_keywords(title + " " + desc, keyword_counts)

    # Deduplicate based on link
    github_repos = list({v['link']:v for v in github_repos}.values())
    arxiv_papers = list({v['link']:v for v in arxiv_papers}.values())
    
    # Sort GitHub by stars descending
    github_repos.sort(key=lambda x: x['stars'], reverse=True)

    return github_repos, arxiv_papers, keyword_counts, total_lines

def _count_keywords(text, keyword_counts):
    text_lower = text.lower()
    for category, patterns in KEYWORDS.items():
        for pattern in patterns:
            if re.search(r'\b' + pattern + r'\b', text_lower):
                keyword_counts[category][pattern] += 1

def generate_report(github_repos, arxiv_papers, keyword_counts, total_lines):
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research"))
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"parsed_synthesis_{date_str}.md")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("# Algorithmic Research Synthesis\n\n")
        f.write(f"> Processed {total_lines} lines of raw research data.\n")
        f.write(f"> Extracted {len(github_repos)} unique GitHub Repos and {len(arxiv_papers)} unique arXiv Papers.\n\n")

        f.write("## 1. Top 25 Highest-Signal Repositories (By Stars)\n")
        for repo in github_repos[:25]:
            f.write(f"* **[{repo['title']}]({repo['link']})** (Stars: {repo['stars']}): {repo['desc'][:150]}...\n")
        f.write("\n")
        
        f.write("## 2. Top 25 Highly Relevant Academic Papers\n")
        for paper in arxiv_papers[:25]:
            f.write(f"* **[{paper['title']}]({paper['link']})**\n")
        f.write("\n")

        f.write("## 3. Algorithmic Trend Frequencies\n")
        for category, counts in keyword_counts.items():
            f.write(f"### {category}\n")
            if not counts:
                f.write("* No specific matches.\n")
            for word, count in counts.most_common():
                f.write(f"* `{word}`: {count} mentions\n")
            f.write("\n")

    print(f"Data parsed successfully. Synthesis saved to {out_file}")

if __name__ == "__main__":
    repos, papers, counts, lines = parse_dumps()
    generate_report(repos, papers, counts, lines)
