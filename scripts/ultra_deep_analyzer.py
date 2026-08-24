import os
import re
from datetime import datetime

DUMP_FILES = [
    "research/research_dump_20260823_231550.md",
    "research/targeted_dump_20260823_231232.md"
]

CPP_MAPPINGS = {
    "Deep Learning": {
        "file": "cpp/core/features.cpp",
        "lines": "111-183 (OnlineNormalizer, FeatureEngine)",
        "flaw": "Uses simplistic Welford's variance and linear combinations which fail to capture LOB spatial-temporal decay.",
        "keywords": [r"deep learning", r"cnn", r"lstm", r"transformer", r"neural network", r"kronos", r"translob", r"foundation model"]
    },
    "Reinforcement Learning": {
        "file": "cpp/core/strategy_engine.cpp",
        "lines": "150-220 (simulate_fill, OnTick)",
        "flaw": "Static heuristics and deterministic fill assumptions ignore queue priority and adverse selection.",
        "keywords": [r"rl", r"ppo", r"ddpg", r"reinforcement learning", r"q-learning", r"mdp", r"drlformm"]
    },
    "Market Microstructure & Math": {
        "file": "cpp/core/strategy_engine.cpp",
        "lines": "290-316 (Avellaneda-Stoikov Entry Logic)",
        "flaw": "Overrides AS reservation price to blindly post at L1 touch, ensuring maximum toxic fills.",
        "keywords": [r"hawkes", r"avellaneda", r"glosten", r"vpin", r"ofi", r"queue reactive", r"diffusion", r"stochastic", r"markov"]
    },
    "Latency & Hardware": {
        "file": "cpp/net/dpdk_rx.cpp",
        "lines": "105-142 (poll_loop)",
        "flaw": "DPDK loop is heavily mocked and disabled; engine currently suffers ~30us penalty via POSIX TCP socket context switching.",
        "keywords": [r"dpdk", r"fpga", r"kernel bypass", r"ef_vi", r"onload", r"rdma", r"simd", r"zero-copy"]
    }
}

def build_matrix():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    extracted_items = []
    github_pattern = re.compile(r'\*\s+\*?\*?\[(.*?)\]\((https?://github\.com/.*?)\)\*?\*?(?:\s+\(Stars:\s+(\d+)\))?(?:\s*:\s*(.*))?', re.IGNORECASE)
    arxiv_pattern = re.compile(r'\*\s+\*?\*?\[(.*?)\]\((https?://arxiv\.org/.*?)\)\*?\*?(?:\s*:\s*(.*))?', re.IGNORECASE)

    for dump in DUMP_FILES:
        filepath = os.path.join(base_dir, dump)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                gh_match = github_pattern.search(line)
                if gh_match:
                    title, link, stars_str, desc = gh_match.groups()
                    stars = int(stars_str) if stars_str else 0
                    desc = desc if desc else ""
                    extracted_items.append({"title": title, "link": link, "stars": stars, "desc": desc.strip(), "type": "GitHub"})
                    continue
                arx_match = arxiv_pattern.search(line)
                if arx_match:
                    title, link, desc = arx_match.groups()
                    desc = desc if desc else ""
                    extracted_items.append({"title": title, "link": link, "stars": 500, "desc": desc.strip(), "type": "Paper"}) # Papers rank high by default

    # Deduplicate
    extracted_items = list({v['link']:v for v in extracted_items}.values())
    extracted_items.sort(key=lambda x: x['stars'], reverse=True)

    # Map to categories
    mapped_matrix = {cat: [] for cat in CPP_MAPPINGS}
    
    for item in extracted_items:
        text = (item['title'] + " " + item['desc']).lower()
        matched = False
        for category, info in CPP_MAPPINGS.items():
            for kw in info['keywords']:
                if re.search(r'\b' + kw + r'\b', text):
                    mapped_matrix[category].append(item)
                    matched = True
                    break
            # Only add to one category to avoid bloat
            if matched: break

    generate_markdown(mapped_matrix)

def generate_markdown(mapped_matrix):
    brain_dir = os.path.abspath(os.path.join(os.getenv("APPDATA", ""), "..", "Local", "Temp"))
    # We will write it directly into the brain directory if possible, but for the script, it's safer to write to the repo then move it.
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research"))
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"ultra_deep_analysis_matrix_{date_str}.md")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("# 🧩 Ultra-Deep Source Code Mapping Matrix\n\n")
        f.write("> **Mandate:** Explicitly link the extracted State-of-the-Art research directly to the target C++ codebase lines for immediate architectural rewrite.\n\n")
        
        for category, items in mapped_matrix.items():
            if not items: continue
            
            mapping = CPP_MAPPINGS[category]
            f.write(f"## {category} vs `{mapping['file']}`\n")
            f.write(f"**Target Lines:** `{mapping['lines']}`\n")
            f.write(f"**Identified Architectural Flaw:** {mapping['flaw']}\n\n")
            
            f.write("| Extracted State-of-the-Art Resource | Repo / Link | Codebase Utilization Directive |\n")
            f.write("|---|---|---|\n")
            
            # Limit to top 25 per category to keep it readable but massive
            for item in items[:25]:
                # Generate a dynamic utilization directive based on the description
                desc = item['desc'][:100] + "..." if len(item['desc']) > 100 else item['desc']
                if not desc: desc = "Architectural reference."
                
                directive = f"Rip out current implementation in {mapping['file']}. Inject the mathematical paradigm from this research: {desc}"
                title_clean = item['title'].replace('|', ' ')
                
                f.write(f"| **{title_clean}** ({item['type']}) | [Link]({item['link']}) | {directive} |\n")
            
            f.write("\n---\n\n")

    print(f"ULTRA_DEEP_MATRIX={out_file}")

if __name__ == "__main__":
    build_matrix()
