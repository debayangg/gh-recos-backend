import os
import requests
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# GitHub API setup
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": "Bearer "+os.getenv('GITHUB_TOKEN')  # Replace with your GitHub token
}
BASE_URL = "https://api.github.com"

# Define model for embeddings
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Rate limit handling
def handle_rate_limit(response):
    if response.status_code == 403:  # Rate limit exceeded
        retry_after = response.headers.get("retry-after")
        if retry_after:
            wait_time = int(retry_after)
        elif response.headers.get("x-ratelimit-remaining") == "0":
            reset_time = int(response.headers.get("x-ratelimit-reset"))
            wait_time = reset_time - int(time.time())
        else:
            wait_time = 60  # Default wait time of 1 minute
        if wait_time < 0:
            wait_time = 60
        print(f"Rate limit exceeded. Waiting for {wait_time} seconds...")
        time.sleep(wait_time + 1)
    elif response.status_code in {500, 502, 503, 504}:  # Retry for server errors
        print("Server error. Retrying after 10 seconds...")
        time.sleep(10)
    else:
        print(f"Unexpected error: {response.status_code} - {response.json()}")
        return False
    return True

# Fetch recent commits
def fetch_recent_commits(owner, repo, max_commits=10):
    url = f"{BASE_URL}/repos/{owner}/{repo}/commits"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return [commit["commit"]["message"] for commit in response.json()[:max_commits]]
    handle_rate_limit(response)
    return []

# Fetch markdown file (preferably README.md)
def fetch_markdown(owner, repo):
    url = f"{BASE_URL}/repos/{owner}/{repo}/contents/"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        files = response.json()
        readme = next((file["download_url"] for file in files if file["name"].lower() == "readme.md"), None)
        if readme:
            content_response = requests.get(readme)
            return content_response.text if content_response.status_code == 200 else ""
    handle_rate_limit(response)
    return ""

# Generate embeddings
def generate_embedding(text):
    return MODEL.encode(text)

# Fetch repositories for the user
def fetch_user_repositories(username, max_repos=10):
    url = f"{BASE_URL}/users/{username}/repos"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()[:max_repos]
    handle_rate_limit(response)
    return []

# Compute cosine similarity
def rank_repositories(embedding, db_name, language):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Fetch all embeddings for the given language
    cursor.execute("SELECT repo_name, repo_url, embedding FROM project_embeddings WHERE main_language = ?", (language,))
    projects = cursor.fetchall()
    conn.close()

    # Calculate cosine similarity
    scores = []
    for name, url, embedding_blob in projects:
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
        scores.append((name, url, similarity))

    # Sort by similarity
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores

# Main execution
def recommend_repositories(username, tags, language, db_name="project_embeddings.db"):
    repositories = fetch_user_repositories(username)
    combined_texts = []

    for repo in repositories:
        owner, repo_name = repo["owner"]["login"], repo["name"]

        # Fetch recent commits
        commits = fetch_recent_commits(owner, repo_name)

        # Fetch markdown content
        markdown = fetch_markdown(owner, repo_name)

        # Combine commits, markdown, and tags
        combined_text = "\n".join(commits + [markdown] + tags)
        combined_texts.append((repo_name, combined_text))

    # Generate embeddings
    embeddings = [(repo_name, generate_embedding(text)) for repo_name, text in combined_texts]

    # Rank repositories by similarity
    scores = []
    for repo_name, embedding in embeddings:
        ranked = rank_repositories(embedding, db_name, language)
        scores.extend(ranked)

    return scores

class ReqParams(BaseModel):
    username: str
    tag: str
    language: str

app = FastAPI()

tags_by_domain = {
    "NLP": ["nlp", "text-analysis", "language-models", "transformers", "chatbot"],
    "ML": ["machine-learning", "deep-learning", "neural-networks", "ai", "data-science"],
    "Tools": ["cli", "automation", "devops", "testing", "debugging"],
    "Web3": ["blockchain", "smart-contracts", "ethereum", "solidity", "nfts"],
    "WebDev": ["frontend", "backend", "react", "django", "flask"]
}

@app.post('/recommend')
def recommend(req : ReqParams):
  recommendations = recommend_repositories(req.username, tags_by_domain[req.tag], req.language)
  result = set()
  for name, url, _ in recommendations:
    result.add(url)
  return result
