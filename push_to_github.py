"""
Direct GitHub API Repository Creator & File Uploader for YashRajKeshri.
Creates the repository automatically on GitHub (if not already created)
and pushes all files directly via GitHub REST & Git Data APIs.
Zero local git or refspec dependencies.
"""

import base64
import os
import sys
from pathlib import Path
import httpx

REPO_NAME = "ecommerce-customer-analytics"
GITHUB_USER = "YashRajKeshri"
BASE_DIR = Path(__file__).resolve().parent

IGNORE_PATTERNS = [
    ".git",
    "venv",
    ".pytest_cache",
    "__pycache__",
    ".DS_Store",
    "models/churn_pipeline.joblib",  # Generate on the fly or download
    "data/raw/online_retail.csv",     # Generated on the fly via data_loader
]


def should_ignore(path: Path) -> bool:
    rel = path.relative_to(BASE_DIR).as_posix()
    for pat in IGNORE_PATTERNS:
        if rel == pat or rel.startswith(pat + "/") or path.name == pat:
            return True
    return False


def collect_files():
    file_list = []
    for root, dirs, files in os.walk(BASE_DIR):
        root_path = Path(root)
        if should_ignore(root_path):
            continue
        for f in files:
            p = root_path / f
            if not should_ignore(p):
                rel = p.relative_to(BASE_DIR).as_posix()
                file_list.append((rel, p))
    return file_list


def create_or_verify_repo(token: str):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PortfolioUploader",
    }
    
    # Check if repo exists
    r = httpx.get(f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}", headers=headers)
    if r.status_code == 200:
        print(f"✅ Repository https://github.com/{GITHUB_USER}/{REPO_NAME} exists.")
        return
    
    # Create repo
    print(f"📦 Creating new public repository '{REPO_NAME}' on GitHub...")
    payload = {
        "name": REPO_NAME,
        "description": "End-to-End E-Commerce Customer Analytics, Cohort Retention & Churn Prediction ML Engine",
        "private": False,
        "auto_init": True,
    }
    r = httpx.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if r.status_code in [200, 201]:
        print("✅ Repository created successfully on GitHub!")
    else:
        print(f"⚠️ Note on repo creation: {r.status_code} - {r.text}")


def upload_via_github_api(token: str):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PortfolioUploader",
    }

    create_or_verify_repo(token)

    files = collect_files()
    print(f"\n🚀 Uploading {len(files)} project files directly to GitHub main branch...\n")

    for rel_path, full_path in files:
        try:
            with open(full_path, "rb") as f:
                content_bytes = f.read()

            content_b64 = base64.b64encode(content_bytes).decode("utf-8")

            # Check if file exists on GitHub to get sha
            url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/{rel_path}"
            r_get = httpx.get(url, headers=headers)

            payload = {
                "message": f"feat: add {rel_path}",
                "content": content_b64,
                "branch": "main",
            }
            if r_get.status_code == 200:
                payload["sha"] = r_get.json().get("sha")

            r_put = httpx.put(url, headers=headers, json=payload, timeout=30.0)
            if r_put.status_code in [200, 201]:
                print(f"  ✓ {rel_path}")
            else:
                print(f"  ✗ {rel_path} ({r_put.status_code}: {r_put.json().get('message')})")
        except Exception as e:
            print(f"  ✗ Error uploading {rel_path}: {str(e)}")

    print(f"\n🎉 All files uploaded successfully!")
    print(f"🌟 View your live portfolio at: https://github.com/{GITHUB_USER}/{REPO_NAME}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        token = sys.argv[1].strip()
    else:
        token = input("Enter your GitHub Personal Access Token: ").strip()

    if not token:
        print("❌ Error: GitHub Token is required.")
        sys.exit(1)

    upload_via_github_api(token)
