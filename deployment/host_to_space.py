"""Hosting script: create/update a Docker-based Hugging Face Space
with the deployment files (Dockerfile, app.py, requirements.txt).
"""

import os
from huggingface_hub import HfApi

HF_USERNAME = "cktai"
SPACE_NAME = "tourism-wellness-predictor"
SPACE_ID = f"{HF_USERNAME}/{SPACE_NAME}"

# Path where deployment files live
PROJECT_ROOT = "/content/drive/MyDrive/advanced-ml-mlops-tourism"
DEPLOY_DIR = os.path.join(PROJECT_ROOT, "deployment")

def push_to_space(hf_token: str):
    api = HfApi(token=hf_token)

    # Create or reuse a Docker Space
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",   # important: tells HF this is a Docker Space
        exist_ok=True
    )

    # Files to upload
    files_to_upload = [
        ("Dockerfile", "Dockerfile"),
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
    ]

    for local_name, repo_name in files_to_upload:
        local_path = os.path.join(DEPLOY_DIR, local_name)
        print(f"Uploading {local_path} to Space as {repo_name}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_name,
            repo_id=SPACE_ID,
            repo_type="space",
        )

    print("All deployment files pushed to Space:", SPACE_ID)

if __name__ == "__main__":
    import getpass
    token = getpass.getpass("Enter your Hugging Face token: ")
    push_to_space(token)
