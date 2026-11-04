"""Download the pinned desklib checkpoint into work/hf/."""
from huggingface_hub import snapshot_download

from common import HF_DIR, REPO_ID, REVISION

if __name__ == "__main__":
    path = snapshot_download(REPO_ID, revision=REVISION, local_dir=str(HF_DIR))
    print("downloaded to", path)
