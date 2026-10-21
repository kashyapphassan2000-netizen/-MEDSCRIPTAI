"""Step 6: publish an adapter or a dataset folder to the Hugging Face Hub (private by default)."""
import argparse
from huggingface_hub import HfApi                  # official Hub client; reads HF_TOKEN from the environment

p = argparse.ArgumentParser()
p.add_argument("--folder", required=True, help="local folder, e.g. outputs/adapters/sft-qa or data/sft")
p.add_argument("--repo", required=True, help="YOUR_HF_USERNAME/Llama-3.1-8B-MedScriptAI-qa-lora")
p.add_argument("--type", default="model", choices=["model", "dataset"])
p.add_argument("--public", action="store_true", help="make the repo public (default private)")
a = p.parse_args()

api = HfApi()
api.create_repo(a.repo, repo_type=a.type, private=not a.public, exist_ok=True)   # no error if it exists
api.upload_folder(folder_path=a.folder, repo_id=a.repo, repo_type=a.type,
                  ignore_patterns=["checkpoint-*", "*.pt", "optimizer*"])          # skip training leftovers
print(f"https://huggingface.co/{'datasets/' if a.type == 'dataset' else ''}{a.repo}")
