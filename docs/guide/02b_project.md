# PART 2B - Project folder + tools on the GPU computer (from an empty disk)

## BEFORE YOU START: what you are building in this part
You now have an empty GPU computer. In this part you create the project folder structure by hand, learn the ONE procedure you will use to create every file in this manual, store your secrets safely, install the tools (uv, tmux, the AWS CLI, FakeCloud), and create three separate Python environments.
Why three Python environments? The libraries fight each other: Unsloth needs transformers 5.17 or older and trl 1.15 or older; distilabel (last release January 2025) needs datasets older than 4 and openai older than 2; the evaluation tools want the newest transformers. An environment (venv) is a private folder of packages, so each tool gets the versions it needs.
| Environment | Used for | Parts |
| .venv-train | data scripts, Unsloth SFT and DPO training, publishing to the Hub | 4, 5, 6A |
| .venv-synth | distilabel synthetic data and preference pairs | 5, 6A |
| .venv-eval | evaluation metrics, the judge, the API and its tests | 5, 6A, 7 |
Result at the end of this part: this folder tree exists, three environments work, the GPU test prints True, and FakeCloud answers on port 4566.
```
/home/ubuntu/medscriptai/
├── scripts/        all Python scripts (data, training, synthetic data, evaluation)
├── serve/          scripts that start the vLLM model servers
├── api/            the FastAPI web app and its Dockerfile
├── deploy/         Docker Compose and cloud deployment scripts
├── tests/          a fake model server and API tests
├── requirements/   the package lists for the three environments
├── data/           datasets (created by the scripts)
└── outputs/        adapters, predictions, reports (created by the scripts)
```

## SECTION 2B.1 - Create the project folders
WHERE: VS Code Terminal (bottom-left shows SSH: medscriptai)
GOAL: the empty folder tree above.
Step 1. Go to your home folder:
```
cd ~
```
Step 2. Create the project folder and all sub-folders in one command:
```
mkdir -p medscriptai/scripts medscriptai/serve medscriptai/api medscriptai/deploy medscriptai/tests medscriptai/requirements medscriptai/data medscriptai/outputs
```
Step 3. Go into the project folder:
```
cd ~/medscriptai
```
Step 4. List what you created:
```
ls
```
You should see: api  data  deploy  outputs  requirements  scripts  serve  tests
| Piece | What it does |
| cd ~ | go to your home folder /home/ubuntu (~ is a shortcut for it) |
| mkdir -p | make folders; -p also creates parent folders and does not complain if they exist |
CHECKPOINT: ls shows the eight folders.

## SECTION 2B.2 - Open the project folder in VS Code
WHERE: VS Code (connected, SSH: medscriptai)
GOAL: the Explorer panel on the left shows your project.
Step 1. In the top menu click File > Open Folder...
Step 2. In the box type /home/ubuntu/medscriptai/ and press Enter (or click OK).
Step 3. If asked "Do you trust the authors of the files in this folder?", click Yes, I trust the authors.
Step 4. Press Ctrl+Shift+E to show the Explorer panel.
Step 5. Open a terminal again: Terminal > New Terminal.
You should see: the Explorer shows MEDSCRIPTAI with the eight folders, and the new terminal starts in ~/medscriptai.
CHECKPOINT: pwd in the terminal prints /home/ubuntu/medscriptai

## SECTION 2B.3 - THE procedure to create any file (practise it once)
WHERE: VS Code Explorer panel (left)
GOAL: you can create a file in the right folder with the exact content. Every file in this manual is made this way.
Step 1. In the Explorer, right-click the folder where the file must go (for this practice: the top folder MEDSCRIPTAI).
Step 2. Click New File...
Step 3. Type the file name (for this practice: README.md) and press Enter. The empty file opens in the editor.
Step 4. Copy the content from the grey box in this manual (select it, Ctrl+C).
Step 5. Click inside the empty editor and press Ctrl+V.
Step 6. Press Ctrl+S to save (the dot on the file's tab disappears).
Practice content for README.md:
```
# MedScriptAI
Medical post-training project: Llama-3.1-8B-Instruct + QLoRA SFT + DPO, served with vLLM multi-LoRA behind FastAPI.
Built with Llama. Educational use only; not medical advice.
```
Step 7. Check the file from the terminal:
```
cat README.md
```
You should see: the three lines you pasted.
Watch out: (1) the file name must match EXACTLY, including the extension (.py, .sh, .txt, .yml); (2) right-click the CORRECT folder, otherwise the file lands in the wrong place; (3) Google Docs may turn straight quotes into curly quotes if you edit the document: always copy from the grey box, and if Python later reports "invalid character", re-paste the file.
CHECKPOINT: cat README.md prints the text.

## SECTION 2B.4 - Store secrets and the model cache location
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: your Hugging Face token and cache settings are loaded in every new terminal, and never written into code.
Step 1. Create the fast cache folder on the big ephemeral disk:
```
sudo mkdir -p /ephemeral/hf-cache && sudo chown ubuntu:ubuntu /ephemeral/hf-cache
```
Step 2. Tell Hugging Face to use it (added to ~/.bashrc, which runs in every new terminal):
```
echo 'export HF_HOME=/ephemeral/hf-cache' >> ~/.bashrc
```
Step 3. Add your token (replace hf_YOUR_TOKEN with the token from secrets.txt):
```
echo 'export HF_TOKEN=hf_YOUR_TOKEN' >> ~/.bashrc
```
Step 4. Load the new settings into this terminal:
```
source ~/.bashrc
```
Step 5. Check both:
```
echo $HF_HOME && echo ${HF_TOKEN:0:6}
```
You should see: /ephemeral/hf-cache and hf_xxx (only the first 6 characters, so the token is not shown in full).
| Piece | What it does |
| sudo | run as administrator (needed to write in /ephemeral) |
| chown ubuntu:ubuntu | make the user ubuntu the owner, so scripts can write there |
| echo '...' >> ~/.bashrc | append a line to the file that runs at every terminal start |
| source ~/.bashrc | run it now in the current terminal |
| ${HF_TOKEN:0:6} | the first 6 characters of the variable |
Watch out: if /ephemeral does not exist on your flavor, use /home/ubuntu/hf-cache instead in Steps 1-2 (without sudo).
CHECKPOINT: both values print.

## SECTION 2B.5 - Install system tools and learn tmux (jobs that survive disconnects)
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: tmux, git, jq and unzip installed; you can start, leave and re-enter a tmux session.
Step 1. Install the tools:
```
sudo apt-get update && sudo apt-get install -y tmux git unzip jq
```
Step 2. Start a practice tmux session:
```
tmux new -s practice
```
Step 3. In the tmux session type a long-running command:
```
for i in $(seq 1 300); do echo "tick $i"; sleep 1; done
```
Step 4. Detach (leave it running): press Ctrl+B, release both keys, then press D.
You should see: "[detached (from session practice)]" and your normal prompt.
Step 5. List sessions:
```
tmux ls
```
You should see: practice: 1 windows ...
Step 6. Re-attach:
```
tmux attach -t practice
```
You should see: the ticks still counting.
Step 7. Stop the loop with Ctrl+C, then type exit and press Enter to close the session.
| Key or command | Meaning |
| tmux new -s NAME | start a new session called NAME |
| Ctrl+B then D | detach: leave the session running in the background |
| tmux ls | list running sessions |
| tmux attach -t NAME | go back into a session |
| exit (inside tmux) | close the session |
Why it matters: training runs for 10-60 minutes. If your laptop sleeps or Wi-Fi drops, a normal terminal kills the job; a tmux session keeps it running on the GPU computer.
CHECKPOINT: you detached and re-attached once.

## SECTION 2B.6 - Install uv (Python package manager)
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: the uv command works.
Step 1. Install:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Step 2. Load it into this terminal:
```
source ~/.bashrc
```
Step 3. Check:
```
uv --version
```
You should see: uv 0.x.x (any recent version, 0.8 or newer)
Why uv: it creates environments and installs packages 10-100x faster than pip, and it can pick the right PyTorch build for your GPU driver.
CHECKPOINT: uv --version prints a version.
If not: "uv: command not found" = close the terminal (trash-can icon), open Terminal > New Terminal, try again.

## SECTION 2B.7 - Create the three requirements files
WHERE: VS Code Explorer, folder requirements
GOAL: three text files listing the exact package versions for each environment.
FILE 1: requirements/train.txt. Create it with the procedure of Section 2B.3:
Step 1. Right-click the folder requirements.
Step 2. Click New File...
Step 3. Type train.txt and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include requirements/train.txt
FILE 2: requirements/synth.txt
Step 6. Right-click the folder requirements.
Step 7. Click New File...
Step 8. Type synth.txt and press Enter.
Step 9. Paste the grey box below.
Step 10. Press Ctrl+S.
@@include requirements/synth.txt
FILE 3: requirements/eval.txt
Step 11. Right-click the folder requirements.
Step 12. Click New File...
Step 13. Type eval.txt and press Enter.
Step 14. Paste the grey box below.
Step 15. Press Ctrl+S.
@@include requirements/eval.txt
| Line | What it means |
| # ... | a comment line, ignored by the installer |
| unsloth==2026.10.7 and unsloth_zoo==2026.10.7 | exact Unsloth version; it brings matching trl, peft, transformers, bitsandbytes, datasets and torch versions |
| jinja2 | the template engine used by chat templates (the validator needs it) |
| huggingface_hub | upload/download to the Hugging Face Hub |
| distilabel[openai]==1.5.3 | distilabel with its OpenAI-compatible client (talks to vLLM) |
| datasets<4, openai<2, requests, beautifulsoup4 | the older versions distilabel 1.5.3 works with; we found during testing that without them distilabel fails to import |
| openai==3.35.1 | the client used to call vLLM in evaluation and the API |
| rouge-score, bert-score | the two text-similarity metrics |
| transformers==5.20.2, torch | needed by BERTScore |
| fastapi, uvicorn, langsmith, httpx, pytest | the web API, its server, tracing, and the test tools |
CHECKPOINT: ls requirements shows eval.txt  synth.txt  train.txt

## SECTION 2B.8 - Create the three Python environments
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: .venv-train, .venv-synth and .venv-eval with all packages installed.
Step 1. Create the training environment:
```
uv venv .venv-train --python 3.12
```
You should see: "Creating virtual environment at: .venv-train"
Step 2. Install the training packages (3-6 minutes):
```
uv pip install --python .venv-train -r requirements/train.txt --torch-backend cu126
```
You should see: a list ending with lines such as + torch==2.14.1+cu126, + trl==1.15.0, + transformers==5.17.0, + unsloth==2026.10.7
Step 3. Create the synthetic-data environment:
```
uv venv .venv-synth --python 3.12
```
Step 4. Install its packages:
```
uv pip install --python .venv-synth -r requirements/synth.txt
```
You should see: + distilabel==1.5.3, + datasets==3.6.0, + openai==1.109.1 among the lines.
Step 5. Create the evaluation environment:
```
uv venv .venv-eval --python 3.12
```
Step 6. Install its packages:
```
uv pip install --python .venv-eval -r requirements/eval.txt --torch-backend cu126
```
Step 7. Check all three exist:
```
ls -a
```
You should see: .venv-eval  .venv-synth  .venv-train among the folders.
How you will use them later (memorise this):
| You want to... | Type first |
| run a data or training script | source ~/medscriptai/.venv-train/bin/activate |
| run a distilabel script | source ~/medscriptai/.venv-synth/bin/activate |
| run evaluation, the judge, the API or tests | source ~/medscriptai/.venv-eval/bin/activate |
| leave any environment | deactivate |
The prompt then starts with (.venv-train), (.venv-synth) or (.venv-eval), so you always see which one is active.
| Piece | What it does |
| uv venv NAME --python 3.12 | creates a private Python 3.12 folder called NAME (uv downloads Python 3.12 if needed) |
| --python .venv-train | install INTO that environment without activating it |
| -r requirements/train.txt | read the package list from the file |
| --torch-backend cu126 | use the PyTorch build for CUDA 12.6, which runs on your driver 570 (CUDA 12.8). PyTorch 2.14 has no cu128 build, so cu126 is the right choice |
Watch out: running a script in the wrong environment is the most common error in this project. Look at the prompt before every command.
CHECKPOINT: the three .venv folders exist and Step 2, 4 and 6 ended without red error text.
If not: "No solution found when resolving dependencies" = run uv self update, then repeat the failed step.

## SECTION 2B.9 - GPU test (PyTorch and Docker)
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: proof that Python and Docker both see the GPU.
Step 1. Activate the training environment:
```
source .venv-train/bin/activate
```
Step 2. Run the PyTorch check:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported(), round(torch.cuda.get_device_properties(0).total_memory/1e9), 'GB')"
```
You should see: 2.14.1+cu126 True NVIDIA A100 80GB PCIe True 85 GB
Step 3. Leave the environment:
```
deactivate
```
Step 4. Check Docker can use the GPU:
```
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```
You should see: the same nvidia-smi table as in Section 2A.10, printed from inside a container.
| If you see... | It means... | Do this next |
| permission denied while trying to connect to the Docker daemon socket | your user is not in the docker group | run sudo usermod -aG docker ubuntu, then close VS Code's remote window (File > Close Remote Connection), reconnect, and repeat Step 4 |
| could not select device driver "" with capabilities: [[gpu]] | NVIDIA container toolkit missing | you picked an image without "with Docker"; redeploy with the right image (Section 2A.8) |
| torch.cuda.is_available() False | CPU build installed | uv pip install --python .venv-train -r requirements/train.txt --torch-backend cu126 --reinstall |
| Piece | What it means |
| torch.cuda.is_available() | True = PyTorch can use the NVIDIA driver |
| is_bf16_supported() | bfloat16 = the 16-bit number format used for training; A100/L40/H100 support it |
| --gpus all | give the container access to the GPU |
CHECKPOINT: Step 2 ends with True ... 85 GB and Step 4 shows the GPU table.

## SECTION 2B.10 - Create .gitignore and start a Git repository (for your portfolio)
WHERE: VS Code Explorer and Terminal, folder ~/medscriptai
GOAL: your project is version-controlled; secrets and big files are never committed.
Step 1. Right-click the top folder MEDSCRIPTAI in the Explorer.
Step 2. Click New File...
Step 3. Type .gitignore and press Enter.
Step 4. Paste the grey box below and press Ctrl+S.
@@include .gitignore
Step 5. In the terminal set your name and e-mail for Git (replace both):
```
git config --global user.name "YOUR NAME" && git config --global user.email "you@example.com"
```
Step 6. Start the repository and make the first commit:
```
git init && git add . && git commit -m "Project skeleton"
```
You should see: "[main (root-commit) ...] Project skeleton" and a list of the files.
| Line in .gitignore | Why |
| .env, deploy/.env, *.pem | secret files must never be in Git |
| data/, outputs/ | datasets and adapters are large; they go to the Hugging Face Hub instead |
| .venv*/ | environments can be recreated from requirements/ |
| __pycache__/, *.pyc, .pytest_cache/ | temporary Python files |
Watch out: commit after every working section (git add . && git commit -m "what you did"). In Part 9 you push the repository to GitHub.
CHECKPOINT: git log --oneline shows "Project skeleton".

## SECTION 2B.11 - Install FakeCloud (free AWS) and the AWS CLI
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: the aws command works against FakeCloud now, and against real AWS when you choose.
BEFORE YOU START: what FakeCloud is. FakeCloud is an open-source program (AGPL-3.0) that pretends to be AWS on your own machine at http://localhost:4566. You use the normal aws command; only the address changes. We tested it while writing this manual: ECR (including real docker push), S3, IAM, Secrets Manager, CloudWatch, EC2 and the SageMaker control plane all work. It has no GPU and cannot run your model; it is for practising the AWS steps for free.
Step 1. Download the AWS CLI v2 installer:
```
cd /tmp && curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
```
Step 2. Unpack and install it:
```
unzip -q awscliv2.zip && sudo ./aws/install && cd ~/medscriptai
```
Step 3. Check:
```
aws --version
```
You should see: aws-cli/2.x.x Python/3.x Linux/...
Step 4. Install FakeCloud version 0.51.0 (pinned; the installer's "latest" lookup can be blocked by GitHub limits):
```
curl -fsSL https://fakecloud.dev/install.sh | bash -s -- --version v0.51.0
```
You should see: "binary: /usr/local/bin/fakecloud" and "Start the emulator: fakecloud"
Step 5. Create a folder for FakeCloud's saved state:
```
mkdir -p ~/fakecloud-data
```
Step 6. Start FakeCloud in the background:
```
nohup fakecloud --storage-mode persistent --data-path ~/fakecloud-data > ~/fakecloud.log 2>&1 &
```
Step 7. Check it is healthy:
```
curl -s http://localhost:4566/_fakecloud/health | head -c 120
```
You should see: {"status":"ok","version":"0.51.0","services":[...
Step 8. Configure the CLI for the column you use. Do every row of your column:
| Step | FakeCloud (free practice) {w=70} | Real AWS (same step) {blue} |
| Credentials | `export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test` (FakeCloud accepts any values) | AWS console > IAM > Users > Create user > name medscriptai-admin > Attach policies directly > AdministratorAccess > Create user > open the user > Security credentials > Create access key > Command Line Interface (CLI) > copy both keys. Then run `aws configure` and paste them |
| Region | `export AWS_DEFAULT_REGION=us-east-1` | in `aws configure` type us-east-1 and output format json |
| Endpoint | `export AWS_ENDPOINT_URL=http://localhost:4566` | `unset AWS_ENDPOINT_URL` |
| Test | `aws sts get-caller-identity` | `aws sts get-caller-identity` |
| You should see | "Account": "123456789012" (FakeCloud's fake account) | your real 12-digit account number |
| Safety | nothing to do | AWS console > Billing and Cost Management > Budgets > Create budget > Use a template > Monthly cost budget > US$10 > your e-mail > Create budget. Also enable MFA on the root user |
| Piece | What it does |
| nohup ... & | run in the background and keep running after you close the terminal |
| --storage-mode persistent --data-path | FakeCloud saves what you create, so it survives a restart |
| > ~/fakecloud.log 2>&1 | write its messages into ~/fakecloud.log |
| AWS_ENDPOINT_URL | the ONE variable that decides FakeCloud or real AWS |
Watch out: before every real-AWS step run echo $AWS_ENDPOINT_URL; it must print an empty line. export lines only last for the current terminal: in a new terminal, repeat the FakeCloud exports.
Watch out: never use root-user keys; never put access keys in code or Git.
CHECKPOINT: with the FakeCloud exports, aws sts get-caller-identity returns 123456789012.
If not: "Could not connect to the endpoint URL" = FakeCloud is not running: repeat Step 6 and read tail ~/fakecloud.log.

## SECTION 2B.12 - (Real AWS only) Request GPU quota early
WHERE: web browser > AWS console, region us-east-1 (top-right region menu: US East (N. Virginia))
GOAL: permission to start one GPU machine in Part 6B (new accounts have 0; approval takes hours to days).
Step 1. Open https://console.aws.amazon.com/servicequotas
Step 2. Click AWS services.
Step 3. Search and click Amazon Elastic Compute Cloud (Amazon EC2).
Step 4. Search Running On-Demand G and VT instances and click it.
Step 5. Click Request increase at account level.
Step 6. In Increase quota value type 8 and click Request.
You should see: the request with status Pending (later Approved, by e-mail).
CHECKPOINT: the request exists. Continue with Part 3 and Part 4 while you wait.

## PART 2 RESULT CHECK (all must be true before Part 4)
| Check | Command | Expected |
| GPU visible | nvidia-smi | NVIDIA A100 80GB PCIe |
| folders exist | ls ~/medscriptai | api data deploy outputs requirements scripts serve tests |
| three environments | ls -a ~/medscriptai | .venv-train .venv-synth .venv-eval |
| PyTorch on GPU | Section 2B.9 Step 2 | ... True ... 85 GB |
| Docker on GPU | docker run --rm --gpus all ubuntu:24.04 nvidia-smi | GPU table |
| token loaded | echo ${HF_TOKEN:0:3} | hf_ |
| FakeCloud | curl -s localhost:4566/_fakecloud/health | "status":"ok" |
| tmux | tmux ls | no error (or "no server running") |
