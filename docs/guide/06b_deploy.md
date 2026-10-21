# PART 6B - ADVANCED: Docker image, Compose stack, and cloud deployment (FakeCloud and AWS side by side)

## BEFORE YOU START: the words of this part
| Word | Plain meaning |
| Docker image | a frozen package: an operating system layer + Python + your code + its libraries. Built once, runs the same everywhere |
| container | a running copy of an image |
| Dockerfile | the recipe to build an image, step by step |
| registry | an online store for images (Docker Hub is public; AWS ECR is your private one) |
| tag | a version label of an image, e.g. medscriptai-api:v1 |
| Docker Compose | starts several containers together (vLLM + API) from one YAML file |
| ECR | Elastic Container Registry: AWS's private image registry |
| S3 | AWS file storage (buckets) |
| IAM role | a set of permissions a machine can use without any password on its disk |
| Secrets Manager | AWS's safe for passwords and tokens |
| CloudWatch | AWS's logs and alarms |
| security group | AWS's firewall: which ports are open, to whom |
| EC2 | AWS virtual machines; g6.xlarge = 1x NVIDIA L4 24 GB GPU, about US$0.805 per hour |
| AMI | the disk image an EC2 machine starts from; we use AWS's Deep Learning AMI (Ubuntu 24.04, NVIDIA driver 595, Docker included) |
| user data | a script the EC2 machine runs automatically on its first boot |
The plan of this part:
```
6B.1-6B.3  build and test the API Docker image on Hyperstack
6B.4-6B.6  create the Compose files and run vLLM + API together on Hyperstack
6B.7       the boot script the AWS machine runs on first start
6B.8       deploy step by step: ECR -> S3 -> Secrets -> IAM -> CloudWatch -> firewall -> EC2 (two columns)
6B.9-6B.10 the same deployment as one script, and the teardown script
6B.11      optional: SageMaker objects
```

## SECTION 6B.1 - Create api/Dockerfile
WHERE: VS Code Explorer, folder api
GOAL: the recipe for the API image.
Step 1. Right-click the folder api.
Step 2. Click New File...
Step 3. Type Dockerfile (no extension, capital D) and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include api/Dockerfile
| Line | What it does (plain English) |
| `ARG BASE_IMAGE=public.ecr.aws/docker/library/python:3.12-slim` | the official Python image through AWS's ECR Public mirror; we hit Docker Hub's anonymous download limit (HTTP 429) while testing, so this avoids it. Switch with --build-arg BASE_IMAGE=python:3.12-slim |
| `FROM ${BASE_IMAGE}` | start from a small Debian Linux with Python 3.12 |
| `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1` | no .pyc files; log lines appear immediately |
| `WORKDIR /app` | all following commands run in /app inside the image |
| `COPY api/requirements.txt .` then `RUN pip install --no-cache-dir -r requirements.txt` | install the pinned libraries in their own layer, so code changes rebuild fast |
| `COPY scripts/common.py api/app.py ./` | only the two files the API needs; this is why you build from the project root |
| `RUN useradd --create-home appuser` and `USER appuser` | the container does not run as root (security) |
| `EXPOSE 8080` | documents the port |
| `HEALTHCHECK ... urlopen('http://localhost:8080/health')` | Docker marks the container healthy or unhealthy |
| `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]` | start the API with 2 worker processes |
CHECKPOINT: ls api shows Dockerfile, app.py, requirements.txt

## SECTION 6B.2 - Build the image
WHERE: VS Code Terminal, folder ~/medscriptai (the project ROOT)
GOAL: an image called medscriptai-api:v1
Step 1. Build:
```
cd ~/medscriptai && docker build -f api/Dockerfile -t medscriptai-api:v1 .
```
You should see: build steps [1/7] ... [7/7] and "naming to docker.io/library/medscriptai-api:v1".
Step 2. List it:
```
docker images medscriptai-api
```
You should see: medscriptai-api   v1   ...   about 290MB
| Piece | What it does |
| -f api/Dockerfile | which recipe file |
| -t medscriptai-api:v1 | the image name and tag |
| . (the final dot) | the build context = the current folder (the project root), so COPY scripts/common.py works |
| If you see... | Do this |
| 429 Too Many Requests | wait 10 minutes, or add --build-arg BASE_IMAGE=python:3.12-slim |
| COPY failed: file not found | you are not in ~/medscriptai: run cd ~/medscriptai and build again |
CHECKPOINT: docker images shows medscriptai-api v1.

## SECTION 6B.3 - Test the container against the running vLLM
WHERE: VS Code Terminal; student vLLM running on port 8000 (Section 6A.6)
GOAL: the containerised API works and is healthy.
Step 1. Run it:
```
docker run -d --name api-test --network host -e VLLM_URL=http://localhost:8000/v1 -e API_KEY=dev-key medscriptai-api:v1
```
Step 2. Check readiness (wait 5 seconds first):
```
sleep 5 && curl -s localhost:8080/ready | jq -r .status
```
You should see: ready
Step 3. Check Docker's health status after 30 seconds:
```
sleep 30 && docker inspect --format '{{.State.Health.Status}} user={{.Config.User}}' api-test
```
You should see: healthy user=appuser (this exact result was produced when we tested the image)
Step 4. Remove the test container:
```
docker rm -f api-test
```
CHECKPOINT: you saw ready and healthy.

## SECTION 6B.4 - Create the Compose files
WHERE: VS Code Explorer, folder deploy
GOAL: docker-compose.yml, compose.aws.yml and .env.example
FILE 1: deploy/docker-compose.yml
Step 1. Right-click the folder deploy.
Step 2. Click New File...
Step 3. Type docker-compose.yml and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include deploy/docker-compose.yml
| Line | What it does (plain English) |
| `services: vllm: ... api: ...` | two containers that Compose starts and connects |
| `image: ${VLLM_IMAGE:-vllm/vllm-openai:v0.32.0}` | the vLLM image; .env sets the -cu129 build for your driver |
| `command: > --model ... --lora-modules ...` | the same arguments as start_vllm.sh ("> " folds the lines into one) |
| `${ADAPTERS_DIR:?set ADAPTERS_DIR in .env}` | Compose STOPS with this message if the variable is missing |
| `deploy.resources.reservations.devices: driver nvidia, count all, capabilities [gpu]` | give the container the GPU |
| `healthcheck: ... urlopen('http://localhost:8000/health')`, `start_period: 120s` | vLLM is healthy when /health answers; 2 minutes grace to load |
| `restart: unless-stopped` | restart after crashes and reboots |
| no `ports:` under vllm | vLLM is NOT reachable from outside, only by the api container (by the name vllm) |
| `VLLM_URL: http://vllm:8000/v1` | Compose's internal DNS: a service name works as a host name |
| `ports: "8080:8080"` | only the API is published |
| `depends_on: vllm: condition: service_healthy` | start the API only after vLLM is healthy |
FILE 2: deploy/compose.aws.yml (extra settings used only on AWS)
Step 6. Right-click the folder deploy.
Step 7. Click New File...
Step 8. Type compose.aws.yml and press Enter.
Step 9. Paste the grey box below.
Step 10. Press Ctrl+S.
@@include deploy/compose.aws.yml
| Line | What it does (plain English) |
| `logging: driver: awslogs` | Docker sends container logs to CloudWatch Logs |
| `awslogs-group: /medscriptai/api` | the log group name |
| `awslogs-create-group: "true"` | create the group if missing |
FILE 3: deploy/.env.example (a template; the real .env is never committed)
Step 11. Right-click the folder deploy.
Step 12. Click New File...
Step 13. Type .env.example and press Enter.
Step 14. Paste the grey box below.
Step 15. Press Ctrl+S.
@@include deploy/.env.example
CHECKPOINT: ls -a deploy shows .env.example, compose.aws.yml, docker-compose.yml

## SECTION 6B.5 - Fill in deploy/.env
WHERE: VS Code Terminal and editor
GOAL: your real settings file.
Step 1. Copy the template:
```
cd ~/medscriptai && cp deploy/.env.example deploy/.env
```
Step 2. Make a long random API key:
```
openssl rand -hex 24
```
Step 3. In the Explorer, click deploy > .env to open it.
Step 4. Replace hf_REPLACE_ME with your HF token, REPLACE_WITH_A_LONG_RANDOM_STRING with the key from Step 2, and lsv2_REPLACE_ME with your LangSmith key.
Step 5. Press Ctrl+S.
| Line in .env | Value | Why |
| ADAPTERS_DIR | /home/ubuntu/medscriptai/outputs/adapters | where your adapters are |
| HF_HOME | /ephemeral/hf-cache | the model cache from Section 2B.4 |
| VLLM_IMAGE | vllm/vllm-openai:v0.32.0-cu129 | matches your R570 driver |
| VLLM_ENABLE_CUDA_COMPATIBILITY | 1 | needed with R570 + the cu129 image |
| API_IMAGE | medscriptai-api:v1 | your image |
Watch out: .env is in .gitignore; never commit it.
CHECKPOINT: grep -c REPLACE deploy/.env prints 0.

## SECTION 6B.6 - Run the full stack with one command
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: vLLM + API started together by Compose.
Step 1. Stop the standalone servers (Compose starts its own vLLM):
```
docker rm -f medscriptai-vllm medscriptai-teacher
```
Step 2. Start the stack:
```
cd ~/medscriptai/deploy && docker compose --env-file .env up -d && cd ..
```
Step 3. Watch the status (2-4 minutes until healthy; Ctrl+C to stop watching):
```
watch -n 10 docker compose -f deploy/docker-compose.yml ps
```
You should see: vllm ... (healthy) and api ... Up.
Step 4. Call the summary task through the stack:
```
curl -s -X POST localhost:8080/v1/generate -H "Content-Type: application/json" -H "X-API-Key: $(grep ^API_KEY deploy/.env | cut -d= -f2)" -d '{"task":"summary","input":"A 54-year-old man with type 2 diabetes presented with chest pain for 2 hours. ECG showed ST elevation. He underwent primary PCI."}' | jq
```
You should see: "model": "summary" and parsed_json with fields such as "patient information".
Step 5. Prove vLLM is NOT exposed (this must fail):
```
curl -s --max-time 3 localhost:8000/v1/models || echo "port 8000 closed - correct"
```
You should see: port 8000 closed - correct
CHECKPOINT: Step 4 returned parsed JSON and Step 5 printed "closed - correct".
If not: docker compose -f deploy/docker-compose.yml logs vllm | tail -50 shows vLLM's error.

## SECTION 6B.7 - Create deploy/ec2_user_data.sh (what the AWS machine does on first boot)
WHERE: VS Code Explorer, folder deploy
GOAL: the boot script.
Step 1. Right-click the folder deploy.
Step 2. Click New File...
Step 3. Type ec2_user_data.sh and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include deploy/ec2_user_data.sh
| Line | What it does (plain English) |
| `set -euxo pipefail` | stop on errors and print every command into /var/log/cloud-init-output.log |
| `REGION=__REGION__` etc. | placeholders that row 22 of Section 6B.8 (or deploy.sh) replaces with real values |
| `aws s3 sync "s3://$BUCKET/adapters" ...` | download the adapters using the machine's role (no keys on the box) |
| `secret() { aws secretsmanager get-secret-value ... }` | read one secret |
| `cat > $APP_DIR/.env <<EOF ... EOF` and `chmod 600` | write the settings file, readable only by root |
| `VLLM_IMAGE=vllm/vllm-openai:v0.32.0` and `VLLM_ENABLE_CUDA_COMPATIBILITY=0` | the AWS Deep Learning AMI has driver 595 / CUDA 13.2, so the default CUDA 13 image runs natively |
| `aws ecr get-login-password ... \| docker login ...` | let Docker pull your private image |
| `aws s3 cp ".../docker-compose.yml" ...` | the same Compose files you tested on Hyperstack |
| `docker compose ... up -d` | start vLLM + API with CloudWatch logging |
CHECKPOINT: ls deploy shows ec2_user_data.sh

## SECTION 6B.8 - Deploy step by step (registry, storage, secrets, IAM, logs, firewall, GPU machine)
WHERE: VS Code Terminal on the Hyperstack VM, folder ~/medscriptai
GOAL: the same production deployment done twice: first FREE on FakeCloud, then for real on AWS.
Before you start (both columns): FakeCloud running (curl localhost:4566/_fakecloud/health); export API_KEY=$(grep ^API_KEY deploy/.env | cut -d= -f2); HF_TOKEN and LANGSMITH_API_KEY set (echo ${HF_TOKEN:0:3}). Blue column only: GPU quota approved (Section 2B.12).
Step 1. Write the trust policy file (WHO may use the role: EC2 machines):
```
cat > /tmp/medscriptai-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
```
Step 2. Do the rows of the table below one by one in your column. Then come back for Step 3.
| Step | FakeCloud (free practice) {w=60} | Real AWS (same step) {blue} |
| 1 Target | `export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1` | `unset AWS_ENDPOINT_URL && export AWS_DEFAULT_REGION=us-east-1` |
| 2 Who am I | `export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) && echo $ACCOUNT_ID` | same (prints your real 12-digit ID) |
| 3 Registry address | `export REGISTRY=localhost:4566` | `export REGISTRY=$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com` |
| 4 Create the image repository | `aws ecr create-repository --repository-name medscriptai-api --image-scanning-configuration scanOnPush=true` | same |
| 5 Log Docker in | `aws ecr get-login-password \| docker login --username AWS --password-stdin $REGISTRY` | same; you should see Login Succeeded |
| 6 Tag the image | `docker tag medscriptai-api:v1 $REGISTRY/medscriptai-api:v1` | same |
| 7 Push | `docker push $REGISTRY/medscriptai-api:v1` | same (about 290 MB upload) |
| 8 Verify | `aws ecr describe-images --repository-name medscriptai-api --query 'imageDetails[].imageTags'` | same; console: ECR > Repositories > medscriptai-api |
| 9 Bucket | `export BUCKET=medscriptai-artifacts-$ACCOUNT_ID-us-east-1 && aws s3 mb s3://$BUCKET` | same (ignore "already exists" if you made it in Part 4) |
| 10 Adapters + compose files to S3 | `aws s3 sync outputs/adapters s3://$BUCKET/adapters --exclude "*/checkpoint-*" && aws s3 cp deploy/docker-compose.yml s3://$BUCKET/deploy/ && aws s3 cp deploy/compose.aws.yml s3://$BUCKET/deploy/` | same |
| 11 Secrets | `for n in HF_TOKEN LANGSMITH_API_KEY API_KEY; do aws secretsmanager create-secret --name medscriptai/$n --secret-string "${!n}"; done` | same |
| 12 IAM role | `aws iam create-role --role-name medscriptai-ec2-role --assume-role-policy-document file:///tmp/medscriptai-trust.json` | same |
| 13 Permissions policy file | write it with the command in Step 3 below, then come back | same |
| 14 Attach the policy | `aws iam put-role-policy --role-name medscriptai-ec2-role --policy-name medscriptai-runtime --policy-document file:///tmp/medscriptai-policy.json` | same |
| 15 Instance profile | `aws iam create-instance-profile --instance-profile-name medscriptai-ec2-profile && aws iam add-role-to-instance-profile --instance-profile-name medscriptai-ec2-profile --role-name medscriptai-ec2-role` | same, then `sleep 10` (real IAM needs a few seconds) |
| 16 Logs | `aws logs create-log-group --log-group-name /medscriptai/api && aws logs put-retention-policy --log-group-name /medscriptai/api --retention-in-days 14` | same |
| 17 Alarm | `aws cloudwatch put-metric-alarm --alarm-name medscriptai-cpu-high --namespace AWS/EC2 --metric-name CPUUtilization --statistic Average --period 300 --evaluation-periods 2 --threshold 90 --comparison-operator GreaterThanThreshold` | same |
| 18 Firewall | `export SG_ID=$(aws ec2 create-security-group --group-name medscriptai-sg --description "MedScriptAI API" --query GroupId --output text)` | same (uses your default VPC) |
| 19 Allow only your IP | `export MY_IP=$(curl -s https://checkip.amazonaws.com) && for p in 22 8080; do aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port $p --cidr $MY_IP/32; done` | same. MY_IP is the Hyperstack VM's IP; to call the API from your laptop too, repeat with your laptop's IP |
| 20 SSH key for EC2 | `aws ec2 create-key-pair --key-name medscriptai-key --query KeyMaterial --output text > ~/.ssh/medscriptai-key.pem && chmod 400 ~/.ssh/medscriptai-key.pem` | same |
| 21 Machine image (AMI) | `export AMI=$(aws ec2 describe-images --owners amazon --query 'Images[0].ImageId' --output text)` (FakeCloud's demo AMIs) | `export AMI=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" "Name=state,Values=available" --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)` |
| 22 Boot script | `sed -e "s\|__REGION__\|us-east-1\|g" -e "s\|__REGISTRY__\|$REGISTRY\|g" -e "s\|__BUCKET__\|$BUCKET\|g" -e "s\|__TAG__\|v1\|g" deploy/ec2_user_data.sh > /tmp/user-data.sh` | same |
| 23 Launch | `aws ec2 run-instances --image-id $AMI --instance-type g6.xlarge --key-name medscriptai-key --security-group-ids $SG_ID --iam-instance-profile Name=medscriptai-ec2-profile --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' --user-data file:///tmp/user-data.sh --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=medscriptai-gpu}]' --query 'Instances[0].InstanceId' --output text` | same. BILLING STARTS NOW (about US$0.805/h + disk) |
| 24 Get its IP | `aws ec2 describe-instances --filters Name=tag:Name,Values=medscriptai-gpu --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text` | same |
| You should see | an i-... id, pending/running, and a fake IP such as 52.0.0.10. FakeCloud starts a small container to represent the machine; there is no GPU and no model | a real public IP; the boot script needs 10-15 minutes (it downloads about 30 GB) |
| 25 Watch the boot | not applicable | `ssh -i ~/.ssh/medscriptai-key.pem ubuntu@EC2_IP` then `sudo tail -f /var/log/cloud-init-output.log` until docker compose reports Started |
| 26 Test the API | not applicable (no model behind FakeCloud) | `curl -s http://EC2_IP:8080/ready` then the /v1/generate call from Section 6A.9 with your API_KEY |
| 27 Read the logs | `aws logs describe-log-groups --query 'logGroups[].logGroupName'` | console: CloudWatch > Log groups > /medscriptai/api, or `aws logs tail /medscriptai/api --follow` |
| 28 TEAR DOWN | do Section 6B.10 (it creates teardown.sh and runs it on FakeCloud) | after Section 6B.10: `TARGET=aws bash deploy/teardown.sh`, then check EC2 > Instances shows terminated |
Step 3. The permissions policy for table row 13 (least privilege: only our repo, our bucket, our secrets, and logs):
```
cat > /tmp/medscriptai-policy.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],"Resource":"arn:aws:ecr:us-east-1:$ACCOUNT_ID:repository/medscriptai-api"},
 {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]},
 {"Effect":"Allow","Action":["secretsmanager:GetSecretValue"],"Resource":"arn:aws:secretsmanager:us-east-1:$ACCOUNT_ID:secret:medscriptai/*"},
 {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}]}
EOF
```
New command pieces:
| Piece | What it does |
| `get-login-password \| docker login --password-stdin` | ECR gives a 12-hour token; piping it avoids typing it or saving it in history |
| `docker tag LOCAL REGISTRY/REPO:TAG` | a second name for the same image that says where to push it |
| `"${!n}"` | bash: the value of the variable whose NAME is in n |
| trust policy | WHO may use the role (EC2) |
| permissions policy | WHAT the role may do; every Resource is limited to MedScriptAI's own repo, bucket and secrets |
| instance profile | attaches the role to a machine; programs on it get temporary credentials automatically |
| `--cidr $MY_IP/32` | /32 = exactly one IP address |
| `VolumeSize 150, gp3` | 150 GB SSD: the vLLM image (about 11 GB compressed) + the model (16 GB) need room |
| `--user-data file://...` | the boot script, run once at first start |
| Placeholder | Replace with | Example |
| EC2_IP | the PublicIpAddress from row 24 | 3.91.20.14 |
Watch out: the most expensive mistake in this manual is forgetting row 28 on real AWS. Set a phone reminder when you launch.
CHECKPOINT (FakeCloud): rows 1-24 ran without errors. CHECKPOINT (AWS): row 26 returns an answer from your adapter.
If not: "VcpuLimitExceeded" = GPU quota not approved yet; "InvalidAMIID" = run row 21 in us-east-1; no answer after 20 minutes = read /var/log/cloud-init-output.log on the machine.

## SECTION 6B.9 - Create deploy/deploy.sh (the whole table as one script)
WHERE: VS Code Explorer, folder deploy
GOAL: one command that repeats Section 6B.8 rows 1-24 for either target.
Step 1. Right-click the folder deploy.
Step 2. Click New File...
Step 3. Type deploy.sh and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include deploy/deploy.sh
| Line | What it does (plain English) |
| `TARGET="${TARGET:-fakecloud}"` | the default is the free emulator; real AWS only if you type TARGET=aws |
| `MY_IP=...checkip.amazonaws.com` | your public IP, used to lock the firewall |
| `if [ "$TARGET" = "fakecloud" ]; then export AWS_ENDPOINT_URL=...` | the ONLY difference between the targets |
| `aws ecr describe-repositories ... \|\| aws ecr create-repository ...` | "create if missing": safe to run twice (we tested a second run) |
| `aws s3 mb ... 2>/dev/null \|\| true` | ignore "bucket already exists" |
| `value="${!name:-}"` + `continue` | skip secrets you did not export |
| `create-secret ... \|\| put-secret-value ...` | create or update a secret |
| `aws iam get-role ... \|\| aws iam create-role ...` | create the role once |
| `SG_ID=$(aws ec2 describe-security-groups --filters ...)` | re-use the firewall group on a second run |
| `sort_by(Images,&CreationDate)[-1].ImageId` | the newest matching AMI |
| `sed -e "s\|__REGION__\|$AWS_REGION\|g" ...` | fills the placeholders of the boot script |
| `timeout 300 aws ec2 wait instance-running` | wait at most 5 minutes until the machine runs |
CHECKPOINT: ls deploy shows deploy.sh

## SECTION 6B.10 - Create deploy/teardown.sh and use both scripts
WHERE: VS Code Explorer (create), then VS Code Terminal
GOAL: delete everything with one command; repeat the deployment with one command.
Step 1. Right-click the folder deploy.
Step 2. Click New File...
Step 3. Type teardown.sh and press Enter.
Step 4. Paste the grey box below.
Step 5. Press Ctrl+S.
@@include deploy/teardown.sh
| Line | What it does (plain English) |
| `set -uo pipefail` (no -e) | keep going if one resource is already gone |
| `describe-instances --filters "Name=tag:Name,Values=$APP-gpu"` | find our machines by their Name tag |
| `terminate-instances` then `wait instance-terminated` | delete the machine (stops compute billing) and wait |
| `for try in 1 2 3 4 5 6; do delete-security-group ...` | a firewall group can only be deleted after the machine is fully gone; we retry (found while testing) |
| `delete-repository --force` | delete the repo and its images |
| `aws s3 rb ... --force` | delete the bucket and all files in it |
| `delete-secret --force-delete-without-recovery` | delete immediately (no billed recovery window) |
| the four IAM lines | IAM pieces must be removed in this order |
Step 6. Make the three scripts executable:
```
chmod +x ~/medscriptai/deploy/*.sh
```
Step 7. Clean up what you created by hand on FakeCloud:
```
cd ~/medscriptai && TARGET=fakecloud bash deploy/teardown.sh
```
You should see: lines such as terminated i-..., deleted security group, deleted ECR repo, deleted bucket, ... and TEARDOWN DONE (fakecloud)
Step 8. Deploy again with ONE command (FakeCloud):
```
TARGET=fakecloud bash deploy/deploy.sh
```
You should see: "== 1/7 ECR repository + push the API image" ... "== 7/7 Launch the GPU machine" and "DONE -> API will be at http://52.0.0.10:8080 ..." (this exact flow ran during testing).
Step 9. Tear down again:
```
TARGET=fakecloud bash deploy/teardown.sh
```
Step 10. Real AWS (optional, costs money): run TARGET=aws bash deploy/deploy.sh, wait 10-15 minutes, test with row 26 of Section 6B.8, then run TARGET=aws bash deploy/teardown.sh the same day.
CHECKPOINT: both teardown runs end with TEARDOWN DONE; after a real-AWS run, EC2 > Instances shows terminated.

## SECTION 6B.11 - (Optional) SageMaker endpoint objects
WHERE: VS Code Terminal, folder ~/medscriptai
GOAL: know the managed alternative and practise its API calls for free.
| Step | FakeCloud (free practice) {w=60} | Real AWS (same step) {blue} |
| 1 Model | `aws sagemaker create-model --model-name medscriptai-model --primary-container Image=$REGISTRY/medscriptai-api:v1 --execution-role-arn arn:aws:iam::$ACCOUNT_ID:role/medscriptai-ec2-role` | needs a SageMaker-compatible container (GET /ping and POST /invocations on port 8080) and a role trusted by sagemaker.amazonaws.com; our API image is not that. AWS's Large Model Inference (LMI) containers with a vLLM backend are the usual choice (verify current names in the SageMaker docs) |
| 2 Endpoint config | `aws sagemaker create-endpoint-config --endpoint-config-name medscriptai-cfg --production-variants VariantName=AllTraffic,ModelName=medscriptai-model,InstanceType=ml.g6.xlarge,InitialInstanceCount=1` | same shape |
| 3 Endpoint | `aws sagemaker create-endpoint --endpoint-name medscriptai-endpoint --endpoint-config-name medscriptai-cfg` | same; billed per hour while InService |
| 4 Status | `aws sagemaker describe-endpoint --endpoint-name medscriptai-endpoint --query EndpointStatus` | same |
| You should see | "InService" (FakeCloud emulates the control plane only; invoke-endpoint returned NotFound when we tested it) | "Creating" for 5-15 minutes, then "InService" |
| 5 Delete | `aws sagemaker delete-endpoint --endpoint-name medscriptai-endpoint` | same, plus delete-endpoint-config and delete-model |
Interview answer: EC2 + Docker Compose (what you built) gives full control and is cheapest for one always-on model; SageMaker adds managed autoscaling, safe deployments and monitoring, at a higher price and with stricter container rules.
CHECKPOINT: on FakeCloud describe-endpoint returned InService and delete-endpoint succeeded.

## SECTION 6B.12 - Commit
Step 1. Run:
```
cd ~/medscriptai && git add . && git commit -m "Part 6B: Docker, Compose and cloud deployment"
```
Watch out: git status must NOT list deploy/.env (it is ignored). If it does, check .gitignore.

## PART 6B RESULT CHECK
| Check | Command | Expected |
| image | docker images medscriptai-api | v1, about 290 MB |
| container healthy | Section 6B.3 Step 3 | healthy user=appuser |
| stack | docker compose -f deploy/docker-compose.yml ps | vllm healthy, api Up |
| vLLM not exposed | curl --max-time 3 localhost:8000/v1/models | fails |
| FakeCloud deploy | TARGET=fakecloud bash deploy/deploy.sh | DONE |
| clean teardown | TARGET=fakecloud bash deploy/teardown.sh | TEARDOWN DONE (fakecloud) |
| (AWS) live API | curl http://EC2_IP:8080/ready | "status": "ready" |

## 6B.13 Common mistake at this level
The mistake: secrets baked into images or Git, vLLM's port open to the internet, and GPU machines left running.
How to spot it: docker history medscriptai-api:v1 shows no secrets; Section 6B.6 Step 5 fails as it should; after teardown, aws ec2 describe-instances shows nothing running.
