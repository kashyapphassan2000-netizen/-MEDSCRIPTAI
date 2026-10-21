#!/bin/bash
# Runs ONCE, as root, the first time the EC2 machine boots (cloud-init "user data").
# deploy.sh replaces the __PLACEHOLDERS__ before sending it to AWS.
set -euxo pipefail                                          # -x prints every command into the boot log
REGION=__REGION__
REGISTRY=__REGISTRY__
BUCKET=__BUCKET__
TAG=__TAG__
APP_DIR=/opt/medscriptai
mkdir -p $APP_DIR/adapters $APP_DIR/hf-cache

# 1) Adapters from S3 (the machine's IAM role gives read access; no keys on the box).
aws s3 sync "s3://$BUCKET/adapters" $APP_DIR/adapters --region "$REGION"

# 2) Secrets from Secrets Manager into a root-only .env file.
secret() { aws secretsmanager get-secret-value --region "$REGION" --secret-id "medscriptai/$1" \
           --query SecretString --output text 2>/dev/null || true; }
cat > $APP_DIR/.env <<EOF
ADAPTERS_DIR=$APP_DIR/adapters
HF_HOME=$APP_DIR/hf-cache
HF_TOKEN=$(secret HF_TOKEN)
API_KEY=$(secret API_KEY)
LANGSMITH_API_KEY=$(secret LANGSMITH_API_KEY)
LANGSMITH_TRACING=true
VLLM_IMAGE=vllm/vllm-openai:v0.32.0
VLLM_ENABLE_CUDA_COMPATIBILITY=0
API_IMAGE=$REGISTRY/medscriptai-api:$TAG
AWS_REGION=$REGION
EOF
chmod 600 $APP_DIR/.env

# 3) Log Docker in to ECR and pull our API image.
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

# 4) The compose files that deploy.sh uploaded to S3.
aws s3 cp "s3://$BUCKET/deploy/docker-compose.yml" $APP_DIR/ --region "$REGION"
aws s3 cp "s3://$BUCKET/deploy/compose.aws.yml" $APP_DIR/ --region "$REGION"

# 5) Start vLLM + API. vLLM downloads the 16 GB base model on first start.
cd $APP_DIR
docker compose -f docker-compose.yml -f compose.aws.yml --env-file .env up -d
