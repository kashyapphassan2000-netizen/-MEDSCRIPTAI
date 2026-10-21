#!/usr/bin/env bash
# One script, two targets:
#   TARGET=fakecloud bash deploy/deploy.sh   -> free local AWS emulator (localhost:4566)
#   TARGET=aws       bash deploy/deploy.sh   -> your real AWS account (costs money!)
# Run from the repo folder after: the API image is built and outputs/adapters/ exists.
set -euo pipefail

TARGET="${TARGET:-fakecloud}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="$AWS_REGION"
APP=medscriptai
TAG="${TAG:-v1}"
MY_IP="${MY_IP:-$(curl -s https://checkip.amazonaws.com)}"   # only YOUR IP may reach SSH and the API

if [ "$TARGET" = "fakecloud" ]; then
  export AWS_ENDPOINT_URL=http://localhost:4566             # every aws command now talks to FakeCloud
  export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test  # FakeCloud accepts any keys
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ "$TARGET" = "fakecloud" ]; then
  REGISTRY=localhost:4566
else
  REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
fi
BUCKET="$APP-artifacts-$ACCOUNT_ID-$AWS_REGION"               # bucket names must be globally unique
echo "== target=$TARGET account=$ACCOUNT_ID region=$AWS_REGION registry=$REGISTRY"

echo "== 1/7 ECR repository + push the API image"
aws ecr describe-repositories --repository-names "$APP-api" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$APP-api" --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker tag "$APP-api:$TAG" "$REGISTRY/$APP-api:$TAG"
docker push "$REGISTRY/$APP-api:$TAG"

echo "== 2/7 S3 bucket + upload adapters and reports"
aws s3 mb "s3://$BUCKET" 2>/dev/null || true
aws s3 sync outputs/adapters "s3://$BUCKET/adapters" --exclude "*/checkpoint-*" --only-show-errors
[ -d outputs/reports ] && aws s3 sync outputs/reports "s3://$BUCKET/reports" --only-show-errors
aws s3 cp deploy/docker-compose.yml "s3://$BUCKET/deploy/docker-compose.yml" --only-show-errors
aws s3 cp deploy/compose.aws.yml "s3://$BUCKET/deploy/compose.aws.yml" --only-show-errors

echo "== 3/7 Secrets Manager (HF token, LangSmith key, API key)"
for name in HF_TOKEN LANGSMITH_API_KEY API_KEY; do
  value="${!name:-}"
  [ -z "$value" ] && { echo "   skip $name (not set in your shell)"; continue; }
  aws secretsmanager create-secret --name "$APP/$name" --secret-string "$value" >/dev/null 2>&1 \
    || aws secretsmanager put-secret-value --secret-id "$APP/$name" --secret-string "$value" >/dev/null
  echo "   stored $APP/$name"
done

echo "== 4/7 IAM role the EC2 machine will use (least privilege)"
cat > /tmp/$APP-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
cat > /tmp/$APP-policy.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
 {"Effect":"Allow","Action":["ecr:BatchGetImage","ecr:GetDownloadUrlForLayer"],"Resource":"arn:aws:ecr:$AWS_REGION:$ACCOUNT_ID:repository/$APP-api"},
 {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],"Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"]},
 {"Effect":"Allow","Action":["secretsmanager:GetSecretValue"],"Resource":"arn:aws:secretsmanager:$AWS_REGION:$ACCOUNT_ID:secret:$APP/*"},
 {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}]}
EOF
aws iam get-role --role-name "$APP-ec2-role" >/dev/null 2>&1 \
  || aws iam create-role --role-name "$APP-ec2-role" --assume-role-policy-document "file:///tmp/$APP-trust.json" >/dev/null
aws iam put-role-policy --role-name "$APP-ec2-role" --policy-name "$APP-runtime" --policy-document "file:///tmp/$APP-policy.json"
aws iam get-instance-profile --instance-profile-name "$APP-ec2-profile" >/dev/null 2>&1 || {
  aws iam create-instance-profile --instance-profile-name "$APP-ec2-profile" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$APP-ec2-profile" --role-name "$APP-ec2-role"
  sleep 10                                                  # real IAM needs a few seconds to propagate
}

echo "== 5/7 CloudWatch: log group + CPU alarm"
aws logs create-log-group --log-group-name "/$APP/api" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/$APP/api" --retention-in-days 14
aws cloudwatch put-metric-alarm --alarm-name "$APP-cpu-high" --namespace AWS/EC2 --metric-name CPUUtilization \
  --statistic Average --period 300 --evaluation-periods 2 --threshold 90 --comparison-operator GreaterThanThreshold

echo "== 6/7 Security group: port 8080 (API) and 22 (SSH) only from $MY_IP"
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$APP-sg" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "$APP-sg" --description "MedScriptAI API" \
          --query GroupId --output text)
  for port in 22 8080; do
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port $port --cidr "$MY_IP/32" >/dev/null
  done
fi

echo "== 7/7 Launch the GPU machine (g6.xlarge = 1x NVIDIA L4 24 GB)"
if [ "$TARGET" = "fakecloud" ]; then
  AMI=$(aws ec2 describe-images --owners amazon --query 'Images[0].ImageId' --output text)
else
  AMI=$(aws ec2 describe-images --owners amazon \
        --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" "Name=state,Values=available" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
fi
sed -e "s|__REGION__|$AWS_REGION|g" -e "s|__REGISTRY__|$REGISTRY|g" -e "s|__BUCKET__|$BUCKET|g" -e "s|__TAG__|$TAG|g" \
  deploy/ec2_user_data.sh > /tmp/$APP-user-data.sh
aws ec2 describe-key-pairs --key-names "$APP-key" >/dev/null 2>&1 || {
  aws ec2 create-key-pair --key-name "$APP-key" --query KeyMaterial --output text > "$HOME/.ssh/$APP-key.pem"
  chmod 400 "$HOME/.ssh/$APP-key.pem"
}
INSTANCE_ID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "${INSTANCE_TYPE:-g6.xlarge}" \
  --key-name "$APP-key" --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$APP-ec2-profile" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' \
  --user-data "file:///tmp/$APP-user-data.sh" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$APP-gpu}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "   instance $INSTANCE_ID starting (AMI $AMI)"
timeout 300 aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" 2>/dev/null || sleep 5
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "DONE  ->  API will be at http://$IP:8080 in ~10-15 min (first boot downloads ~30 GB)."
echo "      SSH: ssh -i ~/.ssh/$APP-key.pem ubuntu@$IP   |   boot log: sudo tail -f /var/log/cloud-init-output.log"
