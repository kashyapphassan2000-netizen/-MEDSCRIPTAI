#!/usr/bin/env bash
# Delete everything deploy.sh created, so the AWS bill stops.  TARGET=aws bash deploy/teardown.sh
set -uo pipefail                                            # no -e: keep going if one item is already gone
TARGET="${TARGET:-fakecloud}"
export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
APP=medscriptai
if [ "$TARGET" = "fakecloud" ]; then
  export AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test
fi
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="$APP-artifacts-$ACCOUNT_ID-$AWS_REGION"

IDS=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=$APP-gpu" "Name=instance-state-name,Values=pending,running,stopped" \
      --query 'Reservations[].Instances[].InstanceId' --output text)
[ -n "$IDS" ] && aws ec2 terminate-instances --instance-ids $IDS >/dev/null && echo "terminated $IDS"
[ -n "$IDS" ] && timeout 600 aws ec2 wait instance-terminated --instance-ids $IDS 2>/dev/null
for try in 1 2 3 4 5 6; do                                  # the SG can only go once the instance is gone
  aws ec2 delete-security-group --group-name "$APP-sg" >/dev/null 2>&1 && { echo "deleted security group"; break; }
  sleep 10
done
aws ec2 delete-key-pair --key-name "$APP-key" >/dev/null 2>&1 && echo "deleted key pair"
aws ecr delete-repository --repository-name "$APP-api" --force >/dev/null 2>&1 && echo "deleted ECR repo"
aws s3 rb "s3://$BUCKET" --force >/dev/null 2>&1 && echo "deleted bucket"
for name in HF_TOKEN LANGSMITH_API_KEY API_KEY; do
  aws secretsmanager delete-secret --secret-id "$APP/$name" --force-delete-without-recovery >/dev/null 2>&1 && echo "deleted secret $name"
done
aws cloudwatch delete-alarms --alarm-names "$APP-cpu-high" 2>/dev/null && echo "deleted alarm"
aws logs delete-log-group --log-group-name "/$APP/api" 2>/dev/null && echo "deleted log group"
aws iam remove-role-from-instance-profile --instance-profile-name "$APP-ec2-profile" --role-name "$APP-ec2-role" 2>/dev/null
aws iam delete-instance-profile --instance-profile-name "$APP-ec2-profile" 2>/dev/null && echo "deleted instance profile"
aws iam delete-role-policy --role-name "$APP-ec2-role" --policy-name "$APP-runtime" 2>/dev/null
aws iam delete-role --role-name "$APP-ec2-role" 2>/dev/null && echo "deleted role"
echo "TEARDOWN DONE ($TARGET)"
