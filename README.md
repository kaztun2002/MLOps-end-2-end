# MLOps-end-2-end

End-to-end time-series MLOps pipeline with DVC, LightGBM, Evidently, Docker, and AWS Terraform.

## Run with Docker

Build and run the complete DVC pipeline locally:

```powershell
docker build -t mlops-end-2-end:local .
docker run --rm `
	-v "${PWD}/data:/app/data" `
	-v "${PWD}/models:/app/models" `
	-v "${PWD}/reports:/app/reports" `
	mlops-end-2-end:local
```

The container runs `dvc repro` by default. When `ARTIFACTS_BUCKET` is set, it uploads generated `data/`, `models/`, and `reports/` files to an S3 run prefix using the AWS credential chain. Credentials are not stored in the image.

## Provision AWS infrastructure

The Terraform configuration provisions an ECR repository, an on-demand ECS Fargate task definition, CloudWatch logs, a restricted task security group, and an encrypted/versioned S3 artifact bucket. Supply an existing VPC and subnets in a local `terraform.tfvars` file based on `infra/terraform/terraform.tfvars.example`.

```powershell
Copy-Item infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform plan
terraform -chdir=infra/terraform apply
```

Build and push the image after applying Terraform:

```powershell
$region = terraform -chdir=infra/terraform output -raw aws_region
$repository = terraform -chdir=infra/terraform output -raw ecr_repository_url
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $repository
docker build -t "${repository}:latest" .
docker push "${repository}:latest"
```

Run the one-shot task in the configured subnets:

```powershell
$cluster = terraform -chdir=infra/terraform output -raw ecs_cluster_name
$task = terraform -chdir=infra/terraform output -raw ecs_task_definition_arn
$securityGroup = terraform -chdir=infra/terraform output -raw ecs_security_group_id
$subnets = (terraform -chdir=infra/terraform output -json subnet_ids | ConvertFrom-Json) -join ','
$region = terraform -chdir=infra/terraform output -raw aws_region
$publicIp = terraform -chdir=infra/terraform output -raw assign_public_ip
$publicIpSetting = if ($publicIp -eq "true") { "ENABLED" } else { "DISABLED" }
aws ecs run-task --region $region --cluster $cluster --task-definition $task --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[$subnets],securityGroups=[$securityGroup],assignPublicIp=$publicIpSetting}"
```

The selected subnets need outbound internet access to pull the image and retrieve dependencies/data. Set `assign_public_ip = false` when using private subnets with a NAT gateway or the required VPC endpoints.
