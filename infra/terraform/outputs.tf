output "ecr_repository_url" {
  description = "Push the Docker image to this ECR repository."
  value       = aws_ecr_repository.pipeline.repository_url
}

output "ecs_cluster_name" {
  description = "Cluster for one-shot pipeline tasks."
  value       = aws_ecs_cluster.pipeline.name
}

output "ecs_task_definition_arn" {
  description = "Task definition ARN to pass to aws ecs run-task."
  value       = aws_ecs_task_definition.pipeline.arn
}

output "ecs_security_group_id" {
  description = "Outbound-only task security group."
  value       = aws_security_group.pipeline_task.id
}

output "artifact_bucket_name" {
  description = "S3 bucket where each pipeline run uploads generated artifacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "aws_region" {
  description = "AWS region containing the pipeline resources."
  value       = var.aws_region
}

output "subnet_ids" {
  description = "Subnets configured for pipeline task networking."
  value       = var.subnet_ids
}

output "assign_public_ip" {
  description = "Whether ECS tasks should receive a public IP."
  value       = var.assign_public_ip
}