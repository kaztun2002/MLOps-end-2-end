variable "aws_region" {
  description = "AWS region for the pipeline infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Lowercase prefix used to name AWS resources."
  type        = string
  default     = "mlops-end-2-end"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,18}[a-z0-9]$", var.project_name))
    error_message = "project_name must be lowercase, start with a letter, and contain only letters, numbers, and hyphens."
  }
}

variable "vpc_id" {
  description = "VPC in which ECS tasks will run."
  type        = string
}

variable "subnet_ids" {
  description = "Subnets for one-shot Fargate tasks; public subnets are required when assign_public_ip is true."
  type        = list(string)

  validation {
    condition     = length(var.subnet_ids) > 0
    error_message = "Provide at least one subnet ID."
  }
}

variable "assign_public_ip" {
  description = "Assign a public IP for outbound access from public subnets."
  type        = bool
  default     = true
}

variable "image_tag" {
  description = "Container image tag ECS should run after it has been pushed to ECR."
  type        = string
  default     = "latest"
}

variable "task_cpu" {
  description = "Fargate task CPU units."
  type        = number
  default     = 2048
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 8192
}

variable "log_retention_days" {
  description = "CloudWatch log retention period."
  type        = number
  default     = 30
}

variable "artifact_retention_days" {
  description = "Retention period for uploaded pipeline artifacts."
  type        = number
  default     = 90
}