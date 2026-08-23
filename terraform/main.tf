terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Binance's primary matching engine is located in AWS Tokyo (ap-northeast-1)
# Deploying our HFT node here achieves sub-millisecond physical latency.
provider "aws" {
  region = "ap-northeast-1"
}

# ─── VPC & Networking ───────────────────────────────────────────
resource "aws_vpc" "hft_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "HFT-Colo-VPC"
  }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.hft_vpc.id
}

resource "aws_subnet" "hft_subnet" {
  vpc_id                  = aws_vpc.hft_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "ap-northeast-1a" # Nearest AZ to matching engine
}

resource "aws_route_table" "rtb" {
  vpc_id = aws_vpc.hft_vpc.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
}

resource "aws_route_table_association" "rta" {
  subnet_id      = aws_subnet.hft_subnet.id
  route_table_id = aws_route_table.rtb.id
}

# ─── Security Group ─────────────────────────────────────────────
resource "aws_security_group" "hft_sg" {
  name        = "hft-colo-sg"
  description = "Strict firewall for HFT node"
  vpc_id      = aws_vpc.hft_vpc.id

  # Allow inbound SSH only from our specific IP (Placeholder)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # TODO: Restrict to management IP
  }

  # Allow all outbound traffic to reach Binance APIs
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ─── EC2 Instance (c6i.metal for DPDK) ──────────────────────────
# Using a bare-metal instance (c6i.metal) is required to bypass the
# AWS Nitro hypervisor and bind the NIC directly to DPDK.
resource "aws_instance" "hft_node" {
  ami           = "ami-0a0b7b240264a48d7" # Ubuntu 22.04 LTS (ap-northeast-1)
  instance_type = "c6i.metal"             # Bare metal for Kernel Bypass
  subnet_id     = aws_subnet.hft_subnet.id
  vpc_security_group_ids = [aws_security_group.hft_sg.id]

  # Required for DPDK/ENA (Elastic Network Adapter) high-performance networking
  ebs_optimized = true

  root_block_device {
    volume_size = 50
    volume_type = "gp3"
    iops        = 3000
  }

  tags = {
    Name = "HFT-Colo-Tier1"
    Env  = "Production"
  }

  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y build-essential cmake git dpdk dpdk-dev libnuma-dev
              
              # Allocate HugePages for DPDK memory pools
              echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
              mkdir -p /mnt/huge
              mount -t hugetlbfs nodev /mnt/huge
              
              echo "[Colo] Instance bootstrapped for HFT DPDK!" > /var/log/colo-init.log
              EOF
}

output "hft_node_public_ip" {
  value = aws_instance.hft_node.public_ip
}
