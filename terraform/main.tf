# ==========================================================
# Terraform configuration for churn-mlops infrastructure
# ==========================================================

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

# Configure the Google Cloud provider
provider "google" {
  project = "project-df3432dc-e4a6-42ee-aba"
  region  = "asia-south1"
}

# The GCS bucket that stores DVC-tracked data + model
resource "google_storage_bucket" "dvc_remote" {
  name          = "churn-mlops-dvc-215271667398"
  location      = "asia-south1"
  force_destroy = false

  uniform_bucket_level_access = true
}

# The Cloud Run service that serves the churn prediction API
resource "google_cloud_run_v2_service" "churn_api" {
  name     = "churn-api"
  location = "asia-south1"

  deletion_protection = false

  template {
    containers {
      image = "asia-south1-docker.pkg.dev/project-df3432dc-e4a6-42ee-aba/cloud-run-source-deploy/churn-api@sha256:db351d445fd8ef4a2a22ec3870c8d2d768942dfed8c73c6dc673bde338a1b217"

      resources {
        limits = {
          cpu = "1000m"
          memory = "1Gi"
        }
      }

      ports {
        container_port = 8000
      }
    }
  }
}

# Make the Cloud Run service publicly accessible
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  name     = google_cloud_run_v2_service.churn_api.name
  location = google_cloud_run_v2_service.churn_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}
