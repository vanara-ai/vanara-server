"""
Cloud provider taxonomy for deterministic detection.
No LLM calls - pure regex/keyword matching with comprehensive coverage.
"""

import hashlib
import re

CLOUD_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "AWS": {
        "explicit_keywords": ["aws", "amazon web services", "amazon cloud"],
        "services": [
            # Compute
            "ec2",
            "lambda",
            "ecs",
            "eks",
            "fargate",
            "elastic beanstalk",
            "lightsail",
            "batch",
            # Storage
            "s3",
            "ebs",
            "efs",
            "fsx",
            "storage gateway",
            "snowball",
            # Database
            "rds",
            "aurora",
            "dynamodb",
            "elasticache",
            "redshift",
            "neptune",
            "documentdb",
            "keyspaces",
            "timestream",
            "qldb",
            # Analytics
            "athena",
            "emr",
            "kinesis",
            "quicksight",
            "data pipeline",
            "glue",
            "lake formation",
            "msk",
            # ML/AI
            "sagemaker",
            "comprehend",
            "rekognition",
            "polly",
            "lex",
            "transcribe",
            "translate",
            "personalize",
            "forecast",
            "bedrock",
            # Networking
            "vpc",
            "cloudfront",
            "route 53",
            "route53",
            "api gateway",
            "direct connect",
            "global accelerator",
            "elb",
            "alb",
            "nlb",
            # Security
            "iam",
            "cognito",
            "secrets manager",
            "kms",
            "waf",
            "shield",
            "guardduty",
            "inspector",
            "macie",
            # DevOps
            "cloudformation",
            "cloudwatch",
            "cloudtrail",
            "config",
            "systems manager",
            "codepipeline",
            "codebuild",
            "codedeploy",
            "codecommit",
            # Integration
            "sns",
            "sqs",
            "eventbridge",
            "step functions",
            "appflow",
            "mwaa",
        ],
    },
    "GCP": {
        "explicit_keywords": ["gcp", "google cloud", "google cloud platform"],
        "services": [
            # Compute
            "compute engine",
            "gce",
            "cloud run",
            "cloud functions",
            "gke",
            "app engine",
            "anthos",
            # Storage
            "cloud storage",
            "gcs",
            "persistent disk",
            "filestore",
            # Database
            "cloud sql",
            "cloud spanner",
            "bigtable",
            "firestore",
            "memorystore",
            "cloud datastore",
            # Analytics
            "bigquery",
            "dataflow",
            "dataproc",
            "pub/sub",
            "pubsub",
            "data fusion",
            "composer",
            "looker",
            "dataform",
            # ML/AI
            "vertex ai",
            "automl",
            "vision ai",
            "natural language ai",
            "speech-to-text",
            "text-to-speech",
            "dialogflow",
            "document ai",
            # Networking
            "cloud cdn",
            "cloud dns",
            "cloud load balancing",
            "cloud armor",
            "cloud nat",
            "cloud vpn",
            "cloud interconnect",
            # Security
            "cloud iam",
            "secret manager",
            "cloud kms",
            "security command center",
            "beyondcorp",
            # DevOps
            "cloud build",
            "cloud deploy",
            "artifact registry",
            "container registry",
            "cloud monitoring",
            "cloud logging",
            "cloud trace",
        ],
    },
    "Azure": {
        "explicit_keywords": ["azure", "microsoft azure", "azure cloud"],
        "services": [
            # Compute
            "azure vm",
            "virtual machines",
            "azure functions",
            "aks",
            "azure kubernetes",
            "app service",
            "container instances",
            "azure batch",
            # Storage
            "blob storage",
            "azure blob",
            "azure files",
            "azure disk",
            "data lake storage",
            "adls",
            "azure data lake",
            # Database
            "azure sql",
            "cosmos db",
            "cosmosdb",
            "azure database",
            "azure cache",
            "azure synapse",
            "synapse analytics",
            "azure postgresql",
            "azure mysql",
            # Analytics
            "azure databricks",
            "data factory",
            "adf",
            "azure data factory",
            "stream analytics",
            "event hubs",
            "azure analysis services",
            "power bi",
            "azure purview",
            # ML/AI
            "azure ml",
            "azure machine learning",
            "cognitive services",
            "azure openai",
            "bot service",
            "form recognizer",
            "azure ai",
            # Networking
            "azure cdn",
            "azure dns",
            "application gateway",
            "azure firewall",
            "expressroute",
            "azure front door",
            "traffic manager",
            "azure load balancer",
            # Security
            "azure ad",
            "azure active directory",
            "entra",
            "key vault",
            "azure sentinel",
            "defender for cloud",
            "azure policy",
            # DevOps
            "azure devops",
            "azure pipelines",
            "azure repos",
            "azure monitor",
            "log analytics",
            "application insights",
            "azure resource manager",
            "arm templates",
            "bicep",
        ],
    },
    "On-Premise": {
        "explicit_keywords": [
            "on-premise",
            "on-prem",
            "on premise",
            "data center",
            "datacenter",
            "self-hosted",
            "bare metal",
        ],
        "services": [
            # Big Data
            "hadoop",
            "hdfs",
            "mapreduce",
            "hive",
            "pig",
            "hbase",
            "zookeeper",
            "oozie",
            "sqoop",
            "flume",
            "spark",
            "pyspark",
            "spark streaming",
            "spark sql",
            "kafka",
            "confluent",
            "kafka streams",
            "ksql",
            "flink",
            "storm",
            "samza",
            # Databases
            "oracle",
            "oracle db",
            "pl/sql",
            "oracle rac",
            "sql server",
            "mssql",
            "t-sql",
            "ssis",
            "ssrs",
            "ssas",
            "postgresql",
            "postgres",
            "mysql",
            "mariadb",
            "mongodb",
            "cassandra",
            "couchbase",
            "redis",
            "memcached",
            "teradata",
            "netezza",
            "greenplum",
            "vertica",
            # ETL
            "informatica",
            "talend",
            "datastage",
            "pentaho",
            "nifi",
            "airflow",
            # BI
            "tableau",
            "qlik",
            "microstrategy",
            "cognos",
            "business objects",
            # Infrastructure
            "vmware",
            "esxi",
            "vcenter",
            "hyper-v",
            "openstack",
            "kubernetes",
            "k8s",
            "docker",
            "podman",
        ],
    },
}

# Services that exist across multiple clouds (cloud-agnostic)
CLOUD_AGNOSTIC_TECHNOLOGIES: set[str] = {
    # Languages
    "python",
    "java",
    "scala",
    "sql",
    "javascript",
    "typescript",
    "go",
    "golang",
    "rust",
    "c++",
    "c#",
    ".net",
    # Frameworks
    "spark",
    "pyspark",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "django",
    "flask",
    "fastapi",
    "spring",
    "spring boot",
    "node.js",
    "react",
    "angular",
    "vue",
    # Tools
    "git",
    "github",
    "gitlab",
    "bitbucket",
    "jenkins",
    "terraform",
    "ansible",
    "puppet",
    "chef",
    "docker",
    "kubernetes",
    "k8s",
    "helm",
    "prometheus",
    "grafana",
    "elk",
    "elasticsearch",
    "kibana",
    "logstash",
    "airflow",
    "dbt",
    "great expectations",
    "mlflow",
    "kubeflow",
    # Concepts
    "etl",
    "elt",
    "data pipeline",
    "data warehouse",
    "data lake",
    "data mesh",
    "data modeling",
    "ci/cd",
    "devops",
    "mlops",
    "dataops",
    "agile",
    "scrum",
    "microservices",
    "rest api",
    "graphql",
}


def _create_word_boundary_pattern(term: str) -> re.Pattern:
    """Create regex pattern with word boundaries for accurate matching."""
    # Escape special regex characters
    escaped = re.escape(term)
    # Add word boundaries
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def detect_cloud_providers(text: str) -> list[str]:
    """
    Detect cloud providers using word-boundary matching.
    Returns list of detected providers, or ["Cloud-Agnostic"] if none found.

    Uses two-tier detection:
    1. Explicit keywords (high confidence)
    2. Service names (requires 2+ matches for confidence)
    """
    if not text:
        return ["Cloud-Agnostic"]

    text_lower = text.lower()
    detected: list[str] = []

    for provider, config in CLOUD_TAXONOMY.items():
        # Tier 1: Explicit keywords (single match sufficient)
        for keyword in config["explicit_keywords"]:
            pattern = _create_word_boundary_pattern(keyword)
            if pattern.search(text_lower):
                if provider not in detected:
                    detected.append(provider)
                break

        if provider in detected:
            continue

        # Tier 2: Service names (need 2+ matches)
        service_matches = 0
        for svc in config["services"]:
            pattern = _create_word_boundary_pattern(svc)
            if pattern.search(text_lower):
                service_matches += 1

        if service_matches >= 2:
            detected.append(provider)

    return detected if detected else ["Cloud-Agnostic"]


def detect_role_contamination(
    original_role: dict, updated_role: dict
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """
    Detect cross-cloud contamination at the ROLE level by comparing original vs updated.

    This is the correct way to detect contamination:
    1. Detect clouds in original role (across ALL responsibilities)
    2. Detect clouds in updated role (across ALL responsibilities)
    3. Flag contamination if updated has clouds NOT in original

    Args:
        original_role: The original role dict with responsibilities
        updated_role: The updated role dict with responsibilities

    Returns:
        Tuple of (contaminations, original_clouds, updated_clouds)
        - contaminations: List of contaminating technologies
        - original_clouds: Clouds detected in original role
        - updated_clouds: Clouds detected in updated role
    """
    # Combine all responsibilities for original role
    original_text = " ".join(original_role.get("responsibilities", []))
    original_clouds = detect_cloud_providers(original_text)

    # Combine all responsibilities for updated role
    updated_text = " ".join(updated_role.get("responsibilities", []))
    updated_clouds = detect_cloud_providers(updated_text)

    # Convert to sets for proper comparison (removes duplicates)
    original_clouds_set = set(original_clouds) - {"Cloud-Agnostic"}
    updated_clouds_set = set(updated_clouds) - {"Cloud-Agnostic"}

    # Find clouds that were added (not in original)
    added_clouds = list(updated_clouds_set - original_clouds_set)

    # If no new clouds added, no contamination
    if not added_clouds:
        return [], original_clouds, updated_clouds

    # Find specific technologies from the added clouds
    contaminations: list[dict[str, str]] = []
    updated_text_lower = updated_text.lower()

    for added_cloud in added_clouds:
        if added_cloud in CLOUD_TAXONOMY:
            config = CLOUD_TAXONOMY[added_cloud]

            # Check for explicit keywords
            for keyword in config["explicit_keywords"]:
                pattern = _create_word_boundary_pattern(keyword)
                if pattern.search(updated_text_lower):
                    contaminations.append(
                        {
                            "technology": keyword,
                            "cloud": added_cloud,
                            "type": "explicit_keyword",
                            "reason": f"{added_cloud} not in original role",
                        }
                    )

            # Check for services
            for service in config["services"]:
                pattern = _create_word_boundary_pattern(service)
                if pattern.search(updated_text_lower):
                    # Context check for ambiguous terms
                    if service in ["glue", "power bi", "databricks"]:
                        if service == "glue" and added_cloud == "AWS" and "aws" not in updated_text_lower:
                            continue
                        if service == "power bi" and added_cloud == "Azure" and "azure" not in updated_text_lower:
                            continue
                        if service == "databricks" and added_cloud == "Azure" and "azure" not in updated_text_lower:
                            continue

                    contaminations.append(
                        {
                            "technology": service,
                            "cloud": added_cloud,
                            "type": "service",
                            "reason": f"{added_cloud} not in original role",
                        }
                    )

    return contaminations, original_clouds, updated_clouds


def generate_role_id(role: dict) -> str:
    """
    Generate stable, unique identifier for a role.
    Uses hash of company + title + start_date + end_date.
    """
    company = (role.get("company") or "").strip()
    title = (role.get("title") or "").strip()
    start_date = (role.get("start_date") or "").strip()
    end_date = (role.get("end_date") or "").strip()

    # Create stable string
    role_string = f"{company}|{title}|{start_date}|{end_date}"

    # Generate hash
    role_hash = hashlib.md5(role_string.encode()).hexdigest()[:12]

    return role_hash
