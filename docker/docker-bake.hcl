# docker buildx bake -f docker/docker-bake.hcl [--push] [target]
#
# Build both images (default):
#   docker buildx bake -f docker/docker-bake.hcl
#
# Build and push to registry:
#   PUSH=1 docker buildx bake -f docker/docker-bake.hcl
#
# Override wheel index (e.g. local mirror):
#   WHEEL_INDEX=http://192.168.1.1:8081/index/gfx1201/ docker buildx bake -f docker/docker-bake.hcl

# ─── Registry ─────────────────────────────────────────────────────────────────

variable "REGISTRY" {
  default = ""
}

variable "REGISTRY_INSECURE" {
  default = "true"
}

variable "IMAGE_PREFIX" {
  default = "agent-kv"
}


variable "PUSH" {
  default = "0"
}

# ─── GPU / ROCm ───────────────────────────────────────────────────────────────

variable "GFX_ARCH" {
  default = "gfx1201"
}

variable "ROCM_VERSION" {
  default = "10.1.0-full"
}

# ─── Wheel source ─────────────────────────────────────────────────────────────

# PEP 503 index published by sbates130272/rocm-bake for gfx1201 wheels.
# Contains pre-built torch, triton, vllm, lmcache, cupy, and their deps.
variable "WHEEL_INDEX" {
  default = "https://sbates130272.github.io/rocm-bake/data/wheels/${GFX_ARCH}/"
}

# ─── Software versions ────────────────────────────────────────────────────────

variable "PYTORCH_VERSION" {
  default = "2.15.0a0"
}

variable "VLLM_VERSION" {
  default = "0.33.0"
}

variable "LMCACHE_VERSION" {
  default = "0.5.6"
}

# ─── Base image ───────────────────────────────────────────────────────────────

# AMD public ROCm base image (Ubuntu 26.04, ROCm 10.1.0-full) from Docker Hub.
# Ubuntu 26.04 ships Python 3.14 natively — same base as rocm-bake uses.
variable "ROCM_BASE_IMAGE" {
  default = "rocm/dev-ubuntu-26.04:10.1.0-full"
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

function "img" {
  params = [name, tag]
  result = REGISTRY != "" ? "${REGISTRY}/${IMAGE_PREFIX}/${name}:${tag}" : "${IMAGE_PREFIX}/${name}:${tag}"
}

function "output_type" {
  params = []
  result = PUSH == "1" ? "type=registry,insecure=${REGISTRY_INSECURE}" : "type=docker"
}

# ─── Groups ───────────────────────────────────────────────────────────────────

group "default" {
  targets = ["lmcache", "vllm"]
}

# ─── Targets ──────────────────────────────────────────────────────────────────

target "lmcache" {
  context    = "."
  dockerfile = "docker/Dockerfile.lmcache"
  args = {
    ROCM_BASE_IMAGE = ROCM_BASE_IMAGE
    WHEEL_INDEX     = WHEEL_INDEX
    PYTORCH_VERSION = PYTORCH_VERSION
    LMCACHE_VERSION = LMCACHE_VERSION
  }
  tags   = [img("lmcache", LMCACHE_VERSION)]
  output = [output_type()]
}

target "vllm" {
  context    = "."
  dockerfile = "docker/Dockerfile.vllm"
  args = {
    ROCM_BASE_IMAGE = ROCM_BASE_IMAGE
    WHEEL_INDEX     = WHEEL_INDEX
    PYTORCH_VERSION = PYTORCH_VERSION
    VLLM_VERSION    = VLLM_VERSION
    LMCACHE_VERSION = LMCACHE_VERSION
  }
  tags   = [img("vllm", VLLM_VERSION)]
  output = [output_type()]
}
