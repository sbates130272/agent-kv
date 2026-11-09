# docker buildx bake -f docker/docker-bake.hcl [--push] [target]
#
# Build both images (default):
#   docker buildx bake -f docker/docker-bake.hcl
#
# Build and push to registry:
#   PUSH=1 docker buildx bake -f docker/docker-bake.hcl
#
# Override wheel index (e.g. local mirror):
#   WHEEL_INDEX=http://192.168.1.1:8081/index/gfx1250/ docker buildx bake -f docker/docker-bake.hcl

# ─── Registry ─────────────────────────────────────────────────────────────────

variable "REGISTRY" {
  default = "snoc-strix.fold-leaffish.ts.net:8080"
}

variable "REGISTRY_INSECURE" {
  default = "true"
}

variable "IMAGE_PREFIX" {
  default = "agent-kv"
}

variable "IMAGE_TAG" {
  default = "latest"
}

variable "PUSH" {
  default = "0"
}

# ─── GPU / ROCm ───────────────────────────────────────────────────────────────

variable "GFX_ARCH" {
  default = "gfx1250"
}

variable "ROCM_VERSION" {
  default = "10.1.0-full"
}

# ─── Wheel source ─────────────────────────────────────────────────────────────

# PEP 503 index published by sbates130272/rocm-bake for gfx1250 wheels.
# Contains pre-built torch, triton, vllm, lmcache, cupy, and their deps.
variable "WHEEL_INDEX" {
  default = "https://sbates130272.github.io/rocm-bake/data/wheels/gfx1250/"
}

# ─── Software versions ────────────────────────────────────────────────────────

variable "PYTORCH_VERSION" {
  default = "2.15.0"
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
# Override with the rocm-bake rocm-runtime image for a smaller local build:
#   ROCM_BASE_IMAGE=snoc-strix.fold-leaffish.ts.net:8080/rocm-pytorch/rocm-runtime:latest
variable "ROCM_BASE_IMAGE" {
  default = "rocm/dev-ubuntu-26.04:10.1.0-full"
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

function "img" {
  params = [name]
  result = "${REGISTRY}/${IMAGE_PREFIX}/${name}:${IMAGE_TAG}"
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
  tags   = [img("lmcache")]
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
  tags   = [img("vllm")]
  output = [output_type()]
}
