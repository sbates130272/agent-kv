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

# pytorch image from rocm-bake: ROCm 10.1.0 + Python 3.14 + PyTorch 2.15.0.
# The gfx1250 tag was published as gfx1201 internally — override PYTORCH_IMAGE
# if the tag differs on your registry.
variable "PYTORCH_IMAGE" {
  default = "snoc-strix.fold-leaffish.ts.net:8080/rocm-pytorch/pytorch:latest"
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
    PYTORCH_IMAGE   = PYTORCH_IMAGE
    WHEEL_INDEX     = WHEEL_INDEX
    LMCACHE_VERSION = LMCACHE_VERSION
  }
  tags   = [img("lmcache")]
  output = [output_type()]
}

target "vllm" {
  context    = "."
  dockerfile = "docker/Dockerfile.vllm"
  args = {
    PYTORCH_IMAGE   = PYTORCH_IMAGE
    WHEEL_INDEX     = WHEEL_INDEX
    VLLM_VERSION    = VLLM_VERSION
    LMCACHE_VERSION = LMCACHE_VERSION
  }
  tags   = [img("vllm")]
  output = [output_type()]
}
