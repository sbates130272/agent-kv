#!/usr/bin/env bash
# Set up a dm-delay device over the KV NVMe partition for nvme-sweep experiments.
# Must be run as root (or via sudo).
#
# What this does:
#   1. Creates a dm-delay device "kv-delay" that wraps $KV_DISK.
#   2. Formats and mounts it at $MOUNT (or uses the existing filesystem).
#   3. Adds a sudoers NOPASSWD rule so the agent-kv user can change the delay
#      at runtime without a password prompt.
#
# dm-delay table format:  0 <sectors> delay <dev> 0 <read_ms> <dev> 0 <write_ms>
# read_ms=0 at setup time; agent-kv nvme-sweep changes it live via dmsetup.
#
# To tear down:
#   umount /mnt/kv-nvme && dmsetup remove kv-delay

set -euo pipefail

KV_DISK="${KV_DISK:-/dev/disk/by-id/nvme-WD_BLACK_SN850X_2000GB_24520J802767}"
DEVICE_NAME="${DEVICE_NAME:-kv-delay}"
MOUNT="${MOUNT:-/mnt/kv-nvme}"
USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root (sudo $0)"
[[ -e "$KV_DISK" ]] || die "KV_DISK $KV_DISK not found — check /dev/disk/by-id/"

# dmsetup requires the canonical /dev path in the table, not a symlink.
REAL_DEV=$(realpath "$KV_DISK")
SECTORS=$(blockdev --getsz "$REAL_DEV")
echo "by-id:   $KV_DISK"
echo "Device:  $REAL_DEV ($SECTORS sectors)"
echo "DM name: $DEVICE_NAME"
echo "Mount:   $MOUNT"
echo ""

# 1. Create dm-delay device (read_delay=0 ms, write_delay=0 ms)
if dmsetup info "$DEVICE_NAME" &>/dev/null; then
    echo "dm-delay device /dev/mapper/$DEVICE_NAME already exists — skipping create"
else
    echo "0 $SECTORS delay $REAL_DEV 0 0 $REAL_DEV 0 0" | dmsetup create "$DEVICE_NAME"
    echo "Created /dev/mapper/$DEVICE_NAME"
fi

DM_DEV="/dev/mapper/$DEVICE_NAME"

# 2. Create filesystem if needed, then mount
if ! blkid "$DM_DEV" &>/dev/null; then
    echo "Formatting $DM_DEV with ext4..."
    mkfs.ext4 -F "$DM_DEV"
fi

mkdir -p "$MOUNT"
if mountpoint -q "$MOUNT"; then
    echo "$MOUNT already mounted"
else
    mount "$DM_DEV" "$MOUNT"
    echo "Mounted $DM_DEV at $MOUNT"
fi

mkdir -p "$MOUNT/agent-kv-l2"
chown -R "$USER":"$USER" "$MOUNT/agent-kv-l2" 2>/dev/null || true

# 3. Add sudoers NOPASSWD for dmsetup (needed for runtime delay changes)
SUDOERS_FILE="/etc/sudoers.d/agent-kv-dmsetup"
if [[ -f "$SUDOERS_FILE" ]]; then
    echo "Sudoers rule $SUDOERS_FILE already exists"
else
    cat > "$SUDOERS_FILE" <<EOF
# Allow $USER to adjust dm-delay device without a password prompt.
# Required by: agent-kv nvme-sweep --dm-device $DEVICE_NAME
$USER ALL=(root) NOPASSWD: /sbin/dmsetup table $DEVICE_NAME, \
                            /sbin/dmsetup load $DEVICE_NAME, \
                            /sbin/dmsetup resume $DEVICE_NAME
EOF
    chmod 440 "$SUDOERS_FILE"
    echo "Created $SUDOERS_FILE"
fi

echo ""
echo "Setup complete.  Run the sweep with:"
echo ""
echo "  KV_TIER=file KV_STORE=$MOUNT/agent-kv-l2 \\"
echo "    docker compose -f docker/docker-compose.yml --profile vllm-and-lmcache up -d"
echo ""
echo "  agent-kv nvme-sweep \\"
echo "    --dm-device $DEVICE_NAME \\"
echo "    --delays 0,10,50,100,200 \\"
echo "    --fill-fractions 0.25,0.50,0.75,1.0"
echo ""
echo "To change read delay manually (for testing):"
echo "  sudo dmsetup table $DEVICE_NAME"
echo "  # edit field 5 (read_ms), then:"
echo "  echo '0 $SECTORS delay $REAL_DEV 0 <read_ms> $REAL_DEV 0 0' \\"
echo "    | sudo dmsetup load $DEVICE_NAME && sudo dmsetup resume $DEVICE_NAME"
echo ""
echo "To tear down:"
echo "  sudo umount $MOUNT && sudo dmsetup remove $DEVICE_NAME"
