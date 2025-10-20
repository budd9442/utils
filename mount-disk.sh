#!/usr/bin/env bash

# ==========================================
# Usage: sudo bash mount-disk.sh /dev/sdX /mnt/mountpoint
# ==========================================

DISK="$1"
MOUNT_POINT="$2"

if [ -z "$DISK" ] || [ -z "$MOUNT_POINT" ]; then
  echo "Usage: sudo $0 <device> <mount_point>"
  echo "Example: sudo $0 /dev/sdb /mnt/datadisk1"
  exit 1
fi

if [ ! -b "$DISK" ]; then
  echo "❌ $DISK is not a valid block device."
  exit 1
fi

echo "📀 Target disk: $DISK"
echo "📁 Mount point: $MOUNT_POINT"

if lsblk "$DISK" | grep -q "${DISK##*/}1"; then
  echo "⚠️  Partition already exists. Skipping fdisk..."
else
  echo "🪛 Creating a single primary partition..."
  (
    echo n     # new partition
    echo p     # primary
    echo 1     # partition number
    echo       # default - first sector
    echo       # default - last sector (use all space)
    echo w     # write changes
  ) | fdisk "$DISK"
fi

PARTITION="${DISK}1"


if blkid "$PARTITION" >/dev/null 2>&1; then
  echo "⚠️  Filesystem already exists. Skipping mkfs..."
else
  echo "🧰 Formatting $PARTITION as ext4..."
  mkfs.ext4 "$PARTITION"
fi


if [ ! -d "$MOUNT_POINT" ]; then
  echo "📁 Creating mount point: $MOUNT_POINT"
  mkdir -p "$MOUNT_POINT"
fi


echo "📂 Mounting $PARTITION..."
mount "$PARTITION" "$MOUNT_POINT"

echo "✅ Mounted! Checking:"
df -h | grep "$MOUNT_POINT"


UUID=$(blkid -s UUID -o value "$PARTITION")

if grep -q "$UUID" /etc/fstab; then
  echo "✅ UUID already exists in /etc/fstab. Skipping..."
else
  echo "📝 Adding to /etc/fstab..."
  echo "UUID=$UUID   $MOUNT_POINT   ext4   defaults   0   2" >> /etc/fstab
fi


echo "🔁 Testing mount..."
umount "$MOUNT_POINT"
mount -a

echo "🎉 Done!"
echo "$PARTITION is now mounted at $MOUNT_POINT and will persist across reboots."
