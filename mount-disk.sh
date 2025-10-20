#!/usr/bin/env bash
# Usage: sudo ./safe-mount-disk.sh /dev/sdX /mnt/mountpoint

set -e

DISK="$1"
MOUNT_POINT="$2"

if [ -z "$DISK" ] || [ -z "$MOUNT_POINT" ]; then
  echo "Usage: sudo $0 <device> <mount_point>"
  echo "Example: sudo $0 /dev/sdb /mnt/data"
  exit 1
fi

if [ ! -b "$DISK" ]; then
  echo "Error: $DISK is not a valid block device."
  exit 1
fi

echo "Disk: $DISK"
echo "Mount point: $MOUNT_POINT"

if lsblk "$DISK" | grep -q "${DISK##*/}1"; then
  echo "Partition already exists:"
  lsblk "$DISK"
  read -p "Delete and recreate it? (y/N): " CONFIRM
  if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    (
      echo d
      echo n
      echo p
      echo 1
      echo
      echo
      echo w
    ) | fdisk "$DISK"
  else
    echo "Keeping existing partition."
  fi
else
  echo "Creating new primary partition..."
  (
    echo n
    echo p
    echo 1
    echo
    echo
    echo w
  ) | fdisk "$DISK"
fi

PARTITION="${DISK}1"

if blkid "$PARTITION" >/dev/null 2>&1; then
  echo "Filesystem detected on $PARTITION:"
  blkid "$PARTITION"
  read -p "Reformat it (ERASES DATA)? (y/N): " FORMAT_CONFIRM
  if [[ "$FORMAT_CONFIRM" =~ ^[Yy]$ ]]; then
    mkfs.ext4 -F "$PARTITION"
  else
    echo "Keeping existing filesystem."
  fi
else
  echo "Formatting $PARTITION as ext4..."
  mkfs.ext4 -F "$PARTITION"
fi

if [ ! -d "$MOUNT_POINT" ]; then
  mkdir -p "$MOUNT_POINT"
fi

mount "$PARTITION" "$MOUNT_POINT"
df -h | grep "$MOUNT_POINT" || echo "Mount failed."

UUID=$(blkid -s UUID -o value "$PARTITION")

if grep -q "$UUID" /etc/fstab; then
  echo "Entry already exists in /etc/fstab."
else
  echo "UUID=$UUID   $MOUNT_POINT   ext4   defaults,nofail   0   2" >> /etc/fstab
  echo "Added to /etc/fstab."
fi

umount "$MOUNT_POINT"
mount -a

echo "$PARTITION is now mounted at $MOUNT_POINT and will persist across reboots."
