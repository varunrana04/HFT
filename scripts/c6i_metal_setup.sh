#!/usr/bin/env bash
# c6i_metal_setup.sh
# AWS c6i.metal Ubuntu Benchmarking Setup Script
# MUST BE RUN AS ROOT

set -e

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

echo "=== AWS c6i.metal HFT Tuning Script ==="

# 1. Update and install toolchains
echo "[1/4] Installing C++ toolchains and DPDK dependencies..."
apt-get update -y
apt-get install -y build-essential cmake linux-headers-$(uname -r) \
    python3-pyelftools libnuma-dev pkg-config ninja-build gcc-11 g++-11

# Set GCC 11 as default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# 2. Kernel parameters for isolation
echo "[2/4] Configuring GRUB for CPU isolation..."
# Assuming 72 cores (36 physical) on c6i.metal. Isolating 2-35 (physical cores)
ISOL_CORES="2-35"
KERNEL_PARAMS="isolcpus=$ISOL_CORES nohz_full=$ISOL_CORES rcu_nocbs=$ISOL_CORES mce=ignore_ce intel_idle.max_cstate=0 processor.max_cstate=0"

# Check if already added
if ! grep -q "$KERNEL_PARAMS" /etc/default/grub; then
    sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*/& $KERNEL_PARAMS/" /etc/default/grub
    update-grub
    echo "GRUB updated. Reboot required for isolation to take effect."
else
    echo "GRUB already configured with isolation parameters."
fi

# 3. CPU frequency scaling
echo "[3/4] Setting performance scaling governor..."
apt-get install -y linux-tools-common linux-tools-generic linux-tools-$(uname -r)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done
# Disable turbo boost for deterministic TSC
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo "1" > /sys/devices/system/cpu/intel_pstate/no_turbo
    echo "Turbo boost disabled."
fi

# 4. Configure Hugepages for memory-mapped operations (DPDK)
echo "[4/4] Allocating 1GB Hugepages..."
echo 4 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages

echo "=== Setup Complete ==="
echo "Please REBOOT the instance to apply kernel isolation."
echo "After reboot, run benchmarks with:"
echo "taskset -c 2 ./build/hft_bench"
