# 01 · 主机前置（一次性）

目标机画像（生产机实测）：

| 项 | 值 |
|---|---|
| GPU | 2× **NVIDIA CMP 170HX**（GA100 die，**SM80**，物理 8 GB → 解锁 **64 GB** HBM2e） |
| CPU | Intel Xeon E5 v4（22 核，**无 AVX-512**，单 NUMA，超线程已在 BIOS 关闭） |
| 内存 | 157~220 GB DDR3/DDR4-1866（**四通道**，STREAM 实测 50.3 GB/s） |
| 系统盘 | 256 GB NVMe（Ubuntu 20.04.6，ext4） |
| 数据盘 | 1.9 TB NVMe（ext4，标签 `data2t`，挂载 `/media/ll/data`，顺序读 2.1 GB/s、8 路 3.34 GB/s） |
| 驱动 | 610.43.03（开源内核模块，DKMS） |
| 内核 | 5.15.178（自编译，开 `CONFIG_PCI_P2PDMA=y`；**非必需**，本方案不用 GDS） |

> **SM80 很重要**：CMP 170HX 的卡名是解锁固件 override 的，`torch.cuda.get_device_capability()` 实测 `(8,0)`。
> 自己编 CUDA 代码必须按 `sm_80`；按 `sm_86` 编的 kernel 会**静默失败**（launch 返错但 `cudaDeviceSynchronize` 不报）。

---

## 1. NVIDIA 驱动

```bash
sudo apt-get install -y build-essential dkms pkg-config libglvnd-dev
sudo systemctl stop gdm && sudo modprobe -r nouveau
sudo sh NVIDIA-Linux-x86_64-610.43.03.run \
     --no-questions --ui=none --dkms --kernel-module-type=open \
     --no-x-check --no-nouveau-check          # 期望 exit 0
```

**下载坑**：部分机房到 `*.download.nvidia.com` 主站不通，只有 `developer.download.nvidia.com` 可达；
必要时从能出网的机器下载 `.run`（610.43.03 = 461,538,429 字节）再 scp 过去。

## 2. CMP 170HX 显存解锁（8 GB → 64 GB）

```bash
# cmpunlocker（社区工具），驱动装好后执行
cd /opt/cmpunlocker && sudo ./install.sh --profile=8gb --no-passthrough
sudo reboot
```

重启后**必须验证**（判据只有这一条）：

```bash
nvidia-smi --query-gpu=name,memory.total,vbios_version --format=csv
# 期望：memory.total ≈ 65536 MiB，VBIOS 92.00.6D.00.0A
```

> 解锁属于固件级改动，**重刷 VBIOS 后需重做**。ECC / retired pages 全 `N/A`，软件层查不到坏页——
> 想做显存健康度压测，参考仓库外的 `vram_test.cu`（务必带**金丝雀自检**：故意植入损坏→验证器必须抓到，否则测试无效）。

## 3. PCIe Gen2 早钩子（**不做的话 decode 明显吃亏**）

机制与判据见 `docs/08-pitfalls.md#P33`。一句话：Gen2 能力窗口只在 nvidia 模块 probe 的头几秒存在，
必须在窗口内由上游根端口发起 retrain，所以**不能等 systemd**。

```bash
sudo cp scripts/host/98-cmp-gen2-early.rules /etc/udev/rules.d/
sudo install -m755 scripts/host/gen2-early-launch /usr/local/sbin/
sudo install -m755 scripts/host/gen2-early-run   /usr/local/sbin/
sudo udevadm control --reload-rules
# 依赖 cmpunlocker 提供的 gen2-hammer；没有它就自己写一个"写 LNKCTL2=2 + Retrain"的循环
```

验证（**别看 dmesg 的 `OPT_GEN23 FAILED`，那是所有 170HX 必现的假阴性**）：

```bash
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max --format=csv
# 期望：2, 2
nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader
# 期望：16（被主板端口接线限到 x8 也能跑，但会掉带宽）
```

## 4. 内核参数

```bash
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-vllm-overcommit.conf
sudo sysctl --system
```

**为什么必需**：BF16 路线要 mmap 单个 102.4 GB 的 safetensors 分片，默认启发式（0）会 `ENOMEM`。
INT8 路线虽然单文件只有 47.7 GB，也建议保留（NVFP4 时代最大单文件 5.1 GB 从未触发，所以这不是"可选洁癖"）。

## 5. 数据盘读预读（read_ahead_kb = 128）

```bash
sudo cp scripts/host/99-nvme-readahead.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --action=change --settle
```

规则**按 `ENV{ID_SERIAL}` 匹配**（本机重启后数据盘与系统盘盘符会互换，写死 `nvmeXn1` 会打到系统盘）：

```
ACTION=="add|change", SUBSYSTEM=="block", ENV{DEVTYPE}=="disk",
ENV{ID_SERIAL}=="<你的数据盘序列号>", ATTR{queue/read_ahead_kb}="128"
```

两个坑：①编号必须 **99-**（`ID_SERIAL` 由 `60-persistent-storage.rules` 导入）；
②键必须写 **`ENV{DEVTYPE}`**，写 `DEVTYPE` 会 `Invalid key` 且静默不生效。

实测拐点与"为什么不能设 0 / 不需要设 2048"见 `docs/08#P28`。

## 6. 功耗墙与持久模式

```bash
sudo install -m755 scripts/host/gpu-power-limit.sh /usr/local/bin/
sudo cp systemd-units/gpu-power-limit.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-power-limit.service
```

三个细节（都踩过）：
1. **必须先 `nvidia-smi -pm 1`**，否则 `-pl` 重启即失效。
2. `WantedBy=` 要挂 **`basic.target`**，不能挂 `multi-user.target`——这类机器开机常卡 `plymouth-quit-wait`，
   `multi-user.target` 永不到达，服务就静默不跑。
3. **所有 `nvidia-smi` 调用都要套 `timeout`**：遇到坏卡（GSP 起不来）时 `nvidia-smi -L` 会无限阻塞，
   systemd 默认 90 s 启动超时一掐，"开机自动限功率"就悄悄失效了（journal 里表现为只有 Starting 没有 Started）。

CMP 170HX 可调范围 100~300 W。生产值 **210 W**（历史 200→250→210）。
注意 PP1 rank（草稿 + lm_head + 采样）才是撞墙的那张卡，见 `docs/08#P30`。

## 7. swap 建议关闭

```bash
sudo swapoff -a            # 并从 /etc/fstab 注释掉对应行
```

理由：PLE 表 48~95 GB 必须常驻，一旦被换出，服务期缺页会直接打穿 decode；
若 swapfile 与 PLE 表同盘，还会抢 I/O。

## 8. chroot 挂载服务

本方案不用 docker，用 chroot。`/proc /sys /dev` 与 NVIDIA 用户态库需要在开机时挂好，
否则 chroot 内 torch 报 `Unable to open /dev/urandom`：

```bash
sudo cp scripts/setup-chroot.sh /home/ll/deploy/
sudo cp systemd-units/chroot-setup.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now chroot-setup.service
```

## 9.（可选）给控制台账号配 NOPASSWD

仓库脚本默认走 `sudo -n`（NOPASSWD）或环境变量 `SUDO_PASS`。推荐前者：

```bash
# /etc/sudoers.d/dsh-deploy   （改 sudoers 一律先 visudo -cf 再落盘）
ll ALL=(root) NOPASSWD: /usr/sbin/chroot, /usr/bin/kill, /usr/bin/pkill, /usr/sbin/nvidia-smi
```

> **坑**：sudoers 里命令参数含冒号（如 `intel-rapl:0`）必须转义成 `\:`，否则**整个 sudo 失效**。
> 另：`sudo -n cmd` 成功不等于 NOPASSWD 生效（可能是时间戳缓存假阳性），要换一个全新会话复验。

## 10. 验收

```bash
bash scripts/verify-deployment.sh --offline   # A 段全绿即可继续
```
