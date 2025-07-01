# In-memory Training on Analog Devices

This repository contains the complete implementation used in our **NeurIPS 2025** submission on analog in-memory training with warm-start initialization. Our work is based on and **modifies the IBM [AIHWKit](https://github.com/IBM/aihwkit)** simulator to support **multi-tile updates under limited conductance states**, enabling high-precision training on low-precision analog devices.

---

## 📌 Project Purpose

Analog in-memory computing enables energy-efficient deep learning, but suffers from limited precision due to quantization noise, device asymmetry, and low conductance states (e.g., 4–10 states). To overcome this, we propose a **multi-timescale residual learning (MRL)** framework combined with a **warm-start initialization**, which updates the most significant tiles first to accelerate convergence and improve final accuracy.

Specifically:

* We **customize the low-level AIHWKit core** (notably `rpu_transfer_device.cpp`) to support **warm-start tile updates**, triggered during training when loss plateau is detected.
* We adopt a **geometrically decaying update schedule** for tiles during residual learning, with **per-tile learning rates** that emphasize higher-precision accumulation.
* We demonstrate consistent improvements over baselines (TT-v1, TT-v2) on both **full analog LeNet-5 (MNIST)** and **partially analog ResNets (CIFAR-10)** under limited-state (4–10 state) ReRAM devices.

---

## 🔧 Modifications to AIHWKit

We implemented a custom warm-start flag mechanism and integrated it into the `TransferRPUDevice` logic. Key changes include:

- **`rpu_transfer_device.cpp`**:  
  - Core logic for warm-start tile transfer.
- **`rpu.h`**:  
  - Added `set_flags()` to expose custom device control from Python.
- **`rpu_simple_device.h`**:  
  - Declared `virtual set_flags_cpp()`.
- **`rpu_based_tiles.cpp`**:  
  - Bound `set_flags()` to Python via `this->set_flags_cpp()`.
- **`rpu_transfer_device.h`**:  
  - Overrode `set_flags_cpp()` and defined a warm-start control flag used in `finalCycleUpdate()`.

> ✅ The modified `aihwkit/` directory is included directly in this repo.  
> ❌ No external pip installation is required.

---
## 🚀 Usage

1. **Create environment:**

   ```bash
    conda create -n aihwkit-cuda-dev python=3.10 -y
    conda activate aihwkit-cuda-dev
    pip install torch numpy
    conda install mkl mkl-include -y
    conda install tensorboard matplotlib -y
    cd aihwkit
    source ./load_env.sh
    make build_inplace_cuda
    cd ..


2. **Run Training:**

You can run training experiments with different analog configurations using the following commands:

### 🟢 LeNet-5 on MNIST (fully analog)

```bash
python Mnist_LeNet5.py --SETTING="ResL" --CUDA=0
```

### 🟡 ResNet-18 on CIFAR-10 (partially analog)

```bash
python CIFAR-Resnet.py --optimizer="ResL" \
  -block-number 2 2 2 2 \
  -block-type D D D A A A \
  --CUDA=0 --io-perfect
```

### 🔵 ResNet-34 on CIFAR-10 (partially analog)

```bash
python CIFAR-Resnet.py --optimizer="ResL" \
  -block-number 3 4 6 3 \
  -block-type D D D A A A \
  --CUDA=0 --io-perfect
```

---

