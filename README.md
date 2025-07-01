# In-memory-Training-on-Analog-Devices
Code for Nips 2025
# In-memory Training on Analog Devices

This repository contains code and modified components of [AIHWKit](https://github.com/IBM/aihwkit), used in our experiments for a NeurIPS 2025 submission. The focus of this project is analog in-memory training with warm-start support, implemented via low-level customization of AIHWKit internals.

---

## 📌 Project Purpose

We apply analog training techniques to neural networks by modifying the `rpu_transfer_device.cpp` component in AIHWKit to support warm-start mechanisms that update multiple tiles across training epochs. All experiments are conducted on customized analog-aware models in PyTorch.

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

## 📁 Project Structure

```
.
├── aihwkit/               # Modified AIHWKit (with C++ backend and bindings)
├── Mnist_LeNet5.py        # MNIST training script
├── CIFAR-Resnet.py        # CIFAR training script
├── load_env.sh            # Environment setup helper
└── README.md              # Project documentation
```

---