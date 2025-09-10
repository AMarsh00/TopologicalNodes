# 🔗 Topological Node Module

This repo contains the PyTorch and LibTorch implementations for everything described in the **Topological Node** paper — including both strict and loose node variants.

You’ll find Python and C++ code, examples, and precompiled binaries ready to use.

---

## 📦 What's Included

- PyTorch implementations of the topological node losses
- A C++ `LibTorch` module (`LooseTopologicalNode`) with a PyBind11 wrapper
- Precompiled `.pyd` / `.so` files for easy import in Python
- Example scripts to help you get started
- All algorithms from the paper, already implemented and ready to run

---

## ⚙️ Dependencies

No complex setup needed — just a few Python packages:

```bash
py -m pip install torch numpy matplotlib sklearn networkx scipy ninja
```

> 🔧 **Ninja** is optional but makes compilation faster.  
> 🛠️ If you plan to compile the C++ module yourself, you’ll need:
> - `g++` on Linux
> - MSVC via the x64 Visual Studio Developer Command Prompt on Windows

---

## 🧪 Examples

The [`Examples`](./Examples) folder contains working implementations of all algorithms from the paper.

Each script saves its output in the same directory it's run from, so you can easily inspect results.

---

## 🚀 Release Folder

The [`Release`](./Release) folder contains:

- `Strict Node Penalties.py` — Python implementation of strict node loss
- `LooseTopologicalNode.pyd` (Windows) / `LooseTopologicalNode.so` (Linux) — compiled C++ module
- `Loose Node Example.py` — simple usage example for the loose module

### 🔧 Using the Module

If the compiled `.pyd` or `.so` file is in the same directory as your script, just import it like any other module:

```python
import LooseTopologicalNode
```

> 💡 You can also download the latest prebuilt files from the [Releases](../../releases) section on GitHub.

---

## 🛠️ Source Code & Building

The [`Source Code`](./Source%20Code) folder includes:

- The full C++ implementation of the `LooseTopologicalNode` module using LibTorch
- A PyBind11 wrapper for Python integration
- Build scripts for Windows and Linux

### 🏗️ Building the Module

Run the appropriate script from the `Source Code` folder:

- **Linux**:
    ```bash
    ./Build_Module_Linux.bat
    ```

- **Windows** (use the x64 Visual Studio Developer Command Prompt):
    ```bat
    Build_Module_Windows.bat
    ```

> ⚡ Using Ninja can speed up the build:
> ```bash
> py -m pip install ninja
> ```

### 🔍 Output

After building, you'll get a `.so` or `.pyd` file in the `Release` folder — ready to be imported in Python:

```python
import LooseTopologicalNode
```

> 📌 The code is written for **C++17** and **Python 3.13** — other versions might not work out of the box.

---

## 📁 Folder Structure

```text
.
├── Release/
│   ├── Strict Node Penalties.py
│   ├── LooseTopologicalNode.pyd / .so
│   └── Loose Node Example.py
├── Source Code/
│   ├── C++ implementation
│   ├── PyBind11 wrapper
│   └── Build scripts
├── Examples/
│   └── Algorithm implementations
└── README.md
```

---

## 📄 License

[GNU Affero Public License](./LICENSE)

---
