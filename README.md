<h1>Overview</h1>
WRITE

<h1>Dependencies</h1>
WRITE

<h1>Examples</h1>
WRITE

<h1>Release</h1>
The `Release` folder contains a python file with our strict node loss functions (`Strict Node Penalties.py`), a `.pyd` (Windows) and `.so` (Linux) pre-compiled version of our LooseTopologicalNode module, which can be used by `import LooseTopologicalNode` if the `.pyd` or `.so` file is in the same directory as your custom script. We have included an example (`Loose Node Example.py`) to illustrate how to use the LooseTopologicalNode module. One can simply download the latest Release of the project (under Releases) to get this easily.

<h1>Source Code</h1>
In the `Source Code` folder, we have our C++ LibTorch implementation of the Loose Topological Node module and a PyBind11 wrapper for it.

These can be compiled:

- On **Linux** by running:

    ```bash
    ./Build_Module_Linux.bat
    ```

- On **Windows** by running:

    ```bat
    Build_Module_Windows.bat
    ```

> Run the Windows script in the **x64 Visual Studio Developer Command Prompt**.

We recommend using **Ninja** to compile faster (optional):

```bash
py -m pip install ninja
```

After building, the output will be a `.so` or `.pyd` file that will match those in the `Release` folder. Our code is written for `C++ 17` and `Python 3.13` and will not necessarily run in other versions.
