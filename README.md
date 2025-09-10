<h1>Overview</h1>
WRITE

<h1>Dependencies</h1>
WRITE

<h1>Examples</h1>
WRITE

<h1>Release</h1>
WRITE

<h1>Source Code</h1>
In the Source Code folder, we have our C++ LibTorch implementation of the Loose Topological Node module and a PyBind11 wrapper for it. These can be compiled on Linux by running 
```bash
./Build_Module_Linux.bat
``` 
or on Windows by running 
```bat
Build_Module_Windows.bat
```
in the x64 Visual Studio Developer Command Prompt. We recommend using Ninja:
```bat
py -m pip install ninja
``` 
to compile faster, but not necessary. The output of this will be the .so or .pyd files found in the Release folder.
