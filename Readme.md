# MemTracer

Bounding Volume Hierarchy Reconstruction from Memory Traces (using Binary Instrumentation).

## Overview

- `rt` example ray tracer
- `mem_trace` binary instrumentation tool
- `meminf` backchannel to pass buffer information

## Building

Download [NVBit](https://github.com/NVlabs/NVBit/releases) and copy the `core` directory into this directory, then run CMake.

Additional dependencies that should be installed:

Debian and Ubuntu: `sudo apt-get install libpng-dev libjpeg-dev libopenexr-dev`
Arch Linux: `sudo pacman -S libpng libjpeg openexr`
Fedora: `sudo dnf install libpng-devel libjpeg-devel openexr-devel`

## Running

```
LD_PRELOAD=./mem_trace/libmem_trace.so 
./rt/rt ../workspace/data/bun_zipper.ply 256 -0.0081 0.1079 5.5 1 0 0 0 1 0 0 0 1 3 sah
```
