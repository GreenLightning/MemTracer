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

Example configuration file `config.toml`:

```
input = 'bun_zipper.ply'
# output = 'output.png'
heuristic = 'sah'
size = [256, 256]

[camera]
position = [-0.0160, 0.1079, 0.2]
vertical_fov = 60
```

Command line options are:

```
./rt -config config.toml -width 128 -height 128 -input input.ply -output output.png -heuristic sah
```

```
./rt/rt -config ../workspace/data/config.toml -output test.png
TOOL_FILENAME=trace.bin LD_PRELOAD=./mem_trace/libmem_trace.so ./rt/rt -config ../workspace/data/config.toml -output test.png
```
