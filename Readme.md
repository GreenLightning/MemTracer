# MemTracer

Source code for my master's thesis (currently not publicly available) and the paper **Reconstructing Bounding Volume Hierarchies from Memory Traces of Ray Tracers** to be published during PG 2022.

## Overview

- `rt` example ray tracer
- `mem_trace` binary instrumentation tool
- `mem_vis` trace analysis tool and gui

## Building

Clone recursively (`git clone --recursive`) or initialize submodules after cloning (`git submodule update --init`).

Download [NVBit](https://github.com/NVlabs/NVBit/releases) (version 1.5.4 or later) and copy the `core` directory into this directory.

Optionally, install image format libraries for the output of the example ray tracer (pbm is always available):

Debian and Ubuntu: `sudo apt-get install libpng-dev libjpeg-dev libopenexr-dev`
Arch Linux: `sudo pacman -S libpng libjpeg openexr`
Fedora: `sudo dnf install libpng-devel libjpeg-devel openexr-devel`

Run CMake.

## Running

Example configuration file `config.toml`:

```
input = 'bun_zipper.ply'
# output = 'output.png'
heuristic = 'sah'
size = [256, 256]

[camera]
position = [-0.0160, 0.1079, 0.2]

# euler rotation angles in degrees
rotation = [10, 20, 30]

# alternatively, rotation matrix
matrix = [
	1, 0, 0,
	0, 1, 0,
	0, 0, 1,
]

vertical_fov = 60
```

Command line options are:

```
./rt -config config.toml -width 128 -height 128 -input input.ply -output output.png -heuristic sah
```

```
./rt/rt -config ../workspace/data/config.toml -output test.png
TOOL_FILENAME=trace.bin TOOL_STORE_CONTENTS=1 LD_PRELOAD=./mem_trace/libmem_trace.so ./rt/rt -config ../workspace/data/config.toml -output test.png
```
