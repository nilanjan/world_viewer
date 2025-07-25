# world_viewer

## Changelog

**Recent changes since last commit:**
- Added a new `raytracer.rs` module for software ray tracing, including:
  - BVH (Bounding Volume Hierarchy) construction and update methods
  - BVH traversal and ray-triangle intersection as separate functions
  - Raytraced framebuffer output compatible with the SDL2 viewer
- Raytracer now supports scene rotation animation via a rotation angle argument (see `render_raytraced_scene`).
- Updated CLI (`main.rs`) to support a `--raytracer` flag for selecting between rasterization and ray tracing (currently uses a static angle, but can be animated in a loop).
- Added `show_sdl2_framebuffer` to the viewer for displaying arbitrary framebuffers
- Fixed and improved warnings (unused variables, type mismatches, division errors)
- Improved code documentation and modularity

---

A 3D world viewer written in Rust. Loads and displays glTF scenes using either a real-time software rasterizer or a software raytracer, with SDL2 for display. Supports textured and untextured models, interactive camera, command-line scene selection, and animated scene rotation (in both rasterizer and raytracer modes).

---

## Features
- Loads [glTF](https://www.khronos.org/gltf/) 2.0 3D scenes (ASCII `.gltf` and binary `.glb`)
- **Two rendering modes:**
  - **Software rasterizer:**
    - Real-time, interactive
    - Vertex positions, normals, UVs
    - Textured and untextured rendering
    - Basic Lambertian lighting
    - Z-buffering
    - Backface culling
    - Animated scene rotation (default)
  - **Software raytracer:**
    - BVH acceleration structure for fast ray traversal
    - Ray-triangle intersection
    - Scene rotation animation (via angle argument)
    - Framebuffer output compatible with SDL2 viewer
    - (Experimental, slower than rasterizer)
- SDL2-based window for real-time display
- Command-line interface for scene, camera, and renderer selection
- Modular codebase: parsing, rendering, raytracing, and viewing separated
- Doxygen-style and rustdoc documentation
- Ready for CI/CD with GitHub Actions (see below)

---

## Getting Started

### Prerequisites
- Rust (1.62+)
- Cargo
- [SDL2](https://www.libsdl.org/) development libraries (for your OS)
- [ImageMagick](https://imagemagick.org/) (optional, for asset conversion)

#### On macOS:
```sh
brew install rust sdl2 imagemagick
```

### Building
```sh
cargo build --release
```

### Running
```sh
cargo run --release -- --scene Cube --camera-z 5.5
```
- `--scene` can be: Cube, ToyCar, Lantern, Sponza, Triangle
- `--camera-z` sets the camera's Z (depth) position (default: 5.5)
- `--raytracer` uses the raytracer instead of the default rasterizer (currently with a static rotation angle; for animation, call `render_raytraced_scene` in a loop with an increasing angle)

#### Rasterizer (default, real-time, animated):
```sh
cargo run --release -- --scene ToyCar --camera-z 7.0
```

#### Raytracer (static image, no animation by default):
```sh
cargo run --release -- --scene ToyCar --camera-z 7.0 --raytracer
```

#### Raytracer Animation Example (advanced):
To animate the raytraced scene, call `render_raytraced_scene` in a loop, incrementing the angle:
```rust
let mut angle = 0.0;
loop {
    render_raytraced_scene(&scene, camera_z, angle, &mut framebuffer);
    show_sdl2_framebuffer(&framebuffer);
    angle += 0.01;
}
```

---

## Project Structure

```
world_viewer/
├── src/
│   ├── main.rs         # CLI entry point
│   ├── lib.rs          # Module declarations
│   ├── gltf_parser.rs  # glTF parsing logic
│   ├── render.rs       # Software rasterizer
│   ├── raytracer.rs    # Software raytracer (BVH, intersection, framebuffer, animation)
│   └── viewer.rs       # SDL2-based viewer
├── assets/
│   └── model/
│       ├── Cube/
│       ├── ToyCar/
│       ├── Lantern/
│       ├── Sponza/
│       └── Triangle/
├── Cargo.toml
├── LICENSE
└── README.md
```

- **Each model directory** contains a `glTF/` subdirectory with `.gltf` files and textures.

---

## Assets
- Place your glTF models in `assets/model/<SceneName>/glTF/`.
- Example: `assets/model/Cube/glTF/Cube.gltf`
- Textures should be referenced by the glTF file and placed in the same directory.

---

## Dependencies
- [gltf](https://crates.io/crates/gltf) - glTF parsing
- [image](https://crates.io/crates/image) - Image decoding
- [sdl2](https://crates.io/crates/sdl2) - Window/display
- [clap](https://crates.io/crates/clap) - CLI parsing
- [nalgebra](https://crates.io/crates/nalgebra) - Math utilities
- [wgpu, winit, log, env_logger, pollster, cfg-if] (not all may be used in current code)

---

## Documentation

### Rustdoc (Recommended)
To generate and view documentation:
```sh
cargo doc --open
```

---

## Continuous Integration (CI/CD)

A GitHub Actions workflow is provided to build and test the project on macOS:

- File: `.github/workflows/ci.yml`

---

## License
MIT License. See [LICENSE](LICENSE).

---

## Author
Nilanjan Goswami (<one0blue@gmail.com>) 
