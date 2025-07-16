# world_viewer

A 3D world viewer written in Rust. Loads and displays glTF scenes using software rasterization and SDL2 for display. Supports textured and untextured models, interactive camera, and command-line scene selection.

---

## Features
- Loads [glTF](https://www.khronos.org/gltf/) 2.0 3D scenes (ASCII `.gltf` and binary `.glb`)
- Software rasterizer with support for:
  - Vertex positions, normals, UVs
  - Textured and untextured rendering
  - Basic Lambertian lighting
  - Z-buffering
  - Backface culling
- SDL2-based window for real-time display
- Command-line interface for scene and camera selection
- Modular codebase: parsing, rendering, viewing separated
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

### Example
```sh
cargo run --release -- --scene ToyCar --camera-z 7.0
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
