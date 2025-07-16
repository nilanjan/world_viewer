use std::fs::File;
use std::io::Read;
use std::path::Path;
use gltf;

#[derive(Debug)]
pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

#[derive(Debug)]
pub struct GltfScene {
    pub meshes: Vec<Mesh>,
}

/// Reads a glTF file (.gltf or .glb), parses the mesh data, and returns a GltfScene.
/// This function expects the file to be in the standard glTF 2.0 format.
/// Only basic mesh data (positions, normals, indices) are extracted for GPU rendering.
pub fn load_gltf_scene<P: AsRef<Path>>(path: P) -> Result<GltfScene, String> {
    // Use the `gltf` crate for parsing. Add `gltf = "0.15"` to Cargo.toml dependencies.
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let gltf = gltf::Gltf::from_reader(file).map_err(|e| format!("Failed to parse glTF: {}", e))?;

    // Load the binary buffers
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for buffer in gltf.buffers() {
        let mut data = Vec::new();
        match buffer.source() {
            gltf::buffer::Source::Uri(uri) => {
                let buffer_path = path.parent().unwrap_or_else(|| Path::new(".")).join(uri);
                let mut f = File::open(&buffer_path)
                    .map_err(|e| format!("Failed to open buffer file {}: {}", uri, e))?;
                f.read_to_end(&mut data)
                    .map_err(|e| format!("Failed to read buffer file {}: {}", uri, e))?;
            }
            gltf::buffer::Source::Bin => {
                // For .glb files, the buffer is embedded
                if let Some(blob) = gltf.blob.as_ref() {
                    data.extend_from_slice(blob);
                } else {
                    return Err("Missing binary blob in .glb file".to_string());
                }
            }
        }
        buffers.push(data);
    }

    let mut meshes = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            // Positions
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            // Normals
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            // Indices
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|read_indices| read_indices.into_u32().collect())
                .unwrap_or_default();

            meshes.push(Mesh {
                positions,
                normals,
                indices,
            });
        }
    }

    Ok(GltfScene { meshes })
} 