/*! \file gltf_parser.rs
    \brief glTF parsing logic for world_viewer. Extracts mesh, material, and texture data from glTF files.
*/

use std::fs::File;
use std::io::Read;
use std::path::Path;
use gltf;
use image::io::Reader as ImageReader;

/// Represents a single mesh with geometry and material reference.
#[derive(Debug)]
pub struct Mesh {
    /// Vertex positions (x, y, z)
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (x, y, z)
    pub normals: Vec<[f32; 3]>,
    /// Vertex UV coordinates (u, v)
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices
    pub indices: Vec<u32>,
    /// Index into the materials array (if any)
    pub material_index: Option<usize>,
}

/// Represents a decoded RGBA8 texture.
#[derive(Debug)]
pub struct Texture {
    /// Texture width in pixels
    pub width: u32,
    /// Texture height in pixels
    pub height: u32,
    /// Raw RGBA8 pixel data (row-major)
    pub data: Vec<u8>,
}

/// Represents a material, including a reference to a base color texture.
#[derive(Debug)]
pub struct Material {
    /// Index into the textures array for the base color texture (if any)
    pub base_color_texture: Option<usize>,
}

/// Represents a parsed glTF scene, including all meshes, textures, and materials.
#[derive(Debug)]
pub struct GltfScene {
    /// All meshes in the scene
    pub meshes: Vec<Mesh>,
    /// All decoded textures in the scene
    pub textures: Vec<Texture>,
    /// All materials in the scene
    pub materials: Vec<Material>,
}

/// \brief Loads a glTF file and parses all mesh, material, and texture data.
/// \param path Path to the .gltf or .glb file
/// \return GltfScene containing all parsed data, or an error string
pub fn load_gltf_scene<P: AsRef<Path>>(path: P) -> Result<GltfScene, String> {
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
                if let Some(blob) = gltf.blob.as_ref() {
                    data.extend_from_slice(blob);
                } else {
                    return Err("Missing binary blob in .glb file".to_string());
                }
            }
        }
        buffers.push(data);
    }

    // Parse images (decode to RGBA8)
    let mut textures = Vec::new();
    for image in gltf.images() {
        let img_data = match image.source() {
            gltf::image::Source::Uri { uri, .. } => {
                let img_path = path.parent().unwrap_or_else(|| Path::new(".")).join(uri);
                let img = ImageReader::open(&img_path)
                    .map_err(|e| format!("Failed to open image {}: {}", uri, e))?
                    .decode()
                    .map_err(|e| format!("Failed to decode image {}: {}", uri, e))?;
                img.to_rgba8()
            }
            gltf::image::Source::View { view, mime_type: _ } => {
                let buffer = &buffers[view.buffer().index()];
                let start = view.offset();
                let end = start + view.length();
                let img = image::load_from_memory(&buffer[start..end])
                    .map_err(|e| format!("Failed to decode embedded image: {}", e))?;
                img.to_rgba8()
            }
        };
        let (width, height) = (img_data.width(), img_data.height());
        textures.push(Texture {
            width,
            height,
            data: img_data.into_raw(),
        });
    }

    // Parse materials
    let mut materials = Vec::new();
    for mat in gltf.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let base_color_texture = pbr.base_color_texture().map(|info| info.texture().index());
        materials.push(Material {
            base_color_texture,
        });
    }

    // Parse meshes
    let mut meshes = Vec::new();
    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_default();
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|read_indices| read_indices.into_u32().collect())
                .unwrap_or_default();
            let material_index = primitive.material().index();
            meshes.push(Mesh {
                positions,
                normals,
                uvs,
                indices,
                material_index,
            });
        }
    }

    Ok(GltfScene {
        meshes,
        textures,
        materials,
    })
} 