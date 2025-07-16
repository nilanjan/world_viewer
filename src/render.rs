/*! \file render.rs
    \brief Software rasterizer and math utilities for world_viewer. Handles transformation, lighting, and texture sampling.
*/

use std::f32::consts::PI;
use crate::gltf_parser::{GltfScene, Texture};

/// Output framebuffer width in pixels
pub const WIDTH: usize = 800;
/// Output framebuffer height in pixels
pub const HEIGHT: usize = 600;

/// 3D vector (x, y, z)
#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 4D vector (x, y, z, w)
#[derive(Debug, Clone, Copy)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec3 {
    /// Create a new Vec3
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    /// Convert to Vec4 with given w
    pub fn to_vec4(self, w: f32) -> Vec4 {
        Vec4 { x: self.x, y: self.y, z: self.z, w }
    }
    /// Normalize the vector
    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len == 0.0 { self } else { Self { x: self.x / len, y: self.y / len, z: self.z / len } }
    }
    /// Dot product
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    /// Cross product
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    /// Subtract two vectors
    pub fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
    /// Add two vectors
    pub fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
    /// Scale by a scalar
    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Vec4 {
    /// Convert to Vec3 by dividing by w (if w != 0)
    pub fn to_vec3(self) -> Vec3 {
        if self.w == 0.0 { Vec3::new(self.x, self.y, self.z) }
        else { Vec3::new(self.x / self.w, self.y / self.w, self.z / self.w) }
    }
}

/// 4x4 matrix for transformations
#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    /// Identity matrix
    #[allow(dead_code)]
    pub fn identity() -> Self {
        Self { m: [
            [1.0,0.0,0.0,0.0],
            [0.0,1.0,0.0,0.0],
            [0.0,0.0,1.0,0.0],
            [0.0,0.0,0.0,1.0],
        ]}
    }
    /// Multiply matrix by Vec4
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        let m = &self.m;
        Vec4 {
            x: m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            y: m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            z: m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            w: m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w,
        }
    }
    /// Matrix multiplication
    pub fn mul_mat4(&self, other: &Mat4) -> Mat4 {
        let mut result = Mat4 { m: [[0.0; 4]; 4] };
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result.m[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        result
    }
    /// Perspective projection matrix
    pub fn perspective(fov_y: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        let f = 1.0 / (fov_y / 2.0).tan();
        let nf = 1.0 / (z_near - z_far);
        let mut m = [[0.0; 4]; 4];
        m[0][0] = f / aspect;
        m[1][1] = f;
        m[2][2] = (z_far + z_near) * nf;
        m[2][3] = (2.0 * z_far * z_near) * nf;
        m[3][2] = -1.0;
        Self { m }
    }
    /// Look-at view matrix
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = center.sub(eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        let mut m = [[0.0; 4]; 4];
        m[0][0] = s.x; m[0][1] = s.y; m[0][2] = s.z; m[0][3] = -s.dot(eye);
        m[1][0] = u.x; m[1][1] = u.y; m[1][2] = u.z; m[1][3] = -u.dot(eye);
        m[2][0] = -f.x; m[2][1] = -f.y; m[2][2] = -f.z; m[2][3] = f.dot(eye);
        m[3][3] = 1.0;
        Self { m }
    }
    /// Y-axis rotation matrix
    pub fn rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self { m: [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]}
    }
    /// Translation matrix
    #[allow(dead_code)]
    pub fn translation(x: f32, y: f32, z: f32) -> Self {
        let mut m = Self::identity();
        m.m[0][3] = x;
        m.m[1][3] = y;
        m.m[2][3] = z;
        m
    }
    /// Scale matrix
    #[allow(dead_code)]
    pub fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        let mut m = Self::identity();
        m.m[0][0] = sx;
        m.m[1][1] = sy;
        m.m[2][2] = sz;
        m
    }
}

/// Convert NDC coordinates to screen pixel coordinates
pub fn to_screen(v: Vec3) -> (i32, i32) {
    let x = ((v.x + 1.0) * 0.5 * (WIDTH as f32)) as i32;
    let y = ((1.0 - (v.y + 1.0) * 0.5) * (HEIGHT as f32)) as i32;
    (x, y)
}

/// Compute the edge function for rasterization
fn edge_function(a: (i32, i32), b: (i32, i32), c: (i32, i32)) -> i32 {
    (c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0)
}

/// Clamp a value between min and max
fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min { min } else if x > max { max } else { x }
}

/// Sample a texture at the given UV coordinates (with wrapping)
fn sample_texture(texture: &Texture, uv: [f32; 2]) -> [u8; 4] {
    let u = uv[0].fract();
    let v = uv[1].fract();
    let x = (u * texture.width as f32).clamp(0.0, (texture.width - 1) as f32) as u32;
    let y = ((1.0 - v) * texture.height as f32).clamp(0.0, (texture.height - 1) as f32) as u32;
    let idx = ((y * texture.width + x) * 4) as usize;
    if idx + 3 < texture.data.len() {
        [texture.data[idx], texture.data[idx + 1], texture.data[idx + 2], texture.data[idx + 3]]
    } else {
        [255, 255, 255, 255]
    }
}

/// \brief Render the given scene to the framebuffer using software rasterization.
/// \param scene The parsed glTF scene
/// \param angle The model rotation angle (radians)
/// \param camera_z The camera's Z (depth) position
/// \param framebuffer Output ARGB framebuffer (WIDTH*HEIGHT)
/// \param zbuffer Output Z-buffer (WIDTH*HEIGHT)
pub fn render_scene(scene: &GltfScene, angle: f32, camera_z: f32, framebuffer: &mut [u32], zbuffer: &mut [f32]) {
    framebuffer.fill(0x202020ff); // dark gray background
    zbuffer.fill(f32::INFINITY);

    // Camera and transforms
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let proj = Mat4::perspective(PI / 3.0, aspect, 0.1, 100.0);
    let view = Mat4::look_at(Vec3::new(0.0, 0.5, camera_z), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    let model = Mat4::rotation_y(angle);
    let mvp = proj.mul_mat4(&view).mul_mat4(&model);

    let light_dir = Vec3::new(0.5, 1.0, 1.0).normalize();

    // For each mesh in the scene
    for mesh in &scene.meshes {
        let positions = &mesh.positions;
        let normals = &mesh.normals;
        let uvs = &mesh.uvs;
        let indices = &mesh.indices;
        let material_index = mesh.material_index;
        let texture = material_index
            .and_then(|mat_idx| scene.materials.get(mat_idx))
            .and_then(|mat| mat.base_color_texture)
            .and_then(|tex_idx| scene.textures.get(tex_idx));

        // Transform all vertices to NDC and world space
        let mut ndc_positions = Vec::with_capacity(positions.len());
        let mut world_normals = Vec::with_capacity(normals.len());
        for (i, &pos) in positions.iter().enumerate() {
            let p = Vec3::new(pos[0], pos[1], pos[2]);
            let v4 = mvp.mul_vec4(p.to_vec4(1.0));
            let ndc = v4.to_vec3();
            ndc_positions.push(ndc);

            // Transform normal (ignore translation, only rotation/scale)
            let n = if i < normals.len() {
                let n = Vec3::new(normals[i][0], normals[i][1], normals[i][2]);
                let n4 = model.mul_vec4(n.to_vec4(0.0));
                n4.to_vec3().normalize()
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            world_normals.push(n);
        }

        // Rasterize triangles
        for tri in indices.chunks(3) {
            if tri.len() < 3 { continue; }
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let v0 = ndc_positions[i0];
            let v1 = ndc_positions[i1];
            let v2 = ndc_positions[i2];

            let n0 = world_normals[i0];
            let n1 = world_normals[i1];
            let n2 = world_normals[i2];

            let uv0 = if i0 < uvs.len() { uvs[i0] } else { [0.0, 0.0] };
            let uv1 = if i1 < uvs.len() { uvs[i1] } else { [0.0, 0.0] };
            let uv2 = if i2 < uvs.len() { uvs[i2] } else { [0.0, 0.0] };

            // Backface culling
            let s0 = to_screen(v0);
            let s1 = to_screen(v1);
            let s2 = to_screen(v2);
            let area = edge_function(s0, s1, s2);
            if area <= 0 { continue; }

            // Bounding box for the triangle
            let min_x = clamp(s0.0.min(s1.0).min(s2.0), 0, (WIDTH-1) as i32);
            let max_x = clamp(s0.0.max(s1.0).max(s2.0), 0, (WIDTH-1) as i32);
            let min_y = clamp(s0.1.min(s1.1).min(s2.1), 0, (HEIGHT-1) as i32);
            let max_y = clamp(s0.1.max(s1.1).max(s2.1), 0, (HEIGHT-1) as i32);

            // Perspective correct barycentric interpolation
            let z0 = v0.z;
            let z1 = v1.z;
            let z2 = v2.z;

            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    let p = (x, y);
                    let w0 = edge_function(s1, s2, p) as f32;
                    let w1 = edge_function(s2, s0, p) as f32;
                    let w2 = edge_function(s0, s1, p) as f32;
                    let area_f = area as f32;
                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        // Interpolate z (depth)
                        let b0 = w0 / area_f;
                        let b1 = w1 / area_f;
                        let b2 = w2 / area_f;
                        let z = b0 * z0 + b1 * z1 + b2 * z2;
                        let idx = y as usize * WIDTH + x as usize;
                        if z < zbuffer[idx] {
                            zbuffer[idx] = z;

                            // Interpolate normal for lighting
                            let normal = n0.scale(b0).add(n1.scale(b1)).add(n2.scale(b2)).normalize();
                            let intensity = clamp(normal.dot(light_dir), 0.0, 1.0);

                            // Interpolate UV for texture sampling
                            let uv = [
                                b0 * uv0[0] + b1 * uv1[0] + b2 * uv2[0],
                                b0 * uv0[1] + b1 * uv1[1] + b2 * uv2[1],
                            ];

                            // Sample the texture if present, otherwise use grayscale
                            let color = if let Some(tex) = texture {
                                let texel = sample_texture(tex, uv);
                                let r = (texel[0] as f32 * intensity) as u8;
                                let g = (texel[1] as f32 * intensity) as u8;
                                let b = (texel[2] as f32 * intensity) as u8;
                                (r as u32) << 16 | (g as u32) << 8 | (b as u32) | 0xff000000
                            } else {
                                let c = (intensity * 200.0) as u8 as u32;
                                (c << 16) | (c << 8) | c | 0xff000000
                            };
                            framebuffer[idx] = color;
                        }
                    }
                }
            }
        }
    }
} 