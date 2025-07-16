use std::fs::File;
use std::io::Read;
use std::path::Path;
use gltf;

use std::f32::consts::PI;
use std::time::{Duration, Instant};

use sdl2::pixels::PixelFormatEnum;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;


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

#[derive(Debug, Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug, Clone, Copy)]
struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn to_vec4(self, w: f32) -> Vec4 {
        Vec4 { x: self.x, y: self.y, z: self.z, w }
    }
    fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len == 0.0 { self } else { Self { x: self.x / len, y: self.y / len, z: self.z / len } }
    }
    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
    fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }
}

impl Vec4 {
    fn to_vec3(self) -> Vec3 {
        if self.w == 0.0 { Vec3::new(self.x, self.y, self.z) }
        else { Vec3::new(self.x / self.w, self.y / self.w, self.z / self.w) }
    }
}

#[derive(Debug, Clone, Copy)]
struct Mat4 {
    m: [[f32; 4]; 4],
}

impl Mat4 {
#[allow(dead_code)]
    fn identity() -> Self {
        Self { m: [
            [1.0,0.0,0.0,0.0],
            [0.0,1.0,0.0,0.0],
            [0.0,0.0,1.0,0.0],
            [0.0,0.0,0.0,1.0],
        ]}
    }
    fn mul_vec4(&self, v: Vec4) -> Vec4 {
        let m = &self.m;
        Vec4 {
            x: m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z + m[0][3]*v.w,
            y: m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z + m[1][3]*v.w,
            z: m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z + m[2][3]*v.w,
            w: m[3][0]*v.x + m[3][1]*v.y + m[3][2]*v.z + m[3][3]*v.w,
        }
    }
    fn mul_mat4(&self, other: &Mat4) -> Mat4 {
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
    fn perspective(fov_y: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
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
    fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
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
    fn rotation_y(angle: f32) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        Self { m: [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]}
    }
    #[allow(dead_code)]
    fn translation(x: f32, y: f32, z: f32) -> Self {
        let mut m = Self::identity();
        m.m[0][3] = x;
        m.m[1][3] = y;
        m.m[2][3] = z;
        m
    }
    #[allow(dead_code)]
    fn scale(sx: f32, sy: f32, sz: f32) -> Self {
        let mut m = Self::identity();
        m.m[0][0] = sx;
        m.m[1][1] = sy;
        m.m[2][2] = sz;
        m
    }
}

fn to_screen(v: Vec3) -> (i32, i32) {
    let x = ((v.x + 1.0) * 0.5 * (WIDTH as f32)) as i32;
    let y = ((1.0 - (v.y + 1.0) * 0.5) * (HEIGHT as f32)) as i32;
    (x, y)
}

fn edge_function(a: (i32, i32), b: (i32, i32), c: (i32, i32)) -> i32 {
    (c.0 - a.0) * (b.1 - a.1) - (c.1 - a.1) * (b.0 - a.0)
}

fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min { min } else if x > max { max } else { x }
}

fn render_scene(scene: &GltfScene, angle: f32, framebuffer: &mut [u32], zbuffer: &mut [f32]) {
    framebuffer.fill(0x202020ff); // dark gray background
    zbuffer.fill(f32::INFINITY);

    // Camera and transforms
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let proj = Mat4::perspective(PI / 3.0, aspect, 0.1, 100.0);
    let view = Mat4::look_at(Vec3::new(0.0, 0.5, 2.5), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    let model = Mat4::rotation_y(angle);
    let mvp = proj.mul_mat4(&view).mul_mat4(&model);

    let light_dir = Vec3::new(0.5, 1.0, 1.0).normalize();

    for mesh in &scene.meshes {
        let positions = &mesh.positions;
        let normals = &mesh.normals;
        let indices = &mesh.indices;

        // Transform all vertices
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

            // Backface culling
            let s0 = to_screen(v0);
            let s1 = to_screen(v1);
            let s2 = to_screen(v2);
            let area = edge_function(s0, s1, s2);
            if area <= 0 { continue; }

            // Bounding box
            let min_x = clamp(s0.0.min(s1.0).min(s2.0), 0, (WIDTH-1) as i32);
            let max_x = clamp(s0.0.max(s1.0).max(s2.0), 0, (WIDTH-1) as i32);
            let min_y = clamp(s0.1.min(s1.1).min(s2.1), 0, (HEIGHT-1) as i32);
            let max_y = clamp(s0.1.max(s1.1).max(s2.1), 0, (HEIGHT-1) as i32);

            // Perspective correct barycentric
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

                            // Interpolate normal
                            let normal = n0.scale(b0).add(n1.scale(b1)).add(n2.scale(b2)).normalize();
                            let intensity = clamp(normal.dot(light_dir), 0.0, 1.0);
                            let color = ((intensity * 200.0) as u8) as u32;
                            let pixel = (color << 16) | (color << 8) | color | 0xff000000;
                            framebuffer[idx] = pixel;
                        }
                    }
                }
            }
        }
    }
}

fn show_sdl2_viewer(scene: &GltfScene) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let window = video_subsystem.window("Software Renderer", WIDTH as u32, HEIGHT as u32)
        .position_centered()
        .vulkan()
        .build()
        .unwrap();
    let mut canvas = window.into_canvas().software().build().unwrap();
    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator.create_texture_streaming(PixelFormatEnum::ARGB8888, WIDTH as u32, HEIGHT as u32).unwrap();

    let mut framebuffer = vec![0u32; WIDTH * HEIGHT];
    let mut zbuffer = vec![0.0f32; WIDTH * HEIGHT];

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut angle = 0.0f32;
    let mut last_time = Instant::now();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
                _ => {}
            }
        }
        let now = Instant::now();
        let dt = now.duration_since(last_time);
        last_time = now;
        //angle += dt.as_secs_f32() * 0.7;
        angle = 0.0f32;

        render_scene(scene, angle, &mut framebuffer, &mut zbuffer);

        texture.with_lock(None, |buffer: &mut [u8], _pitch: usize| {
            let src = unsafe {
                std::slice::from_raw_parts(
                    framebuffer.as_ptr() as *const u8,
                    framebuffer.len() * 4
                )
            };
            buffer.copy_from_slice(src);
        }).unwrap();

        canvas.clear();
        canvas.copy(&texture, None, None).unwrap();
        canvas.present();

        ::std::thread::sleep(Duration::from_millis(16));
    }
}

fn main() {
    let scene = load_gltf_scene("/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/ToyCar/glTF/ToyCar.gltf").unwrap();
    //let scene = load_gltf_scene("/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/Triangle/glTF/Triangle.gltf").unwrap();
    show_sdl2_viewer(&scene);
    //show_sdl2_wireframe(&scene);
}

// This function renders the mesh using SDL2's built-in drawing (not the software framebuffer).
#[allow(dead_code)]
fn show_sdl2_wireframe(scene: &GltfScene) {
    use sdl2::pixels::Color;
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("GLTF Wireframe Viewer", WIDTH as u32, HEIGHT as u32)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut angle = 0.0f32;
    let mut last_time = Instant::now();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => break 'running,
                _ => {}
            }
        }
        let now = Instant::now();
        let dt = now.duration_since(last_time);
        last_time = now;
        angle += dt.as_secs_f32() * 0.7;

        // Camera and transforms
        let aspect = WIDTH as f32 / HEIGHT as f32;
        let proj = Mat4::perspective(PI / 3.0, aspect, 0.1, 100.0);
        let view = Mat4::look_at(Vec3::new(0.0, 0.5, 2.5), Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let model = Mat4::rotation_y(angle);
        let mvp = proj.mul_mat4(&view).mul_mat4(&model);

        canvas.set_draw_color(Color::RGB(32, 32, 32));
        canvas.clear();

        for mesh in &scene.meshes {
            // Project all vertices
            let mut projected: Vec<(i32, i32)> = Vec::with_capacity(mesh.positions.len());
            for &pos in &mesh.positions {
                let v = Vec3::new(pos[0], pos[1], pos[2]).to_vec4(1.0);
                let mut p = mvp.mul_vec4(v);
                if p.w.abs() > 1e-5 {
                    p.x /= p.w;
                    p.y /= p.w;
                    p.z /= p.w;
                }
                let (sx, sy) = to_screen(Vec3::new(p.x, p.y, p.z));
                projected.push((sx, sy));
            }

            // Draw wireframe triangles
            canvas.set_draw_color(Color::RGB(200, 200, 200));
            let indices = &mesh.indices;
            let n = indices.len();
            let mut i = 0;
            while i + 2 < n {
                let ia = indices[i] as usize;
                let ib = indices[i+1] as usize;
                let ic = indices[i+2] as usize;
                if ia < projected.len() && ib < projected.len() && ic < projected.len() {
                    let a = projected[ia];
                    let b = projected[ib];
                    let c = projected[ic];
                    let _ = canvas.draw_line(a, b);
                    let _ = canvas.draw_line(b, c);
                    let _ = canvas.draw_line(c, a);
                }
                i += 3;
            }
        }

        canvas.present();
        ::std::thread::sleep(Duration::from_millis(16));
    }
}




