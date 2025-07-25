/*! \file raytracer.rs
    \brief Software ray tracer for world_viewer. Includes BVH acceleration, ray-triangle intersection, and framebuffer output.
*/

use crate::gltf_parser::{GltfScene, Mesh};
use crate::render::{WIDTH, HEIGHT, Vec3};

use std::f32::consts::PI;

/// Represents a ray in 3D space
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

/// BVH node for acceleration structure
#[derive(Debug)]
pub enum BVHNode {
    Leaf {
        aabb: AABB,
        tri_indices: Vec<(usize, usize, usize)>, // (mesh_idx, tri_start, tri_end)
    },
    Internal {
        aabb: AABB,
        left: Box<BVHNode>,
        right: Box<BVHNode>,
    },
}

/// Build a BVH from the scene's meshes
pub fn bvh_build(scene: &GltfScene) -> BVHNode {
    // For simplicity, flatten all triangles into a list with mesh index
    let mut tris = Vec::new();
    for (mesh_idx, mesh) in scene.meshes.iter().enumerate() {
        for tri in mesh.indices.chunks(3) {
            if tri.len() == 3 {
                tris.push((mesh_idx, tri[0] as usize, tri[1] as usize, tri[2] as usize));
            }
        }
    }
    // Recursively build BVH
    build_bvh_recursive(&scene.meshes, &tris)
}

/// Update the BVH (e.g., after mesh transforms or animation)
pub fn bvh_update(_scene: &GltfScene, _bvh: &mut BVHNode) {
    // For static scenes, this is a no-op. For animated scenes, update AABBs here.
    // Placeholder for future extension.
}

/// Recursively build the BVH
fn build_bvh_recursive(meshes: &[Mesh], tris: &[(usize, usize, usize, usize)]) -> BVHNode {
    // Compute AABB for all triangles in this node
    let aabb = compute_aabb(meshes, tris);
    if tris.len() <= 4 {
        // Leaf node
        return BVHNode::Leaf {
            aabb,
            tri_indices: tris.iter().map(|&(m, a, b, _c)| (m, a, b)).collect(),
        };
    }
    // Split along longest axis
    let axis = longest_axis(&aabb);
    let mut sorted = tris.to_vec();
    sorted.sort_by(|a, b| {
        let ca = centroid(meshes, a);
        let cb = centroid(meshes, b);
        ca[axis].partial_cmp(&cb[axis]).unwrap()
    });
    let mid = sorted.len() / 2;
    let left = build_bvh_recursive(meshes, &sorted[..mid]);
    let right = build_bvh_recursive(meshes, &sorted[mid..]);
    BVHNode::Internal {
        aabb,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Compute the AABB for a set of triangles
fn compute_aabb(meshes: &[Mesh], tris: &[(usize, usize, usize, usize)]) -> AABB {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &(mesh_idx, i0, i1, i2) in tris {
        let mesh = &meshes[mesh_idx];
        for &i in &[i0, i1, i2] {
            let p = mesh.positions[i];
            min.x = min.x.min(p[0]); min.y = min.y.min(p[1]); min.z = min.z.min(p[2]);
            max.x = max.x.max(p[0]); max.y = max.y.max(p[1]); max.z = max.z.max(p[2]);
        }
    }
    AABB { min, max }
}

/// Find the longest axis of an AABB (0=x, 1=y, 2=z)
fn longest_axis(aabb: &AABB) -> usize {
    let dx = aabb.max.x - aabb.min.x;
    let dy = aabb.max.y - aabb.min.y;
    let dz = aabb.max.z - aabb.min.z;
    if dx > dy && dx > dz { 0 } else if dy > dz { 1 } else { 2 }
}

/// Compute centroid of a triangle
fn centroid(meshes: &[Mesh], tri: &(usize, usize, usize, usize)) -> [f32; 3] {
    let mesh = &meshes[tri.0];
    let p0 = mesh.positions[tri.1];
    let p1 = mesh.positions[tri.2];
    let p2 = mesh.positions[tri.3];
    [
        (p0[0] + p1[0] + p2[0]) / 3.0,
        (p0[1] + p1[1] + p2[1]) / 3.0,
        (p0[2] + p1[2] + p2[2]) / 3.0,
    ]
}

/// Test if a ray intersects an axis-aligned bounding box (AABB)
fn aabb_intersect(aabb: &AABB, ray: &Ray) -> bool {
    let inv_dir = Vec3::new(
        1.0 / ray.dir.x,
        1.0 / ray.dir.y,
        1.0 / ray.dir.z,
    );
    let mut tmin = (aabb.min.x - ray.origin.x) * inv_dir.x;
    let mut tmax = (aabb.max.x - ray.origin.x) * inv_dir.x;
    if tmin > tmax { std::mem::swap(&mut tmin, &mut tmax); }
    let mut tymin = (aabb.min.y - ray.origin.y) * inv_dir.y;
    let mut tymax = (aabb.max.y - ray.origin.y) * inv_dir.y;
    if tymin > tymax { std::mem::swap(&mut tymin, &mut tymax); }
    if (tmin > tymax) || (tymin > tmax) { return false; }
    if tymin > tmin { tmin = tymin; }
    if tymax < tmax { tmax = tymax; }
    let mut tzmin = (aabb.min.z - ray.origin.z) * inv_dir.z;
    let mut tzmax = (aabb.max.z - ray.origin.z) * inv_dir.z;
    if tzmin > tzmax { std::mem::swap(&mut tzmin, &mut tzmax); }
    if (tmin > tzmax) || (tzmin > tmax) { return false; }
    true
}

/// Traverse the BVH and find the closest intersection
pub fn bvh_traverse<'a>(bvh: &'a BVHNode, ray: &Ray, scene: &'a GltfScene) -> Option<(usize, usize, f32)> {
    // Returns (mesh_idx, tri_idx, t) for closest hit
    let mut stack = vec![bvh];
    let mut closest = None;
    let mut closest_t = f32::INFINITY;
    while let Some(node) = stack.pop() {
        match node {
            BVHNode::Leaf { aabb, tri_indices } => {
                if !aabb_intersect(aabb, ray) { continue; }
                for &(mesh_idx, tri_start, tri_end) in tri_indices {
                    let mesh = &scene.meshes[mesh_idx];
                    for tri in (tri_start..=tri_end).step_by(3) {
                        if tri + 2 >= mesh.indices.len() { continue; }
                        let i0 = mesh.indices[tri] as usize;
                        let i1 = mesh.indices[tri + 1] as usize;
                        let i2 = mesh.indices[tri + 2] as usize;
                        if let Some(t) = ray_triangle_intersect(ray, &mesh.positions[i0], &mesh.positions[i1], &mesh.positions[i2]) {
                            if t < closest_t {
                                closest_t = t;
                                closest = Some((mesh_idx, tri, t));
                            }
                        }
                    }
                }
            }
            BVHNode::Internal { aabb, left, right } => {
                if !aabb_intersect(aabb, ray) { continue; }
                stack.push(left);
                stack.push(right);
            }
        }
    }
    closest
}

/// Ray-triangle intersection (Möller–Trumbore)
pub fn ray_triangle_intersect(ray: &Ray, v0: &[f32; 3], v1: &[f32; 3], v2: &[f32; 3]) -> Option<f32> {
    let eps = 1e-6;
    let edge1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let edge2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let h = cross3(&ray.dir.to_array(), &edge2);
    let a = dot3(&edge1, &h);
    if a.abs() < eps { return None; }
    let f = 1.0 / a;
    let s = [ray.origin.x - v0[0], ray.origin.y - v0[1], ray.origin.z - v0[2]];
    let u = f * dot3(&s, &h);
    if u < 0.0 || u > 1.0 { return None; }
    let q = cross3(&s, &edge1);
    let v = f * dot3(&ray.dir.to_array(), &q);
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = f * dot3(&edge2, &q);
    if t > eps { Some(t) } else { None }
}

fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Vec3 {
    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

/// Render the scene using ray tracing with a rotation angle. Fills the framebuffer with ARGB pixels.
/// Call this in a loop with an increasing angle for animation.
/// Example:
///   let mut angle = 0.0;
///   loop { render_raytraced_scene(&scene, camera_z, angle, &mut framebuffer); angle += 0.01; }
pub fn render_raytraced_scene(scene: &GltfScene, camera_z: f32, angle: f32, framebuffer: &mut [u32]) {
    // Apply rotation to all mesh vertices (Y axis)
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let mut rotated_scene = scene.clone();
    for mesh in &mut rotated_scene.meshes {
        for pos in &mut mesh.positions {
            let x = pos[0];
            let z = pos[2];
            pos[0] = cos_a * x + sin_a * z;
            pos[2] = -sin_a * x + cos_a * z;
        }
    }
    let bvh = bvh_build(&rotated_scene);
    let width = WIDTH as u32;
    let height = HEIGHT as u32;
    let fov = PI / 3.0;
    let aspect = width as f32 / height as f32;
    let eye = Vec3::new(0.0, 0.5, camera_z);
    let center = Vec3::new(0.0, 0.0, 0.0);
    let forward = Vec3::new(center.x - eye.x, center.y - eye.y, center.z - eye.z).normalize();
    let _right = Vec3::new(forward.z, 0.0, -forward.x).normalize();
    for y in 0..height {
        for x in 0..width {
            let u = (2.0 * ((x as f32 + 0.5) / width as f32) - 1.0) * aspect * fov.tan();
            let v = (1.0 - 2.0 * ((y as f32 + 0.5) / height as f32)) * fov.tan();
            let dir = Vec3::new(u, v, -1.0).normalize();
            let ray = Ray { origin: eye, dir };
            let color = if let Some((_mesh_idx, _tri_idx, t)) = bvh_traverse(&bvh, &ray, &rotated_scene) {
                let shade = (1.0 - (t / 20.0)).clamp(0.0, 1.0);
                let c = (shade * 255.0) as u32;
                (c << 16) | (c << 8) | c | 0xff000000
            } else {
                0x202020ff
            };
            framebuffer[(y * width + x) as usize] = color;
        }
    }
} 