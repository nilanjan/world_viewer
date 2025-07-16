use std::time::{Duration, Instant};
use sdl2::pixels::PixelFormatEnum;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use crate::gltf_parser::GltfScene;
use crate::render::{WIDTH, HEIGHT, render_scene};

pub fn show_sdl2_viewer(scene: &GltfScene) {
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

// This function renders the mesh using SDL2's built-in drawing (not the software framebuffer).
#[allow(dead_code)]
pub fn show_sdl2_wireframe(scene: &GltfScene) {
    use sdl2::pixels::Color;
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;
    use crate::render::{Vec3, Mat4, to_screen};
    use std::f32::consts::PI;

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
