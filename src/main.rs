// ===============================
// world_viewer: Main Entry Point
// ===============================
// This file parses command line arguments and launches the viewer with the selected scene and camera settings.
// It uses the 'clap' crate for robust argument parsing and provides helpful usage information.

use world_viewer::gltf_parser::load_gltf_scene;
use world_viewer::viewer::show_sdl2_viewer;
use world_viewer::render::{WIDTH, HEIGHT};
use world_viewer::raytracer::render_raytraced_scene;
use clap::{Arg, Command};

fn main() {
    // -----------------------------------
    // Command Line Argument Parsing (clap)
    // -----------------------------------
    // Set up the CLI with scene, camera_z, and raytracer options, and help/version info.
    let matches = Command::new("world_viewer")
        .version("0.1.0")
        .author("Nilanjan Goswami <one0blue@gmail.com>")
        .about("A 3D world viewer using Rust. Loads and displays glTF scenes with software rendering.")
        .arg(
            Arg::new("scene")
                .short('s')
                .long("scene")
                .value_name("SCENE")
                .help("Name of the scene to load (Cube, ToyCar, Lantern, Sponza, Triangle)")
                .default_value("Triangle")
        )
        .arg(
            Arg::new("camera_z")
                .short('z')
                .long("camera-z")
                .value_name("DEPTH")
                .help("Camera Z (depth) position for the look-at camera (e.g., 5.5)")
                .default_value("5.5")
        )
        .arg(
            Arg::new("raytracer")
                .long("raytracer")
                .help("Use the raytracer for rendering instead of the default rasterizer.")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    // -----------------------------------
    // Parse Arguments
    // -----------------------------------
    // Get the scene name (case-insensitive), camera depth, and raytracer flag from the CLI.
    let scene_name = matches.get_one::<String>("scene").unwrap().to_lowercase();
    let camera_z: f32 = matches.get_one::<String>("camera_z").unwrap().parse().unwrap_or(5.5);
    let use_raytracer = *matches.get_one::<bool>("raytracer").unwrap_or(&false);

    // -----------------------------------
    // Map Scene Name to File Path
    // -----------------------------------
    // Add new scenes here as needed.
    let scene_path = match scene_name.as_str() {
        "cube" => "/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/Cube/glTF/Cube.gltf",
        "toycar" => "/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/ToyCar/glTF/ToyCar.gltf",
        "lantern" => "/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/Lantern/glTF/Lantern.gltf",
        "sponza" => "/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/Sponza/glTF/Sponza.gltf",
        _ => "/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/Triangle/glTF/Triangle.gltf",
    };

    // -----------------------------------
    // Load the glTF Scene
    // -----------------------------------
    // This parses the mesh, material, and texture data from the file.
    let scene = load_gltf_scene(scene_path).expect("Failed to load glTF scene");

    // -----------------------------------
    // Launch the SDL2 Viewer
    // -----------------------------------
    // This opens a window and renders the scene using a software rasterizer or raytracer.
    if use_raytracer {
        // Raytracer: render the scene to a framebuffer, then display with SDL2
        let mut framebuffer = vec![0u32; WIDTH * HEIGHT];
        render_raytraced_scene(&scene, camera_z, 0.0, &mut framebuffer);
        world_viewer::viewer::show_sdl2_framebuffer(&framebuffer);
    } else {
        // Default: rasterizer
        show_sdl2_viewer(&scene, camera_z);
    }
    // To use the wireframe viewer instead, uncomment the following line:
    // show_sdl2_wireframe(&scene);
}




