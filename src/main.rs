use world_viewer::gltf_parser::load_gltf_scene;
use world_viewer::viewer::show_sdl2_viewer;

fn main() {
    //let scene = load_gltf_scene("/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/ToyCar/glTF/ToyCar.gltf").unwrap();
    let scene = load_gltf_scene("/Users/nilg/Workspace/Code/Rust/world_viewer/assets/model/Triangle/glTF/Triangle.gltf").unwrap();
    show_sdl2_viewer(&scene);
    //show_sdl2_wireframe(&scene);
}




