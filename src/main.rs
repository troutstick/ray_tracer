use rayon::prelude::*;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, BufRead};
use std::ops::{Sub, Add};
use std::path::Path;
use std::io;
use std::time::Instant;
use std::env::args;

const OUTPUT_FOLDER: &str = "./images/output";
const INPUT_FOLDER: &str = "./images/input";

/// The default color of empty space.
const DEFAULT_BACKGROUND_COLOR: Color = Color::new(25, 25, 25);

/// The default color of darkness.
const DEFAULT_SHADOW_COLOR: Color = Color::new(0, 0, 0);

/// Scale the image resolution up/down with this.
const CAMERA_RES_SCALE: usize = 1;

const CAMERA_PIXEL_SIZE: f64 = 0.005 / CAMERA_RES_SCALE as f64;
const CAMERA_VERTICAL_RESOLUTION: usize = 200 * CAMERA_RES_SCALE;
const CAMERA_HORIZONTAL_RESOLUTION: usize = 200 * CAMERA_RES_SCALE;
const CAMERA_POSITION: Vector = Vector::new(0.1, 4.0, -10.0);
const CAMERA_PITCH: Radian = Radian(-0.25);
const CAMERA_YAW: Radian = Radian(0.01);

/// Parameters for sunlight.
const DEFAULT_LIGHTING_ANGLE: Vector = Vector::new(-0.5, 0.9, 0.4);
const DEFAULT_LIGHT_COLOR: Color = Color::white();

fn main() {
    let args: Vec<String> = args().collect();
    
    let input_filename = match args.len() {
        1 => "teapot",
        2 | 3 => args[1].as_str(),
        _ => "wrong number of args provided",
    };

    let num_output_images = if args.len() >= 3 {
        let n = args[2].parse::<usize>()
            .expect("Second arg should be number of output images");
        if n == 0 {
            panic!("must have nonzero output");
        }
        n
    } else {
        6
    };

    create_dir_all(OUTPUT_FOLDER).unwrap();
    create_dir_all(INPUT_FOLDER).unwrap();

    let input_path = format!("{}/{}.obj", INPUT_FOLDER, input_filename);

    println!("Processing `{}`...", input_path);

    let triangles = if let Ok(lines) = read_lines(input_path) {
        parse_input(lines)
    } else {
        panic!("Failed to parse path. Did you enter a valid filename?");
    };

    println!("Num triangles in scene: {:?}", triangles.len());

    let mut scene = Scene::new(triangles);

    let now = Instant::now();
    let mut prev_elapsed = 0.0;
    for i in 0..num_output_images {
        let f = File::create(format!("{}/{}_{}.ppm", OUTPUT_FOLDER, input_filename, i))
            .expect("Unable to create file");
        let f = BufWriter::new(f);

        scene.render_to_output(f);

        // scene.camera.pos.dz -= 1.0;
        scene.camera.yaw.0 += 0.1;

        let elapsed = now.elapsed().as_secs_f64();
        println!("Finished image {} in {:.3} s", i, elapsed - prev_elapsed);
        prev_elapsed = elapsed;
    }
    println!("Time to render images: {:.3} s", now.elapsed().as_secs_f64());

}

/// Parse the input into a set of triangles to render.
fn parse_input(lines: io::Lines<io::BufReader<File>>) -> Vec<Triangle> {
    let mut vertex_coords = Vec::new();
    let mut faces = Vec::new();
    for line in lines {
        if let Ok(s) = line {
            let mut line_iter = s.split_ascii_whitespace();
            if let Some(first_word) = line_iter.next() {
                match first_word {
                    "v" => {
                        let coords: Vec<f64> = line_iter
                            .map(|s| s.parse::<f64>().unwrap())
                            .collect();
                        if coords.len() != 3 {
                            panic!("unable to parse non 3d coordinates");
                        }
                        vertex_coords.push(coords);
                    },
                    "f" => {
                        let vertices: Vec<usize> = line_iter
                            .map(|s| s.parse::<usize>().unwrap())
                            .map(|i| i-1) // normalize into 0 index
                            .collect();
                        if vertices.len() != 3 {
                            panic!("unable to parse non-triangle polygons");
                        }
                        faces.push(vertices);
                    },
                    "#" => (), // ignore comment line
                    _ => panic!("only v and f lines readable in `.obj` files"),
                }
            }
        }
    }

    let get_3d_vect = |coord: &Vec<f64>| -> Vector {
        Vector::new(coord[0], coord[1], coord[2])
    };

    let get_vertices = |face: &Vec<usize>| -> Triangle {
        let v1 = get_3d_vect(&vertex_coords[face[0]]);
        let v2 = get_3d_vect(&vertex_coords[face[1]]);
        let v3 = get_3d_vect(&vertex_coords[face[2]]);
        Triangle::new(v1, v2, v3)
    };

    let triangles: Vec<Triangle> = faces
        .iter()
        .map(get_vertices)
        .collect();

    triangles
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// A triangle represented with its 3 vertices.
#[derive(Debug, Clone, PartialEq)]
struct Triangle {
    v1: Vector,
    v2: Vector,
    v3: Vector,
}

impl Triangle {
    fn new(v1: Vector, v2: Vector, v3: Vector) -> Triangle {
        Triangle { v1, v2, v3 }
    }

    /// Compute the average squared distance between the triangle's vertices
    /// and some target vector. Useful for judging distance to a triangle.
    #[allow(dead_code)]
    fn avg_dist_sq(&self, target: Vector) -> f64 {
        ((self.v1 - target).squared_magnitude()
        + (self.v2 - target).squared_magnitude()
        + (self.v3 - target).squared_magnitude())
        / 3.0
    }

    /// Give the plane that the triangle intersects.
    fn plane(&self) -> Plane {
        // Calculate normal vector from cross product of two sides
        let cp = {
            let v1 = self.v1 - self.v2;
            let v2 = self.v1 - self.v3;
            v1.cross_product(v2)
        }.normalized();

        // calculate offset
        Plane {
            a: cp.dx,
            b: cp.dy,
            c: cp.dz,
            k: -cp.dot_product(self.v1),
        }
    }

    fn bounding_box(&self) -> BoundingBox {
        BoundingBox {
            min_x: self.v1.dx.min(self.v2.dx.min(self.v3.dx)),
            max_x: self.v1.dx.max(self.v2.dx.max(self.v3.dx)),
            min_y: self.v1.dy.min(self.v2.dy.min(self.v3.dy)),
            max_y: self.v1.dy.max(self.v2.dy.max(self.v3.dy)),
            min_z: self.v1.dz.min(self.v2.dz.min(self.v3.dz)),
            max_z: self.v1.dz.max(self.v2.dz.max(self.v3.dz)),
        }
    }
}

/// A triangle's min x, max x, min y, max y, min z, max z.
struct BoundingBox {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
}

impl BoundingBox {

    /// Determine if another box overlaps this one.
    #[inline]
    fn overlaps(&self, other: &Self) -> bool {

        // Check if a 1d interval overlaps with another.
        let interval_overlaps = |min1, max1, min2, max2| min1 <= max2 && min2 <= max1;

        interval_overlaps(self.min_x, self.max_x, other.min_x, other.max_x)
        && interval_overlaps(self.min_y, self.max_y, other.min_y, other.max_y)
        && interval_overlaps(self.min_z, self.max_z, other.min_z, other.max_z)
    }

    /// Determine if a vector intersects this bounding box.
    #[inline]
    fn fast_intersect_check(&self, v: &Vector) -> bool {
        v.dx >= self.min_x && v.dx <= self.max_x
        && v.dy >= self.min_y && v.dy <= self.max_y
        && v.dz >= self.min_z && v.dz <= self.max_z
    }

    /// Determine if a vector intersects the top or bottom face.
    #[inline]
    #[allow(dead_code)]
    fn top_bottom_check(&self, v: Vector) -> bool {
        v.dx >= self.min_x && v.dx <= self.max_x
        && v.dy >= self.min_y && v.dy <= self.max_y
    }
    
    /// Determine if a vector intersects the left or right face.
    #[inline]
    #[allow(dead_code)]
    fn left_right_check(&self, v: Vector) -> bool {
        v.dx >= self.min_x && v.dx <= self.max_x
        && v.dz >= self.min_z && v.dz <= self.max_z
    }

    /// Determine if a vector intersects the left or right face.
    #[inline]
    #[allow(dead_code)]
    fn near_far_check(&self, v: Vector) -> bool {
        v.dy >= self.min_y && v.dy <= self.max_y
        && v.dz >= self.min_z && v.dz <= self.max_z

    }
    
    
    /// Give the six planes that make up the box's sides.
    ///
    /// A diagram of each of the vertices:
    ///
    ///       ^ Z-axis
    ///       |
    ///       |          
    ///       +-v2-----+ v4
    ///      /|       /|
    ///     / |      / |
    ///    +-v6-----+ v8
    ///    |  |     |  | 
    ///    |  +-----|--+----> Y-axis
    ///    | / v1   | / v3
    ///    |/       |/
    ///    +--------+
    ///   / v5       v7
    ///  /
    /// L  X-axis
    fn box_planes(&self) -> [Plane; 6] {
        
        let v1 = Vector::new(self.min_x, self.min_y, self.min_z);
        let v2 = Vector::new(self.min_x, self.min_y, self.max_z);
        // let v3 = Vector::new(self.min_x, self.max_y, self.min_z); // v3 skipped
        let v4 = Vector::new(self.min_x, self.max_y, self.max_z);
        let v5 = Vector::new(self.max_x, self.min_y, self.min_z);
        let v6 = Vector::new(self.max_x, self.min_y, self.max_z);
        let v7 = Vector::new(self.max_x, self.max_y, self.min_z);
        let v8 = Vector::new(self.max_x, self.max_y, self.max_z);
        [
            Triangle::new(v2, v6, v4).plane(), // top plane
            Triangle::new(v5, v1, v7).plane(), // bottom plane
            Triangle::new(v2, v6, v5).plane(), // left plane
            Triangle::new(v8, v4, v7).plane(), // right plane
            Triangle::new(v7, v6, v5).plane(), // near plane
            Triangle::new(v2, v1, v4).plane(), // far plane
        ]
    }
}

/// A plane in 3d space; all points satisfy the equation:
///     `ax + by + cz + k = 0`
struct Plane {
    a: f64,
    b: f64,
    c: f64,
    k: f64,
}

impl Plane {
    /// Return the intersection between this plane and a ray,
    /// defined by an origin and a direction vector.
    /// Also return whether the intersection occurs behind the origin.
    #[inline]
    fn intersection(&self, origin: Vector, direction: Vector) -> (Vector, bool) {
        // deconstruct plane
        let (_a,_b,_c,k) = (self.a, self.b, self.c, self.k);
        let normal_vect = self.normal();
        let lambda = -(normal_vect.dot_product(origin) + k) / normal_vect.dot_product(direction);
        (direction.scale(lambda) + origin, lambda > 0.0)
    }

    #[inline]
    fn normal(&self) -> Vector {
        let (a,b,c,_k) = (self.a, self.b, self.c, self.k);
        Vector::new(a,b,c)
    }
}

/// A view plane for a camera. The view plane is situated 1.0 units away from the camera.
struct ViewPlane {
    /// The height and width of each pixel.
    pixel_size: f64,
    /// How many pixels wide a view is.
    res_width: usize,
    /// How many pixels tall a view is.
    res_height: usize,
}

/// Representation of light coming in from a faraway source;
/// i.e. all incoming light enters at the same angle.
/// The angle vector should be a unit vector.
struct Sunlight {
    angle: Vector,
    color: Color,
}

impl Sunlight {
    fn new(angle: Vector, color: Color) -> Sunlight {
        Sunlight {
            angle: angle.normalized(),
            color,
        }
    }
}

/// The color (and brightness) of light.
/// RGB is in a 0-255 scale.
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl Color {
    const fn white() -> Color {
        Color { r: 255, g: 255, b: 255 }
    }
    const fn red() -> Color {
        Color { r: 255, g: 0, b: 0 }
    }
    const fn blue() -> Color {
        Color { r: 0, g: 0, b: 255 }
    }
    const fn green() -> Color {
        Color { r: 0, g: 255, b: 0 }
    }

    const fn new(r: u8, g: u8, b: u8) -> Color {
        Color { r, g, b }
    }

    fn scale(&self, scalar: f64) -> Color {
        Color {
            r: ((self.r as f64) * scalar) as u8,
            g: ((self.g as f64) * scalar) as u8,
            b: ((self.b as f64) * scalar) as u8,
        }
    }
}

/// A binary tree that allows efficient intersection search of the bounding boxes of a scene.
struct BoundingBoxHierarchy {
    boxes: Vec<BoundingBox>,
}

impl BoundingBoxHierarchy {
    fn new(boxes: &Vec<BoundingBox>) -> BoundingBoxHierarchy {
        let root = {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut min_z = f64::INFINITY;
            let mut max_z = f64::NEG_INFINITY;
            for b in boxes {
                min_x = b.min_x.min(min_x);
                max_x = b.max_x.max(max_x);
                min_y = b.min_y.min(min_y);
                max_y = b.max_y.max(max_y);
                min_z = b.min_z.min(min_z);
                max_z = b.max_z.max(max_z);
            }
            BoundingBox { min_x, max_x, min_y, max_y, min_z, max_z }
        };

        struct OctTreeNode {
            // The dimensions of this node.
            bounding_box: BoundingBox,

            // The bounding box that it contains.
            value: Option<BoundingBox>,
        }


    }
}

/// A set of triangles together make up a scene that can be viewed by a camera.
struct Scene {
    camera: Camera,
    triangles: Vec<Triangle>,
    triangle_planes: Vec<Plane>,
    bounding_boxes: Vec<BoundingBox>,

    /// All triangles in the scene fall within this bounding box.
    scene_bounding_box: BoundingBox,
    /// The six planes of the bounding box:
    /// [top bottom left right near far]
    box_planes: [Plane; 6],

    sunlight: Sunlight,
}

impl Scene {
    fn new(triangles: Vec<Triangle>) -> Scene {
        // Find the corresponding plane for every triangle
        let triangle_planes: Vec<Plane> = triangles
            .iter()
            .map(|t| t.plane())
            .collect();

        // Find the bounding box for every triangle
        let bounding_boxes: Vec<BoundingBox> = triangles
            .iter()
            .map(|t| t.bounding_box())
            .collect();

        let scene_bounding_box = {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut min_z = f64::INFINITY;
            let mut max_z = f64::NEG_INFINITY;
            for b in &bounding_boxes {
                min_x = b.min_x.min(min_x);
                max_x = b.max_x.max(max_x);
                min_y = b.min_y.min(min_y);
                max_y = b.max_y.max(max_y);
                min_z = b.min_z.min(min_z);
                max_z = b.max_z.max(max_z);
            }
            BoundingBox { min_x, max_x, min_y, max_y, min_z, max_z }
        };

        let box_planes = scene_bounding_box.box_planes();

        let sunlight = Sunlight::new(DEFAULT_LIGHTING_ANGLE, DEFAULT_LIGHT_COLOR);

        Scene {
            camera: Camera::new(),
            triangles,
            triangle_planes,
            bounding_boxes,
            scene_bounding_box,
            box_planes,
            sunlight,
        }
    }

    /// Generate an image of the given scene.
    fn iterate_over_rays(&self) -> Vec<Color> {
        let cam = &self.camera;

        let vp = &cam.view_plane;
        let res_height = vp.res_height as isize;
        let res_width = vp.res_width as isize;

        // Get index of center pixel
        let i_center = ((res_width - 1) / 2) as isize;
        let j_center = ((res_height - 1) / 2) as isize;

        let get_ray_direction = |pixel_index| {
            // pixel horizontal dimension
            let i = pixel_index % res_height;
            // pixel vertical dimension
            let j = (res_height * res_width - pixel_index - 1) / res_height;

            // The direction of the ray denoted by m.
            // The origin is the camera position.
            let m = Vector {
                dx: vp.pixel_size * ((i - i_center) as f64),
                dy: vp.pixel_size * ((j - j_center) as f64),
                dz: 1.0,
            }.yaw(cam.yaw).pitch(cam.pitch);
            
            m
        };

        // Get the index of the first triangle the ray strikes.
        let get_triangle_index = |direction| {
            let get_intersection = |p: &Plane| p.intersection(cam.pos, direction);

            // Return true if a direction vector strikes the scene's bounding box
            let intersects_bounding_box = |_direction| {
                self.box_planes.iter()
                    .map(get_intersection)
                    .map(|(v, _is_behind)| self.scene_bounding_box.fast_intersect_check(&v))
                    .any(|x| x)
            };
            if intersects_bounding_box(direction) {    
                self.closest_triangle_index(&get_intersection)
            } else {
                None
            }
        };

        let get_color = |option| {
            match option {
                Some((i, intersect)) => {

                    let get_intersection = |p: &Plane| p.intersection(intersect, self.sunlight.angle);

                    if self.intersects_triangle(&get_intersection, i) {
                        DEFAULT_SHADOW_COLOR
                    } else {
                        let plane: &Plane = &self.triangle_planes[i];
                        let normal_vector = plane.normal();
                        let brightness = normal_vector.dot_product(self.sunlight.angle);
                        self.sunlight.color.scale(brightness)
                    }
                },
                None => DEFAULT_BACKGROUND_COLOR,
            }
        };

        let mut pixel_colors = Vec::with_capacity(vp.res_height*vp.res_width);
        (0..(res_height*res_width))
            .into_par_iter()
            .map(get_ray_direction)
            .map(get_triangle_index)
            .map(get_color)
            .collect_into_vec(&mut pixel_colors);

        pixel_colors
    }

    fn render_to_output(&self, mut writer: BufWriter<File>) {
        let pixels = self.iterate_over_rays();

        let num_cols = self.camera.view_plane.res_width;
        let num_rows = self.camera.view_plane.res_height;
        writer.write(format!("P3\n{} {}\n255\n", num_cols, num_rows).as_bytes()).unwrap();
        for p in pixels {
            let s = format!("{} {} {}\n", p.r, p.g, p.b);
            writer.write(s.as_bytes()).unwrap();
        }
        writer.flush().unwrap();
    }

        /// Return true if a ray defined in an intersection detection closure
    /// collides with a triangle.
    /// Excludes one triangle.
    #[inline]
    fn intersects_triangle(&self, get_intersection: &dyn Fn(&Plane) -> (Vector, bool), to_exclude: usize) -> bool {
        let intersect_map = |(index, plane)| -> Option<(usize, Vector)> {
            let (intersect, is_ahead) = get_intersection(plane);
            if is_ahead {
                Some((index, intersect))
            } else {
                None
            }
        };
        let in_bounding_box = |(index, intersect): &(usize, Vector)| {
            let bbox = &self.bounding_boxes[*index];
            // skip the excluded triangle in check
            *index != to_exclude && bbox.fast_intersect_check(&intersect)
        };
        let intersects_t = |(index, intersect): (usize, Vector)| {
            let t = &self.triangles[index];
            intersect.slow_intersect_check(t)
        };

        self.triangle_planes.iter().enumerate()
            .map(intersect_map)
            .filter_map(|x|x)
            .filter(in_bounding_box)
            .any(intersects_t)
    }

    /// Find the index of closest triangle that intersects a ray,
    /// and the associated intersection point.
    #[inline]
    fn closest_triangle_index(&self, get_intersection: &dyn Fn(&Plane) -> (Vector, bool)) -> Option<(usize, Vector)> {
        let intersect_map = |(index, plane)| -> (usize, Vector) {
            (index, get_intersection(plane).0)
        };
        let in_bounding_box = |(index, intersect): &(usize, Vector)| {
            let bbox = &self.bounding_boxes[*index];
            bbox.fast_intersect_check(&intersect)
        };
        let dist_from_camera = |(index, intersect): (usize, Vector)| {
            let t = &self.triangles[index];
            if intersect.slow_intersect_check(t) {
                let cam_to_triangle = intersect - self.camera.pos;
                (index, cam_to_triangle.squared_magnitude(), intersect)
            } else {
                (0, f64::INFINITY, intersect)
            }

        };
        let min_dist = |tuple1: (usize, f64, Vector), tuple2: (usize, f64, Vector)| {
            if tuple1.1 < tuple2.1 {
                tuple1
            } else {
                tuple2
            }
        };

        let (index, dist, intersect) = self.triangle_planes.iter().enumerate()
            .map(intersect_map)
            .filter(in_bounding_box)
            .map(dist_from_camera)
            .fold((usize::MAX, f64::INFINITY, Vector::zero()), min_dist);

        if dist != f64::INFINITY {
            Some((index, intersect))
        } else {
            None
        }
    }

    /// Find the distance to the closest triangle that intersects a ray.
    /// Uses an externally defined closure to find intersection information.
    #[inline]
    fn closest_triangle_dist(&self, get_intersection: &dyn Fn(&Plane) -> Vector) -> f64 {
        let triangle_iter = self.triangle_planes.iter()
            .zip(self.bounding_boxes.iter()
            .zip(self.triangles.iter()));

        let intersect_map = |(plane,(bbox, t))| {
            let intersect = get_intersection(plane);
            (intersect,(bbox, t))
        };
        let in_bounding_box = |(intersect,(bbox, _t)): &(Vector, (&BoundingBox, &Triangle))| {
            bbox.fast_intersect_check(intersect)
        };
        let in_triangle = |(intersect,(_bbox, t)): &(Vector, (&BoundingBox, &Triangle))| {
            intersect.slow_intersect_check(t)  
        };
        let get_intersect_dist_sq = |(intersect,(_bbox, _t)): (Vector, (&BoundingBox, &Triangle))| {
            intersect.squared_magnitude()
        };
        triangle_iter
            .map(intersect_map)
            .filter(in_bounding_box)
            .filter(in_triangle)
            .map(get_intersect_dist_sq)
            .fold(f64::INFINITY, |a, b| a.min(b))
            .sqrt()
    }
}

/// A camera, determined by position, pitch angle, yaw angle, and view plane.
/// TODO: roll angle?
struct Camera {
    pos: Vector,
    pitch: Radian,
    yaw: Radian,
    view_plane: ViewPlane,
}

impl Camera {
    /// Default camera.
    fn new() -> Camera {
        let pos = CAMERA_POSITION;
        let pitch = CAMERA_PITCH;
        let yaw = CAMERA_YAW;
        let view_plane = ViewPlane {
            pixel_size: CAMERA_PIXEL_SIZE,
            res_height: CAMERA_VERTICAL_RESOLUTION,
            res_width: CAMERA_HORIZONTAL_RESOLUTION,
        };

        Camera { pos, pitch, yaw, view_plane }
    }
}

/// For measuring angles.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Radian(f64);

impl Radian {
    #[inline]
    fn sin(&self) -> f64 { self.0.sin() }
    
    #[inline]
    fn cos(&self) -> f64 { self.0.cos() }
}

/// A vector in 3d space.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Vector {
    dx: f64,
    dy: f64,
    dz: f64,
}

impl Vector {
    const fn zero() -> Vector {
        Vector { dx: 0.0, dy: 0.0, dz: 0.0 }
    }

    const fn new(dx: f64, dy: f64, dz: f64) -> Vector {
        Vector { dx, dy, dz }
    }

    /// Calculate the cross product vector
    #[inline]
    fn cross_product(&self, other: Self) -> Self {
        Vector {
            dx: self.dy * other.dz - self.dz * other.dy,
            dy: self.dz * other.dx - self.dx * other.dz,
            dz: self.dx * other.dy - self.dy * other.dx,
        }
    }

    /// Calculate the dot product scalar.
    #[inline]
    fn dot_product(&self, other: Self) -> f64 {
        self.dx * other.dx +
        self.dy * other.dy +
        self.dz * other.dz
    }

    /// Pitch the vector by r radians.
    #[inline]
    fn pitch(&self, r: Radian) -> Vector {
        Vector {
            dx: self.dx,
            dy: self.dy * r.cos() - self.dz * r.sin(),
            dz: self.dy * r.sin() - self.dz * r.cos(),
        }
    }

    /// Yaw the vector by r radians.
    #[inline]
    fn yaw(&self, r: Radian) -> Vector {
        Vector {
            dx: self.dx * r.cos() + self.dz * r.sin(),
            dy: self.dy,
            dz: -self.dx * r.sin() - self.dz * r.cos(),
        }
    }

    /// Scale the vector by some scalar value n.
    #[inline]
    fn scale(&self, n: f64) -> Vector {
        Vector {
            dx: n * self.dx,
            dy: n * self.dy,
            dz: n * self.dz,
        }
    }

    #[inline]
    fn squared_magnitude(&self) -> f64 {
        self.dx.powi(2) + self.dy.powi(2) + self.dz.powi(2)
    }

    #[inline]
    fn magnitude(&self) -> f64 {
        self.squared_magnitude().sqrt()
    }

    #[inline]
    fn normalized(&self) -> Vector {
        let m = self.magnitude();
        Vector {
            dx: self.dx / m,
            dy: self.dy / m,
            dz: self.dz / m,
        }
    }

    /// Return true if p1 and p2 are both on the same side of the vector v1 -> v2.
    #[inline]
    fn same_side(p1: Vector, p2: Vector, v1: Vector, v2: Vector) -> bool {
        let v = v2 - v1;
        let a = v.cross_product(p1 - v2);
        let b = v.cross_product(p2 - v2);
        a.dot_product(b) > 0.0
    }

    /// Determine if a vector is bounded within a triangle t.
    /// Note: This function is relatively slow; use bounding boxes
    /// for a fast intial intersection check.
    #[inline]
    fn slow_intersect_check(&self, t: &Triangle) -> bool {
        Vector::same_side(*self, t.v1, t.v2, t.v3)
        && Vector::same_side(*self, t.v2, t.v1, t.v3)
        && Vector::same_side(*self, t.v3, t.v1, t.v2)
    }
}

impl Sub for Vector {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Vector {
            dx: self.dx - other.dx,
            dy: self.dy - other.dy,
            dz: self.dz - other.dz,
        }
    }
}

impl Add for Vector {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Vector {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            dz: self.dz + other.dz,
        }
    }
}

/// Convert a 0 to 1 decimal input to a 0 to 255 integer output.
fn denormalize(f: f64) -> i32 {
    (f * 255.99) as i32
}

#[allow(dead_code)]
fn write_ppm_file(mut writer: BufWriter<File>) {
    let num_cols = 200;
    let num_rows = 100;
    writer.write(format!("P3\n{} {}\n255\n", num_cols, num_rows).as_bytes()).unwrap();
    for row in (0..num_rows).rev() {
        for col in 0..num_cols {
            let r = (col as f64) / (num_cols as f64);
            let g = (row as f64) / (num_rows as f64);
            let b = 0.2;
            let s = format!("{} {} {}\n", denormalize(r), denormalize(g), denormalize(b));
            writer.write(s.as_bytes()).unwrap();
        }
    }
    writer.flush().unwrap();
}

#[test]
fn simple_cross_product() {
    let v1 = Vector::new(2.0, 5.0, 6.0);
    let v2 = Vector::new(-3.0, 14.0, 20.0);
    let cp = Vector::new(16.0, -58.0, 43.0);
    assert_eq!(v1.cross_product(v2), cp);
}

#[test]
fn simple_dot_product() {
    let v1 = Vector::new(2.0, 5.0, 6.0);
    let v2 = Vector::new(-3.0, 14.0, 20.0);
    let dp = 184.0;
    assert_eq!(v1.dot_product(v2), dp);
}

#[test]
fn simple_cross_product2() {
    let v1 = Vector::new(2.0, 5.0, 6.0);
    let v2 = Vector::new(-3.0, 14.0, 20.0);
    let cp = Vector::new(16.0, -58.1, 43.0);
    assert_ne!(v1.cross_product(v2), cp);
}