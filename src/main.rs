use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, BufRead};
use std::ops::{Sub, Add};
use std::path::Path;
use std::io;
use std::time::Instant;
use std::env::args;

const OUTPUT_FOLDER: &str = "./images/output";
const INPUT_FOLDER: &str = "./images/input";

/// The default brightness of empty space.
const DEFAULT_BRIGHT: f64 = 0.1;

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
    let mut vertex_coords = Vec::new();
    let mut faces = Vec::new();

    if let Ok(lines) = read_lines(input_path) {
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
    } else {
        println!("Failed to parse path. Did you enter a valid filename?");
        return
    }

    // println!("{:?}", vertex_coords);
    // println!("{:?}", faces);

    let get_3d_vect = |coord: &Vec<f64>| -> Vector {
        Vector {
            dx: coord[0],
            dy: coord[1],
            dz: coord[2],
        }
    };

    let get_vertices = |face: &Vec<usize>| -> Triangle {
        Triangle {
            v1: get_3d_vect(&vertex_coords[face[0]]),
            v2: get_3d_vect(&vertex_coords[face[1]]),
            v3: get_3d_vect(&vertex_coords[face[2]]),
        }
    };

    let triangles: Vec<Triangle> = faces
        .iter()
        .map(get_vertices)
        .collect();

    println!("num triangles: {:?}", triangles.len());

    let mut scene = Scene::new(triangles);

    let now = Instant::now();
    let mut prev_elapsed = 0.0;
    for i in 0..num_output_images {
        let f = File::create(format!("{}/{}_{}.ppm", OUTPUT_FOLDER, input_filename, i))
            .expect("Unable to create file");
        let f = BufWriter::new(f);

        scene.render_to_output(f);

        scene.camera.pos.dz -= 1.0;
        // scene.camera.pitch.0 += 0.1;

        let elapsed = now.elapsed().as_secs_f64();
        println!("Finished image {} in {:.3} s", i, elapsed - prev_elapsed);
        prev_elapsed = elapsed;
    }
    println!("Time to render images: {:.3} s", now.elapsed().as_secs_f64());

}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// A triangle represented with its 3 vertices.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Triangle {
    v1: Vector,
    v2: Vector,
    v3: Vector,
}

impl Triangle {
    fn new(v1: Vector, v2: Vector, v3: Vector) -> Triangle {
        Triangle { v1, v2, v3 }
    }

    /// Give the plane that the triangle intersects.
    fn plane(&self) -> Plane {
        // Calculate normal vector from cross product of two sides
        let cp = {
            let v1 = self.v1 - self.v2;
            let v2 = self.v1 - self.v3;
            v1.cross_product(v2)
        };

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
    /// Determine if a vector intersects this bounding box.
    #[inline]
    fn fast_intersect_check(&self, v: Vector) -> bool {
        v.dx >= self.min_x && v.dx <= self.max_x
        && v.dy >= self.min_y && v.dy <= self.max_y
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
            Triangle::new(v8, v4, v7).plane(), // right plane
            Triangle::new(v2, v6, v5).plane(), // left plane
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
    #[inline]
    fn intersection(&self, origin: Vector, direction: Vector) -> Vector {
        // deconstruct plane
        let (a,b,c,k) = (self.a, self.b, self.c, self.k);
        let abc_vect = Vector::new(a,b,c);
        let lambda = -(abc_vect.dot_product(origin) + k) / abc_vect.dot_product(direction);
        direction.scale(lambda) + origin
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

/// A set of triangles together make up a scene that can be viewed by a camera.
struct Scene {
    camera: Camera,
    triangles: Vec<Triangle>,
    triangle_planes: Vec<Plane>,
    bounding_boxes: Vec<BoundingBox>,

    /// All triangles in the scene fall within this bounding box.
    scene_bounding_box: BoundingBox,
    /// The six planes of the bounding box.
    box_planes: [Plane; 6],
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

        Scene {
            camera: Camera::new(),
            triangles,
            triangle_planes,
            bounding_boxes,
            scene_bounding_box,
            box_planes,
        }
    }

    /// Generate an image of the given scene.
    fn iterate_over_rays(&self) -> Vec<f64> {
        self.camera.iterate_over_rays(&self)
    }

    fn render_to_output(&self, mut writer: BufWriter<File>) {
        let pixels = self.iterate_over_rays();

        let num_cols = self.camera.view_plane.res_width;
        let num_rows = self.camera.view_plane.res_height;
        writer.write(format!("P3\n{} {}\n255\n", num_cols, num_rows).as_bytes()).unwrap();
        for p in pixels {
            let p = (p * 255.0) as isize;
            let r = p;
            let g = p;
            let b = p;
            let s = format!("{} {} {}\n", r, g, b);
            writer.write(s.as_bytes()).unwrap();
        }
        writer.flush().unwrap();
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
        // let pos = Vector::new(7.35889, -6.92579 , 4.95831);
        // let pitch = Radian(1.104793);
        // let yaw = Radian(0.8150688);

        let pos = Vector::new(0.1, 0.0, -10.0);
        let pitch = Radian(0.0);
        let yaw = Radian(0.0);
        let view_plane = ViewPlane {
            pixel_size: 0.005,
            res_height: 200,
            res_width: 200,
        };

        Camera { pos, pitch, yaw, view_plane }
    }

    /// Given a list of triangles and their corresponding planes and bounding boxes,
    /// calculate the rendering of a scene.
    fn iterate_over_rays(&self, scene: &Scene) -> Vec<f64> {

        let vp = &self.view_plane;
        let res_height = vp.res_height as isize;
        let res_width = vp.res_width as isize;

        // Get index of center pixel
        let i_center = ((res_width - 1) / 2) as isize;
        let j_center = ((res_height - 1) / 2) as isize;

        let mut pixel_brightness = Vec::with_capacity(vp.res_height * vp.res_width);

        for j in (0isize..res_height).rev() {
            for i in 0isize..res_width {
                // The direction of the ray denoted by m.
                // The origin is the camera position.
                let m = Vector {
                    dx: vp.pixel_size * ((i - i_center) as f64),
                    dy: vp.pixel_size * ((j - j_center) as f64),
                    dz: 1.0,
                }.yaw(self.yaw).pitch(self.pitch);

                

                let get_intersection = |p: &Plane| { p.intersection(self.pos, m) };
                
                let intersects_bounding_box = scene.box_planes.iter()
                    .map(get_intersection)
                    .map(|v| scene.scene_bounding_box.fast_intersect_check(v))
                    .any(|x| x);
                    

                if intersects_bounding_box {
                    // initialize distance of triangle to infinity
                    let mut min_dist_sq = f64::INFINITY;
    
                    // filter for all triangles that the ray intersects
                    for (plane, (bounding_box, t)) in scene.triangle_planes.iter().zip(scene.bounding_boxes.iter().zip(scene.triangles.iter())) {
    
                        let intersect = get_intersection(plane);
    
                        // squared distance from origin
                        let new_dist_sq = intersect.squared_magnitude();
    
                        if new_dist_sq < min_dist_sq // only look at closest triangle (no transparent triangles)
                            && bounding_box.fast_intersect_check(intersect) // fast initial check
                            && intersect.slow_intersect_check(t) { // final accurate check
                            min_dist_sq = new_dist_sq;
                        }
                    }
    
                    // brightness of pixel corresponds to how far away shape is
                    pixel_brightness.push(if min_dist_sq == f64::INFINITY {
                            DEFAULT_BRIGHT
                        } else {
                            1.0 / min_dist_sq.sqrt()
                        });
                } else {
                    pixel_brightness.push(DEFAULT_BRIGHT);
                }
            }
        }
        pixel_brightness
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
    fn new(dx: f64, dy: f64, dz: f64) -> Vector {
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