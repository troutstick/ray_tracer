use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::ops::{Sub, Add};

const IMAGES_FOLDER: &str = "./images";
const OUTPUT_FOLDER: &str = "./images/output";
const INPUT_FOLDER: &str = "./images/input";

/// Convert a 0 to 1 decimal input to a 0 to 255 integer output.
fn denormalize(f: f64) -> i32 {
    (f * 255.99) as i32
}

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
    create_dir_all(OUTPUT_FOLDER).unwrap();
    create_dir_all(INPUT_FOLDER).unwrap();



    // create test output
    let f = File::create(format!("{}/test.ppm", OUTPUT_FOLDER)).expect("Unable to create file");
    let f = BufWriter::new(f);
    write_ppm_file(f);
}

/// A triangle represented with its 3 vertices.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Triangle {
    v1: Vector,
    v2: Vector,
    v3: Vector,
}

impl Triangle {
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
    fn intersects(&self, v: Vector) -> bool {
        v.dx > self.min_x && v.dx < self.max_x
        && v.dy > self.min_y && v.dy < self.max_y
        && v.dz > self.min_z && v.dz < self.max_z
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

/// A view plane for a camera. The view plane is situated 1.0 units away from the camera.
struct ViewPlane {
    /// The height and width of each pixel.
    pixel_size: f64,
    /// How many pixels wide a view is.
    res_width: usize,
    /// How many pixels tall a view is.
    res_height: usize,
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
        let pos = Vector::new(7.35889, -6.92579 , 4.95831);
        let pitch = Radian(1.104793);
        let yaw = Radian(0.8150688);
        let view_plane = ViewPlane {
            pixel_size: 0.0025,
            res_height: 200,
            res_width: 200,
        };

        Camera { pos, pitch, yaw, view_plane }
    }

    fn iterate_over_rays(&self, triangles: &Vec<Triangle>) -> Vec<f64> {

        // Find the corresponding plane for every triangle
        let triangle_planes: Vec<Plane> = triangles
            .iter()
            .map(|t| t.plane())
            .collect();

        let bounding_boxes: Vec<BoundingBox> = triangles
                .iter()
                .map(|t| t.bounding_box())
                .collect();

        let vp = &self.view_plane;
        let res_height = vp.res_height;
        let res_width = vp.res_width;

        // Get index of center pixel
        let i_center = (res_width - 1) / 2;
        let j_center = (res_height - 1) / 2;

        let mut flattened_brightnesses = Vec::with_capacity(res_height * res_width);
        for i in 0..res_width {
            for j in 0..res_height {
                // direction of ray
                let m = Vector {
                    dx: vp.pixel_size * ((i - i_center) as f64),
                    dy: vp.pixel_size * ((j - j_center) as f64),
                    dz: 1.0,
                }.yaw(self.yaw).pitch(self.pitch);

                // initialize distance of triangle to infinity
                let mut min_dist_sq = f64::INFINITY;

                // filter for all triangles that the ray intersects
                for (plane, (bounding_box, t)) in triangle_planes.iter().zip(bounding_boxes.iter().zip(triangles.iter())) {

                    let intersect = {
                        // deconstruct plane
                        let (a,b,c,k) = (plane.a, plane.b, plane.c, plane.k);
                        let abc_vect = Vector::new(a,b,c);
                        let lambda = -(abc_vect.dot_product(self.pos) + k) / abc_vect.dot_product(m);
                        m.scale(lambda) + self.pos
                    };

                    // NOT actual magnitude
                    let new_dist_sq = intersect.squared_magnitude();

                    if new_dist_sq < min_dist_sq { // only look at closest triangle (no transparency)
                        if bounding_box.intersects(intersect) // fast initial check
                            && Vector::same_side(intersect, t.v1, t.v2, t.v3)
                            && Vector::same_side(intersect, t.v2, t.v1, t.v3)
                            && Vector::same_side(intersect, t.v3, t.v1, t.v2) {
                            min_dist_sq = new_dist_sq;
                        }
                    }
                }

                // add brightness to pixel
                flattened_brightnesses.push(
                    if min_dist_sq == f64::INFINITY {
                            0.0
                        } else {
                            1.0 / min_dist_sq.sqrt()
                        }
                );
            }
        }
        flattened_brightnesses
    }
}

/// For measuring angles.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Radian(f64);

impl Radian {
    fn sin(&self) -> f64 { self.0.sin() }
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
    fn cross_product(&self, other: Self) -> Self {
        Vector {
            dx: self.dy * other.dz - self.dz * other.dy,
            dy: self.dz * other.dx - self.dx * other.dz,
            dz: self.dx * other.dy - self.dy * other.dx,
        }
    }

    /// Calculate the dot product scalar.
    fn dot_product(&self, other: Self) -> f64 {
        self.dx * other.dx +
        self.dy * other.dy +
        self.dz * other.dz
    }

    /// Pitch the vector by r radians.
    fn pitch(&self, r: Radian) -> Vector {
        Vector {
            dx: self.dx,
            dy: self.dy * r.cos() - self.dz * r.sin(),
            dz: self.dy * r.sin() - self.dz * r.cos(),
        }
    }

    /// Yaw the vector by r radians.
    fn yaw(&self, r: Radian) -> Vector {
        Vector {
            dx: self.dx * r.cos() + self.dz * r.sin(),
            dy: self.dy,
            dz: -self.dx * r.sin() - self.dz * r.cos(),
        }
    }

    /// Scale the vector by some scalar value n.
    fn scale(&self, n: f64) -> Vector {
        Vector {
            dx: n * self.dx,
            dy: n * self.dy,
            dz: n * self.dz,
        }
    }

    fn squared_magnitude(&self) -> f64 {
        self.dx.powi(2) + self.dy.powi(2) + self.dz.powi(2)
    }

    fn magnitude(&self) -> f64 {
        self.squared_magnitude().sqrt()
    }

    /// Return true if p1 and p2 are both on the same side of v1 -> v2.
    fn same_side(p1: Vector, p2: Vector, v1: Vector, v2: Vector) -> bool {
        let v = v2 - v1;
        let a = v.cross_product(p1 - v2);
        let b = v.cross_product(p2 - v2);
        a.dot_product(b) > 0.0
    }
}

impl Sub for Vector {
    type Output = Self;

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