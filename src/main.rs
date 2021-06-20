use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::ops::Sub;

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

/// A point in 3D space.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Coordinate {
    x: f64,
    y: f64,
    z: f64,
}

impl Coordinate {
    fn new(x: f64, y: f64, z: f64) -> Coordinate {
        Coordinate { x, y, z }
    }

    /// Convert self into a vector
    fn vectorize(&self) -> Vector {
        Vector { dx: self.x, dy: self.y, dz: self.z, }
    }
}

/// Subtraction on a coordinate gives you a vector.
impl Sub for Coordinate {
    type Output = Vector;

    fn sub(self, other: Self) -> Vector {
        Vector {
            dx: self.x - other.x,
            dy: self.y - other.y,
            dz: self.z - other.z,
        }
    }
}

/// A triangle represented with its 3 vertices.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Triangle {
    v1: Coordinate,
    v2: Coordinate,
    v3: Coordinate,
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
            k: -cp.dot_product(self.v1.vectorize()),
        }
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
    pos: Coordinate,
    pitch: Radian,
    yaw: Radian,
    view_plane: ViewPlane,
}

impl Camera {
    /// Default camera.
    fn new() -> Camera {
        let pos = Coordinate::new(7.35889, -6.92579 , 4.95831);
        let pitch = Radian(1.104793);
        let yaw = Radian(0.8150688);
        let view_plane = ViewPlane {
            pixel_size: 0.0025,
            res_height: 200,
            res_width: 200,
        };

        Camera { pos, pitch, yaw, view_plane }
    }

    fn iterate_over_rays(&self) {
        let vp = &self.view_plane;
        let res_height = vp.res_height;
        let res_width = vp.res_width;

        // Get index of center pixel
        let i_center = (res_width - 1) / 2;
        let j_center = (res_height - 1) / 2;

        for i in 0..res_width {
            for j in 0..res_height {
                // direction of ray
                let m = Vector {
                        dx: vp.pixel_size * ((i - i_center) as f64),
                        dy: vp.pixel_size * ((j - j_center) as f64),
                        dz: 1.0,
                    }.yaw(self.yaw).pitch(self.pitch);

            }
        }
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