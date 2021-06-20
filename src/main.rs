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
        let vect1 = self.v1 - self.v2;
        let vect2 = self.v1 - self.v3;
        unimplemented!();
    }
}

/// A plane in 3d space; all points satisfy the equation:
///     `ax + by + cz + k = 0`
///
struct Plane {
    a: f64,
    b: f64,
    c: f64,
    k: f64,
}

/// A camera, determined by position, pitch angle, and yaw angle.
/// TODO: roll angle?
struct Camera {
    pos: Coordinate,
    pitch: Radian,
    yaw: Radian,
}

/// For measuring angles.
struct Radian(f64);

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
}

#[test]
fn simple_cross_product() {
    let v1 = Vector::new(2.0, 5.0, 6.0);
    let v2 = Vector::new(-3.0, 14.0, 20.0);
    let cp = Vector::new(16.0, -58.0, 43.0);
    assert_eq!(v1.cross_product(v2), cp);
}