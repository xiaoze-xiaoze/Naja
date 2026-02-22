use ndarray::{Array1, Array2};

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005)
                              .wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }
}

pub fn generate_linear_data(
    n: usize,
    d: usize,
    noise: f64,
    seed: u64,
) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = SimpleRng::new(seed);

    let mut x_data = Vec::with_capacity(n * d);
    for _ in 0..n * d {
        x_data.push(rng.next_f64() * 2.0 - 1.0);
    }
    let x = Array2::from_shape_vec((n, d), x_data).unwrap();

    let mut coeffs = Vec::with_capacity(d);
    for i in 0..d {
        let c = if i % 3 == 0 { 0.0 } else { rng.next_f64() * 2.0 - 1.0 };
        coeffs.push(c);
    }
    let w = Array1::from_vec(coeffs);

    let mut y = x.dot(&w);
    for y_i in y.iter_mut() {
        *y_i += (rng.next_f64() - 0.5) * noise * 2.0;
    }

    (x, y, w)
}

#[allow(dead_code)]
fn main() {
    let (_x, _y, _w) = generate_linear_data(8, 3, 0.1, 42);
}
