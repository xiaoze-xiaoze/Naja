use ndarray::{Array1, Array2, ArrayView1};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Penalty {
    None,
    Ridge { alpha: f64 },
    Lasso { alpha: f64 }
}

impl Default for Penalty {
    fn default() -> Self { Penalty::None }
}

impl Penalty {
    pub fn loss(&self, w: ArrayView1<f64>) -> f64 {
        match self {
            Penalty::None => 0.0,
            Penalty::Ridge { alpha } => 0.5 * alpha * w.dot(&w),
            Penalty::Lasso { alpha } => alpha * w.mapv(|x| x.abs()).sum()
        }
    }

    pub fn gradient(&self, w: ArrayView1<f64>) -> Array1<f64> {
        match self {
            Penalty::None => Array1::zeros(w.len()),
            Penalty::Ridge { alpha } => w.mapv(|x| alpha * x),
            Penalty::Lasso { alpha } => w.mapv(|x| alpha * x.signum())
        }
    }

    pub fn apply_l2(&self, xtx: &mut Array2<f64>, intercept: bool) {
        if let Penalty::Ridge { alpha } = self {
            let start_idx = if intercept { 1 } else { 0 };
            for i in start_idx..xtx.nrows() { xtx[[i, i]] += alpha; }
        }
    }

    pub fn apply_l1(&self, z: f64) -> f64 {
        match self {
            Penalty::Lasso { alpha } => {
                let gamma = *alpha;
                if z > gamma { z - gamma }
                else if z < -gamma { z + gamma }
                else { 0.0 }
            },
            _ => z,
        }
    }
}
