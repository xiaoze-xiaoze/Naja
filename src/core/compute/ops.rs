use ndarray::{s, Array1, Array2, Zip};
use crate::core::{Error, Result};
use super::types::{Matrix, MatrixView, Vector, VectorView};

pub fn all_finite_vec(v: VectorView<'_>) -> bool { v.iter().all(|x| x.is_finite()) }

pub fn all_finite_mat(m: MatrixView<'_>) -> bool { m.iter().all(|x| x.is_finite()) }

pub fn ensure_nonempty_vec(v: VectorView<'_>) -> Result<()> {
    if v.len() == 0 { return Err(Error::empty_input("v")); }
    Ok(())
}

pub fn ensure_nonempty_mat(m: MatrixView<'_>) -> Result<()> {
    if m.nrows() == 0 || m.ncols() == 0 { return Err(Error::empty_input("m")); }
    Ok(())
}

pub fn ensure_len(a: VectorView<'_>, b: VectorView<'_>, name_a: &str, name_b: &str) -> Result<()> {
    if a.len() != b.len() { return Err(Error::invalid_shape(format!("vector length mismatch: {name_a} has len {}, {name_b} has len {}", a.len(), b.len() ))); }
    Ok(())
}

pub fn ensure_matmul(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<()> {
    if a.ncols() != b.nrows() { return Err(Error::invalid_shape(format!("matmul dim mismatch: a is {}x{}, b is {}x{} (need a.ncols == b.nrows)", a.nrows(), a.ncols(), b.nrows(), b.ncols()))); }
    Ok(())
}

pub fn ensure_gemv(a: MatrixView<'_>, x: VectorView<'_>) -> Result<()> {
    if a.ncols() != x.len() { return Err(Error::invalid_shape(format!("gemv dim mismatch: a is {}x{}, x has len {} (need a.ncols == x.len)", a.nrows(), a.ncols(), x.len()))); }
    Ok(())
}

pub fn add_intercept(x: MatrixView<'_>) -> Result<Matrix> {
    ensure_nonempty_mat(x)?;
    let x = x.to_owned();
    let n = x.nrows();
    let p = x.ncols();
    let mut out = Array2::<f64>::zeros((n, p + 1));
    out.column_mut(0).fill(1.0);
    out.slice_mut(s![.., 1..]).assign(&x);
    Ok(out)
}

pub fn add_diag_mut(a: &mut Matrix, alpha: f64) -> Result<()> {
    if !alpha.is_finite() { return Err(Error::invalid_param("alpha", "alpha must be finite")); }
    ensure_nonempty_mat(a.view())?;
    if a.nrows() != a.ncols() { return Err(Error::invalid_shape(format!("add_diag_mut requires square matrix, got {}x{}", a.nrows(), a.ncols()))); }
    let n = a.nrows();
    for i in 0..n { a[(i, i)] += alpha; }
    Ok(())
}

pub fn dot(a: VectorView<'_>, b: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(a)?;
    ensure_len(a, b, "a", "b")?;
    Ok(a.dot(&b))
}

pub fn l2_sq(v: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(v)?;
    Ok(v.iter().map(|x| x * x).sum())
}

pub fn l2(v: VectorView<'_>) -> Result<f64> { Ok(l2_sq(v)?.sqrt()) }

pub fn add_scaled_mut(dst: &mut Vector, src: VectorView<'_>, alpha: f64) -> Result<()> {
    if !alpha.is_finite() {
        return Err(Error::invalid_param("alpha", "alpha must be finite"));
    }
    ensure_nonempty_vec(dst.view())?;
    ensure_len(dst.view(), src, "dst", "src")?;

    Zip::from(dst)
        .and(src)
        .for_each(|d, &s| *d += alpha * s);

    Ok(())
}

pub fn matmul(a: MatrixView<'_>, b: MatrixView<'_>) -> Result<Matrix> {
    ensure_nonempty_mat(a)?;
    ensure_nonempty_mat(b)?;
    ensure_matmul(a, b)?;
    Ok(a.dot(&b))
}

pub fn gemv(a: MatrixView<'_>, x: VectorView<'_>) -> Result<Vector> {
    ensure_nonempty_mat(a)?;
    ensure_nonempty_vec(x)?;
    ensure_gemv(a, x)?;
    Ok(a.dot(&x))
}

pub fn xtx(x: MatrixView<'_>) -> Result<Matrix> {
    ensure_nonempty_mat(x)?;
    Ok(x.t().dot(&x))
}

pub fn xty(x: MatrixView<'_>, y: VectorView<'_>) -> Result<Vector> {
    ensure_nonempty_mat(x)?;
    ensure_nonempty_vec(y)?;
    if x.nrows() != y.len() { return Err(Error::invalid_shape(format!("xty dim mismatch: x is {}x{}, y has len {} (need x.nrows == y.len)", x.nrows(), x.ncols(), y.len()))); }
    Ok(x.t().dot(&y))
}

pub fn col_mean(x: MatrixView<'_>) -> Result<Vector> {
    ensure_nonempty_mat(x)?;
    let n = x.nrows() as f64;
    let mut out = Array1::<f64>::zeros(x.ncols());
    for j in 0..x.ncols() {
        let s: f64 = x.column(j).iter().sum();
        out[j] = s / n;
    }
    Ok(out)
}

pub fn col_var(x: MatrixView<'_>, ddof: usize) -> Result<Vector> {
    ensure_nonempty_mat(x)?;
    let n = x.nrows();
    if ddof >= n { return Err(Error::invalid_param("ddof", format!("ddof must be < nrows (ddof={ddof}, nrows={n})"))); }
    let mean = col_mean(x)?;
    let denom = (n - ddof) as f64;
    let mut out = Array1::<f64>::zeros(x.ncols());
    for j in 0..x.ncols() {
        let mj = mean[j];
        let ss: f64 = x.column(j).iter().map(|&v| { let d = v - mj; d * d }).sum();
        out[j] = ss / denom;
    }
    Ok(out)
}

pub fn center_cols_mut(x: &mut Matrix, mean: VectorView<'_>) -> Result<()> {
    ensure_nonempty_mat(x.view())?;
    ensure_nonempty_vec(mean)?;
    if x.ncols() != mean.len() { return Err(Error::invalid_shape(format!("center_cols_mut dim mismatch: x has ncols {}, mean has len {}", x.ncols(), mean.len()))); }
    for j in 0..x.ncols() {
        let mj = mean[j];
        x.column_mut(j).mapv_inplace(|v| v - mj);
    }
    Ok(())
}

pub fn scale_cols_mut(x: &mut Matrix, scale: VectorView<'_>) -> Result<()> {
    ensure_nonempty_mat(x.view())?;
    ensure_nonempty_vec(scale)?;
    if x.ncols() != scale.len() { return Err(Error::invalid_shape(format!("scale_cols_mut dim mismatch: x has ncols {}, scale has len {}", x.ncols(), scale.len()))); }
    for j in 0..x.ncols() {
        let sj = scale[j];
        if !sj.is_finite() { return Err(Error::invalid_param("scale", format!("scale[{j}] must be finite"))); }
        if sj == 0.0 { return Err(Error::invalid_param("scale", format!("scale[{j}] must be nonzero"))); }
        x.column_mut(j).mapv_inplace(|v| v / sj);
    }
    Ok(())
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } 
    else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

pub fn log1pexp(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } 
    else { x.exp().ln_1p() }
}

pub fn logsumexp(v: VectorView<'_>) -> Result<f64> {
    ensure_nonempty_vec(v)?;
    if !all_finite_vec(v) { return Err(Error::non_finite("v")); }
    let m = v.iter().copied().fold(f64::NEG_INFINITY, |acc, x| acc.max(x));
    let sum: f64 = v.iter().map(|&x| (x - m).exp()).sum();
    Ok(m + sum.ln())
}

pub fn softmax_mut(v: &mut Vector) -> Result<()> {
    ensure_nonempty_vec(v.view())?;
    if !all_finite_vec(v.view()) { return Err(Error::non_finite("v")); }
    let lse = logsumexp(v.view())?;
    v.mapv_inplace(|x| (x - lse).exp());
    Ok(())
}

pub fn argmax(v: VectorView<'_>) -> Result<usize> {
    ensure_nonempty_vec(v)?;
    if !all_finite_vec(v) { return Err(Error::non_finite("v")); }
    let mut best_i = 0usize;
    let mut best_v = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    Ok(best_i)
}

#[cfg(feature = "faer-backend")]
fn to_faer_mat(a: MatrixView<'_>) -> faer::Mat<f64> { faer::Mat::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]) }

#[cfg(feature = "faer-backend")]
fn to_faer_col_mat(b: VectorView<'_>) -> faer::Mat<f64> { faer::Mat::from_fn(b.len(), 1, |i, _| b[i]) }

#[cfg(feature = "faer-backend")]
fn from_faer_col_mat(x: faer::Mat<f64>) -> Vector {
    let n = x.nrows();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n { out[i] = x.read(i, 0); }
    out
}

#[cfg(not(feature = "faer-backend"))]
pub fn solve_lu(_a: MatrixView<'_>, _b: VectorView<'_>) -> Result<Vector> { Err(Error::backend_unavailable("faer-backend", "enable feature `faer-backend` to use solve_lu")) }

#[cfg(not(feature = "faer-backend"))]
pub fn solve_cholesky(_a: MatrixView<'_>, _b: VectorView<'_>) -> Result<Vector> { Err(Error::backend_unavailable("faer-backend", "enable feature `faer-backend` to use solve_cholesky")) }

#[cfg(not(feature = "faer-backend"))]
pub fn solve_lstsq(_a: MatrixView<'_>, _b: VectorView<'_>) -> Result<Vector> { Err(Error::backend_unavailable("faer-backend", "enable feature `faer-backend` to use solve_lstsq")) }

#[cfg(feature = "faer-backend")]
pub fn solve_lu(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector> {
    ensure_nonempty_mat(a)?;
    ensure_nonempty_vec(b)?;
    if a.nrows() != a.ncols() { return Err(Error::invalid_shape(format!("solve_lu requires square matrix, got {}x{}", a.nrows(), a.ncols()))); }
    if a.nrows() != b.len() { return Err(Error::invalid_shape(format!("solve_lu dim mismatch: a is {}x{}, b has len {}", a.nrows(), a.ncols(), b.len()))); }
    let a_f = to_faer_mat(a);
    let b_f = to_faer_col_mat(b);
    let plu = a_f.partial_piv_lu();
    let x_f = plu.solve(&b_f);
    Ok(from_faer_col_mat(x_f))
}

#[cfg(feature = "faer-backend")]
pub fn solve_cholesky(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector> {
    ensure_nonempty_mat(a)?;
    ensure_nonempty_vec(b)?;
    if a.nrows() != a.ncols() { return Err(Error::invalid_shape(format!("solve_cholesky requires square matrix, got {}x{}", a.nrows(), a.ncols()))); }
    if a.nrows() != b.len() { return Err(Error::invalid_shape(format!("solve_cholesky dim mismatch: a is {}x{}, b has len {}", a.nrows(), a.ncols(), b.len()))); }
    let a_f = to_faer_mat(a);
    let b_f = to_faer_col_mat(b);
    let llt = a_f.llt(faer::Side::Lower).map_err(|_| Error::lin_alg("cholesky failed: matrix may not be SPD"))?;
    let x_f = llt.solve(&b_f);
    Ok(from_faer_col_mat(x_f))
}

#[cfg(feature = "faer-backend")]
pub fn solve_lstsq(a: MatrixView<'_>, b: VectorView<'_>) -> Result<Vector> {
    ensure_nonempty_mat(a)?;
    ensure_nonempty_vec(b)?;
    if a.nrows() != b.len() { return Err(Error::invalid_shape(format!("solve_lstsq dim mismatch: a is {}x{}, b has len {} (need a.nrows == b.len)", a.nrows(), a.ncols(), b.len()))); }
    if a.nrows() < a.ncols() { return Err(Error::invalid_shape(format!("solve_lstsq expects tall matrix (nrows >= ncols), got {}x{}", a.nrows(), a.ncols()))); }
    let a_f = to_faer_mat(a);
    let b_f = to_faer_col_mat(b);
    let qr = a_f.qr();
    let x_f = qr.solve_lstsq(&b_f);
    Ok(from_faer_col_mat(x_f))
}