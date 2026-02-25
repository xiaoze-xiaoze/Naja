# Model Export (ONNX)

## 概述

ONNX 导出模块面向模型跨平台部署场景，当前处于规划中。

## 模块位置

- `src/io/onnx.rs`

## 预期接口

*   **save_model**
    *   **签名**: 计划中
    *   **含义**: 将已拟合模型导出为 ONNX
    *   **用途**: 供 ONNX Runtime、TensorRT 等推理引擎使用
    *   **状态**: 计划中

## 使用示例

```rust
use naja::algorithms::linrg::LinearRegression;
use naja::io::onnx;
let model = LinearRegression::new();
let fitted = model.fit(&x_train, &y_train)?;
onnx::save_model(&fitted, "model.onnx")?;
```
