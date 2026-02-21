# core::error（错误与 Result）

## 概述

`core::error` 统一了库内错误类型与 `Result` 别名，保证跨模块的错误语义一致。

## 模块位置

- `src/core/error.rs`

## 类型定义

*   **Result**
    *   **签名**: `pub type Result<T> = core::result::Result<T, Error>`
    *   **含义**: 统一返回类型
    *   **用途**: 所有核心模块的标准返回值

*   **Error**
    *   **变体**:
        - `LinAlg(String)`
        - `InvalidShape(String)`
        - `InvalidParam { name: String, msg: String }`
        - `EmptyInput(String)`
        - `InvalidState(String)`
        - `BackendUnavailable { name: String, msg: String }`
    *   **用途**: 描述线性代数失败、形状不匹配、参数非法等常见错误

## 构造方法

*   **lin_alg**
    *   **签名**: `fn lin_alg(msg: impl Into<String>) -> Self`
    *   **含义**: 创建线性代数错误

*   **invalid_shape**
    *   **签名**: `fn invalid_shape(msg: impl Into<String>) -> Self`
    *   **含义**: 创建形状不匹配错误

*   **invalid_param**
    *   **签名**: `fn invalid_param(name: impl Into<String>, msg: impl Into<String>) -> Self`
    *   **含义**: 创建参数非法错误

*   **empty_input**
    *   **签名**: `fn empty_input(name: impl Into<String>) -> Self`
    *   **含义**: 创建空输入错误

*   **invalid_state**
    *   **签名**: `fn invalid_state(msg: impl Into<String>) -> Self`
    *   **含义**: 创建非法状态错误

*   **backend_unavailable**
    *   **签名**: `fn backend_unavailable(name: impl Into<String>, msg: impl Into<String>) -> Self`
    *   **含义**: 创建后端不可用错误
