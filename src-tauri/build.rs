fn main() {
    // ort's prebuilt Windows ONNX Runtime is always compiled with the DirectML
    // GPU provider ("pyke libs always ship compiled with DirectML on Windows",
    // ort-sys build/static_link), so tracer.exe imports DirectML.dll even though
    // Tracer only ever runs on the CPU provider. Delay-loading it means Windows
    // never maps the DLL unless a DirectML function is called (never), so the
    // installer doesn't need to carry the 18.5 MB DLL.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") && target.contains("msvc") {
        println!("cargo:rustc-link-arg-bins=/DELAYLOAD:DirectML.dll");
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
    }
    tauri_build::build()
}
