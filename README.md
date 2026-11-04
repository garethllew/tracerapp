# Tracer

A local AI-text detector. Paste text, get a per-sentence read on how likely it is to be machine-written. Everything runs on your machine; pasted text is never logged or sent anywhere.

## Run

Prerequisites: Node 20+, a stable Rust toolchain, and the [Tauri 2 prerequisites](https://v2.tauri.app/start/prerequisites/) for Windows (WebView2, MSVC build tools).

```bash
npm install
npm run tauri dev          # desktop app (Vite dev server on port 1430)
cargo run --release --bin eval   # evaluation harness
```

The app looks for `tracer-int8.onnx` and `tokenizer.json` in, in order:

1. `$TRACER_MODEL_DIR`
2. the app data dir, `%APPDATA%\com.tracer.app\models`
3. debug builds only: `tools/model-export/work/onnx`
