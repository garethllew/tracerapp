# Third-party notices

Tracer is MIT-licensed (see `LICENSE`). It bundles or downloads the following.

## Detection model (downloaded on first run, not in the installer)

- **desklib/ai-text-detector-v1.01**, MIT License, © desklib.
  https://huggingface.co/desklib/ai-text-detector-v1.01
  Re-exported to ONNX and INT8-quantised for Tracer (`tools/model-export/`).
- **microsoft/deberta-v3-large** (base model), MIT License, © Microsoft Corporation.
  https://huggingface.co/microsoft/deberta-v3-large
- Training data: **RAID** (liamdugan/raid), MIT License, © Liam Dugan.

## Fonts (`public/fonts/`, bundled locally, no network fetch)

- **Syne**: SIL Open Font License 1.1, © Bonjour Monde / The Syne Project.
- **DM Mono**: SIL Open Font License 1.1, © Colophon Foundry / Google.

## Runtime libraries (selected)

- **ONNX Runtime**: MIT License, © Microsoft Corporation (via the `ort` crate, MIT/Apache-2.0, © pyke.io).
- **Hugging Face tokenizers**: Apache License 2.0.
- **Tauri**: MIT/Apache-2.0. **React**: MIT.

Full dependency licence lists can be generated with `cargo about` / `npx license-checker`.
