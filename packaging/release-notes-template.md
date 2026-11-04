## Tracer {{VERSION}}

### What's new

- _(fill in before publishing)_

### Install

Download **Tracer_{{VERSION}}_x64-setup.exe** (per-user installer, no admin prompt) or the `.msi`. Windows 10/11, 64-bit.

{{UNSIGNED_NOTE}}

### The detection models

The two detection models (~1.2 GB: desklib from the `model-v1` release, rasbt from `model-v2`) are not in the installer. They download once on first run and are SHA-256 verified. After that Tracer works fully offline. An install that already has `model-v1` only downloads the ~305 MB of `model-v2`.

### Checksums

`SHA256SUMS.txt` lists the SHA-256 of each installer.

### Licence

Tracer is MIT licensed. Third-party components and the model are covered in [THIRD-PARTY-NOTICES.md](https://github.com/garethllew/tracerapp/blob/main/THIRD-PARTY-NOTICES.md).
