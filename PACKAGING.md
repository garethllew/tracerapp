# Packaging Tracer for Distribution

This guide turns the Tracer source into Windows installers, explains how a release is cut, and covers the optional extras (code signing, winget, publishing a new model). Windows 10/11 x64 only for now. There is no auto-updater: people install the new version over the old one (or use `winget upgrade`).

---

## The publisher name

Installers and winget show a publisher. It is currently **Rotation Studio**, an assumption to be confirmed with the owner. If it changes, edit these places:

| File | Keys |
|---|---|
| `src-tauri/tauri.conf.json` | `bundle.publisher`, `bundle.copyright` |
| `packaging/winget/templates/RotationStudio.Tracer.locale.en-US.yaml` | `Publisher`, `Copyright` |

(The winget `PackageIdentifier` `RotationStudio.Tracer` also embeds it, and changing it later makes winget treat Tracer as a different package, so settle it before the first winget submission.)

---

## Building locally

Prerequisites are the same as for development (see the README): Node, a stable Rust toolchain, WebView2 and the MSVC build tools.

```powershell
cd <your-tracer-checkout>
npm ci
npm run tauri build
```

The first build takes several minutes (full release compile); later ones are faster. Close any running `tracer.exe` first, or Windows will refuse to overwrite it.

When it finishes you will see:

```
Finished 2 bundles at:
  target\release\bundle\nsis\Tracer_0.1.0_x64-setup.exe   (about 17 MB)
  target\release\bundle\msi\Tracer_0.1.0_x64_en-US.msi    (about 22 MB)
```

| File | Notes |
|---|---|
| `Tracer_x.y.z_x64-setup.exe` | NSIS installer. Per-user install (no admin prompt), into `%LOCALAPPDATA%\Tracer`. Recommended; this is what winget uses. |
| `Tracer_x.y.z_x64_en-US.msi` | MSI package, for IT deployment. Usually not needed. |

The app itself is `target\release\tracer.exe`, a single file. The NSIS installer is about 9 MB.

### Runtime DLLs

ONNX Runtime is linked statically into `tracer.exe`, so there is no `onnxruntime.dll`. The prebuilt runtime is always compiled with Microsoft's DirectML GPU provider, so the exe *imports* `DirectML.dll`. Tracer only uses the CPU, so `src-tauri/build.rs` **delay-loads** it (`/DELAYLOAD:DirectML.dll`): Windows never loads the DLL unless a DirectML function is called, and none ever is. The installers therefore don't ship it. (Verified: with no `DirectML.dll` next to the exe, a full check runs and the DLL never appears in the process's loaded modules.) The `DirectML.dll` that `ort` copies into `target\release` during builds is unused. **Don't add DirectML as a GPU option without revisiting this.**

The exe also needs the Microsoft Visual C++ runtime (`MSVCP140.dll`, `MSVCP140_1.dll`), which almost every Windows 10/11 machine already has.

### The detection model is not in the installer

The ~850 MB model is downloaded on first run from the `model-v1` GitHub release, checked against pinned SHA-256s, and stored in `%APPDATA%\com.tracer.app\models`. Installers stay small, and the user needs internet for that first run only.

---

## Cutting a release

1. **Bump the version in three places** (they must match; the release workflow fails fast otherwise):
   - `package.json` -> `"version"`
   - `src-tauri/tauri.conf.json` -> `"version"`
   - `src-tauri/Cargo.toml` -> `version` (then run `cargo check` so `Cargo.lock` updates; `Cargo.toml` at the repo root has a workspace version too, keep it in step)
2. Commit and push to `main`, and wait for the **CI** workflow to go green.
3. Tag and push the tag:
   ```powershell
   git tag v0.2.0
   git push origin v0.2.0
   ```
4. The **Release** workflow runs on the tag: it checks the versions, runs the tests, builds the installers, writes `SHA256SUMS.txt`, and creates a **draft** release with the `.exe`, `.msi` and checksums attached.
5. Open the draft on GitHub (Releases), fill in the "What's new" placeholder, check the attachments, and **Publish**.
6. Optionally update winget (below).

Release notes come from `packaging/release-notes-template.md`; edit that file to change the boilerplate.

You can also run the workflow by hand from the Actions tab (`workflow_dispatch`). That builds and uploads the installers as a workflow artifact but does not create a release (no tag).

---

## The "unsigned app" warning

Until code signing is switched on, builds are unsigned. Windows SmartScreen will show **"Windows protected your PC"** with publisher "Unknown". Tell people to click **More info**, then **Run anyway**. The release workflow adds this note to the release text and the job summary automatically for unsigned builds. `SHA256SUMS.txt` lets anyone verify the download.

---

## Enabling Azure signing (optional, costs money)

Azure Trusted Signing (now renamed **Azure Artifact Signing**) gives builds a real signature, which removes the "Unknown publisher" label and builds SmartScreen reputation. It has a **monthly fee**, so whether to turn it on is the owner's call. Nothing breaks without it: the workflow detects the missing secrets and builds unsigned.

Source of the approach: <https://v2.tauri.app/distribute/sign/windows/> (the Azure Artifact Signing section).

1. **Create the signing account in Azure.** In the Azure portal, create an *Artifact Signing* (Trusted Signing) account in a region near you. Note its **endpoint URL** (for example `https://wus2.codesigning.azure.net`) and the **account name**.
2. **Identity validation.** In the account, complete identity validation for the organisation (or individual). Microsoft verifies this and it can take days. You cannot sign until it is approved.
3. **Certificate profile.** Create a *Public Trust* certificate profile from the validated identity. Note its **profile name**.
4. **App registration.** In Microsoft Entra ID, create an App registration (for example "tracer-github-signing"). Note the **Application (client) ID** and **Directory (tenant) ID**, and create a **client secret** (copy the value now; it is shown once).
5. **Grant the role.** On the Artifact Signing account (Access control / IAM), assign the app registration the **Trusted Signing Certificate Profile Signer** role.
6. **Add to GitHub** (repo Settings -> Secrets and variables -> Actions):

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `AZURE_CLIENT_ID` | the app's client ID |
   | Secret | `AZURE_CLIENT_SECRET` | the client secret value |
   | Secret | `AZURE_TENANT_ID` | the directory (tenant) ID |
   | Variable | `TRUSTED_SIGNING_ENDPOINT` | the endpoint URL |
   | Variable | `TRUSTED_SIGNING_ACCOUNT` | the account name |
   | Variable | `TRUSTED_SIGNING_PROFILE` | the certificate profile name |

7. Push a tag as usual. When all six are present, the workflow installs `artifact-signing-cli`, writes `src-tauri/tauri.signing.conf.json` at build time with this `signCommand`, builds with it, and then checks with `Get-AuthenticodeSignature` that `tracer.exe` and both installers are `Valid`:

   ```
   artifact-signing-cli -e <endpoint> -a <account> -c <profile> -d Tracer %1
   ```

Note: the Tauri docs now call the tool `artifact-signing-cli`; the older `trusted-signing-cli` crate is deprecated in its favour. The flags and the `AZURE_*` environment variables are the same. The workflow variables keep the `TRUSTED_SIGNING_*` names.

---

## winget

The manifest templates live in `packaging/winget/templates/` (package id `RotationStudio.Tracer`, NSIS installer, per-user, x64). Submitting to the community repository is a **manual, owner-run step** after a release is **published** (winget needs the public download URL).

1. Generate the manifests (downloads the published `.exe` and computes its SHA-256):
   ```powershell
   ./packaging/winget/update-manifest.ps1 -Version 0.2.0
   ```
   This writes `packaging/winget/manifests/0.2.0/` (not committed by default).
2. Validate: `winget validate --manifest packaging\winget\manifests\0.2.0`
3. Optionally test locally (needs admin to enable once): `winget settings --enable LocalManifestFiles`, then `winget install --manifest packaging\winget\manifests\0.2.0`.
4. Submit with [wingetcreate](https://github.com/microsoft/winget-create) (`winget install wingetcreate`):
   ```powershell
   wingetcreate submit packaging\winget\manifests\0.2.0
   ```
   This opens a pull request against `microsoft/winget-pkgs`; their bots validate it. The first submission gets human review; later versions are usually automatic.

The manifests use schema `ManifestVersion` 1.28.0 (the latest in `microsoft/winget-pkgs` when this was written).

---

## Publishing a new model

The app only knows the model by the pinned names, sizes and SHA-256s in `src-tauri/src/download.rs`. To ship a new model:

1. Re-run the export in `tools/model-export` to produce the new `tracer-int8.onnx` and `tokenizer.json`.
2. Create a GitHub release named **`model-vN`** (for example `model-v2`) and upload both files as assets. Keep old model releases up: older installed app versions still point at them.
3. Compute size and SHA-256 for each file:
   ```powershell
   (Get-Item tracer-int8.onnx).Length
   (Get-FileHash tracer-int8.onnx -Algorithm SHA256).Hash.ToLower()
   ```
4. In `src-tauri/src/download.rs` update:
   - `MODEL_VERSION` (`"model-vN"`)
   - `BASE_URL` (the `.../releases/download/model-vN` path)
   - the `size` and `sha256` of each entry in `FILES`
5. Run `cargo test -p tracer --lib`, then do a real first-run download on a machine with no model to confirm it verifies.
6. Mention the model change in the app release notes, and release the app as above.

---

## If the build fails

| Error | Fix |
|---|---|
| `linker 'link.exe' not found` | Install or repair the Visual Studio Build Tools (C++ workload). |
| `failed to remove ... tracer.exe` / access denied | A running Tracer is locking the file. Quit it (tray icon -> Quit) or end the `tracer.exe` process. |
| Release workflow: "Version mismatch" | `package.json`, `tauri.conf.json` and `src-tauri/Cargo.toml` do not all equal the tag. Fix, re-tag. |
