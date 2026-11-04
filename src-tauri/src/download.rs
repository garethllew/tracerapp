//! First-run model download: the only network access Tracer ever makes.
//!
//! Files come from the `model-v1` release on github.com/garethllew/tracerapp
//! and are pinned by exact size + SHA-256. Each streams to `<file>.part`,
//! hashed as it arrives, and is renamed into place only once verified, so
//! nothing unverified ever sits at the real path (Vixer's models.rs rule).
//! Unlike Vixer, an interrupted transfer keeps its `.part` and the retry
//! resumes with an HTTP Range request: 838 MB is a lot to restart. A size or
//! checksum mismatch discards the partial file.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use crate::engine::{self, SharedEngine};

pub const MODEL_NAME: &str = "desklib/ai-text-detector-v1.01 (INT8 ONNX)";
pub const MODEL_VERSION: &str = "model-v1";
const BASE_URL: &str = "https://github.com/garethllew/tracerapp/releases/download/model-v1";

pub struct ModelFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// Small file first so a bad connection fails fast.
pub const FILES: [ModelFile; 2] = [
    ModelFile {
        name: engine::TOKENIZER_FILE,
        size: 8_656_624,
        sha256: "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    },
    ModelFile {
        name: engine::MODEL_FILE,
        size: 838_357_272,
        sha256: "de1368a264bd8cfe9f6b72291da3db56191f645e4568f8f037cea0790d380b0b",
    },
];

pub fn total_size() -> u64 {
    FILES.iter().map(|f| f.size).sum()
}

/// Where downloads go: `%APPDATA%\com.tracer.app\models`.
pub fn models_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf> {
    Ok(app.path().app_data_dir().context("no app data dir")?.join("models"))
}

/// One download at a time (a double-click on Retry mustn't start two).
#[derive(Default)]
pub struct DownloadState(AtomicBool);

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Progress {
    /// "downloading" | "verifying" | "done"
    phase: &'static str,
    received: u64,
    total: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    name: &'static str,
    version: &'static str,
    size_bytes: u64,
    installed: bool,
    dir: Option<String>,
}

#[tauri::command]
pub fn model_info<R: Runtime>(app: AppHandle<R>, engine: State<'_, SharedEngine>) -> ModelInfo {
    let dir = engine::installed_dir(&engine);
    ModelInfo {
        name: MODEL_NAME,
        version: MODEL_VERSION,
        size_bytes: total_size(),
        installed: dir.is_some(),
        dir: dir.or_else(|| models_dir(&app).ok()).map(|d| d.display().to_string()),
    }
}

/// Download (or, with `force`, re-download) the model, emitting
/// `download-progress`. Safe to call again after a failure: it resumes.
#[tauri::command]
pub async fn download_model<R: Runtime>(
    app: AppHandle<R>,
    engine: State<'_, SharedEngine>,
    state: State<'_, DownloadState>,
    force: bool,
) -> Result<(), String> {
    if state.0.swap(true, Ordering::SeqCst) {
        return Err("A download is already running.".into());
    }
    if force {
        // Free the files before they're replaced (Windows won't rename over
        // a file a live session might still hold).
        engine::unload(&app, &engine);
    }
    let result = run_download(&app, force).await;
    state.0.store(false, Ordering::SeqCst);
    match result {
        Ok(dir) => {
            engine::model_installed(&app, &engine, dir);
            Ok(())
        }
        Err(e) => {
            log::warn!("model download failed: {e:#}");
            Err(friendly(&e))
        }
    }
}

async fn run_download<R: Runtime>(app: &AppHandle<R>, force: bool) -> Result<PathBuf> {
    let dir = models_dir(app)?;
    std::fs::create_dir_all(&dir).context("create models folder")?;
    let total = total_size();
    let emit = |phase, received| {
        let _ = app.emit("download-progress", Progress { phase, received, total });
    };
    let t = Instant::now();
    let mut done_before = 0u64;
    for f in &FILES {
        let dest = dir.join(f.name);
        if force {
            let _ = std::fs::remove_file(part_path(&dest));
        } else if dest.is_file() {
            emit("verifying", done_before);
            if verify_file(&dest, f.size, f.sha256).is_ok() {
                done_before += f.size;
                continue;
            }
            log::warn!("{} failed verification; downloading again", f.name);
        }
        let base = done_before;
        download_resumable(&format!("{BASE_URL}/{}", f.name), &dest, f.size, f.sha256, &|received| {
            emit("downloading", base + received)
        })
        .await
        .with_context(|| format!("downloading {}", f.name))?;
        done_before += f.size;
    }
    emit("done", total);
    log::info!("model download finished in {:.0} s", t.elapsed().as_secs_f32());
    Ok(dir)
}

fn friendly(e: &anyhow::Error) -> String {
    let s = format!("{e:#}");
    if s.contains("checksum") || s.contains("larger than") {
        "The downloaded file didn't match the expected checksum, so it was discarded. Please try again.".into()
    } else if s.contains("No space") || s.contains("os error 112") {
        "Not enough disk space: the model needs about 850 MB.".into()
    } else {
        "Download interrupted. Check your connection and press Retry: it continues where it left off.".into()
    }
}

pub(crate) fn part_path(dest: &Path) -> PathBuf {
    let mut p = dest.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

pub(crate) fn verify_file(path: &Path, size: u64, sha256: &str) -> Result<()> {
    let meta = std::fs::metadata(path)?;
    if meta.len() != size {
        bail!("size {} != {size}", meta.len());
    }
    let mut h = Sha256::new();
    hash_file_into(path, &mut h)?;
    let got = hex::encode(h.finalize());
    if got != sha256 {
        bail!("checksum mismatch");
    }
    Ok(())
}

fn hash_file_into(path: &Path, h: &mut Sha256) -> Result<u64> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let r = f.read(&mut buf)?;
        if r == 0 {
            return Ok(n);
        }
        h.update(&buf[..r]);
        n += r as u64;
    }
}

/// Stream `url` into `<dest>.part`, resuming an existing partial file with a
/// Range request, then verify exact size + SHA-256 and rename into place.
/// `progress(received)` fires every ~200 ms and at the end. Network errors
/// keep the partial file (resumable); size/checksum failures delete it.
pub(crate) async fn download_resumable(
    url: &str,
    dest: &Path,
    expected_size: u64,
    expected_sha256: &str,
    progress: &(dyn Fn(u64) + Send + Sync),
) -> Result<()> {
    let part = part_path(dest);
    let mut hasher = Sha256::new();
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > expected_size {
        let _ = std::fs::remove_file(&part);
        have = 0;
    }
    if have > 0 {
        have = hash_file_into(&part, &mut hasher).context("read partial download")?;
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60)) // per read, so slow-but-moving is fine
        .build()
        .context("build download client")?;
    let mut req = client.get(url);
    if have > 0 && have < expected_size {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let mut resp = req.send().await.context("download request")?.error_for_status().context("download request")?;

    let resumed = have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let mut file = if resumed {
        log::info!("resuming download at {have} bytes");
        std::fs::OpenOptions::new().append(true).open(&part).context("open partial download")?
    } else {
        // Fresh start (no partial, already complete, or the server ignored Range).
        hasher = Sha256::new();
        have = 0;
        std::fs::File::create(&part).context("create download file")?
    };
    let mut file_buf = std::io::BufWriter::with_capacity(1 << 20, &mut file);

    let mut received = have;
    let mut last = Instant::now();
    progress(received);
    while let Some(chunk) = resp.chunk().await.context("download interrupted")? {
        received += chunk.len() as u64;
        if received > expected_size {
            drop(file_buf);
            let _ = std::fs::remove_file(&part);
            bail!("download larger than the pinned size ({received} > {expected_size} bytes)");
        }
        file_buf.write_all(&chunk).context("write download")?;
        hasher.update(&chunk);
        if last.elapsed() >= Duration::from_millis(200) {
            last = Instant::now();
            progress(received);
        }
    }
    file_buf.flush().context("flush download")?;
    drop(file_buf);
    drop(file);
    if received != expected_size {
        // A clean early close: keep the partial file so Retry resumes.
        bail!("download incomplete: {received} of {expected_size} bytes");
    }
    let digest = hex::encode(hasher.finalize());
    if digest != expected_sha256 {
        let _ = std::fs::remove_file(&part);
        bail!("checksum mismatch (got sha256:{digest})");
    }
    std::fs::rename(&part, dest).context("finalize model file")?;
    progress(received);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Arc;

    /// Minimal HTTP/1.1 server for one file. Honours `Range: bytes=N-` with a
    /// 206; `cut_after` closes the connection early on the first request to
    /// simulate a dropped download.
    fn serve(body: Vec<u8>, cut_after: Option<usize>, honour_range: bool) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = Arc::new(body);
        std::thread::spawn(move || {
            let mut first = true;
            for stream in listener.incoming() {
                let mut s = stream.unwrap();
                let mut reader = BufReader::new(s.try_clone().unwrap());
                let mut start = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        start = v.trim().trim_end_matches('-').parse().unwrap();
                    }
                }
                let start = if honour_range { start } else { 0 };
                let slice = &body[start..];
                let status = if start > 0 { "206 Partial Content" } else { "200 OK" };
                let head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", slice.len());
                s.write_all(head.as_bytes()).unwrap();
                let n = if first { cut_after.unwrap_or(slice.len()).min(slice.len()) } else { slice.len() };
                let _ = s.write_all(&slice[..n]);
                first = false;
            }
        });
        format!("http://{addr}/file")
    }

    fn body(n: usize) -> (Vec<u8>, String) {
        let b: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
        let sha = hex::encode(Sha256::digest(&b));
        (b, sha)
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
    }

    #[test]
    fn downloads_and_verifies() {
        let (b, sha) = body(300_000);
        let url = serve(b.clone(), None, true);
        let dir = tempdir();
        let dest = dir.join("m.onnx");
        rt().block_on(download_resumable(&url, &dest, b.len() as u64, &sha, &|_| {})).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b);
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn interrupted_download_resumes_with_range() {
        let (b, sha) = body(500_000);
        let url = serve(b.clone(), Some(123_457), true);
        let dir = tempdir();
        let dest = dir.join("m.onnx");
        let first = rt().block_on(download_resumable(&url, &dest, b.len() as u64, &sha, &|_| {}));
        assert!(first.is_err(), "cut connection must fail");
        assert_eq!(std::fs::metadata(part_path(&dest)).unwrap().len(), 123_457, "partial kept for resume");
        let seen = std::sync::Mutex::new(Vec::new());
        rt().block_on(download_resumable(&url, &dest, b.len() as u64, &sha, &|r| seen.lock().unwrap().push(r))).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b);
        assert_eq!(seen.lock().unwrap()[0], 123_457, "second attempt starts where the first stopped");
    }

    #[test]
    fn server_ignoring_range_restarts_cleanly() {
        let (b, sha) = body(200_000);
        let url = serve(b.clone(), Some(50_000), false);
        let dir = tempdir();
        let dest = dir.join("m.onnx");
        assert!(rt().block_on(download_resumable(&url, &dest, b.len() as u64, &sha, &|_| {})).is_err());
        rt().block_on(download_resumable(&url, &dest, b.len() as u64, &sha, &|_| {})).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b);
    }

    #[test]
    fn checksum_mismatch_discards_partial() {
        let (b, _) = body(100_000);
        let url = serve(b.clone(), None, true);
        let dir = tempdir();
        let dest = dir.join("m.onnx");
        let r = rt().block_on(download_resumable(&url, &dest, b.len() as u64, &"0".repeat(64), &|_| {}));
        assert!(format!("{:#}", r.unwrap_err()).contains("checksum"));
        assert!(!dest.exists() && !part_path(&dest).exists());
    }

    #[test]
    fn oversized_response_is_rejected() {
        let (b, sha) = body(100_000);
        let url = serve(b.clone(), None, true);
        let dir = tempdir();
        let dest = dir.join("m.onnx");
        let r = rt().block_on(download_resumable(&url, &dest, 50_000, &sha, &|_| {}));
        assert!(format!("{:#}", r.unwrap_err()).contains("larger"));
        assert!(!part_path(&dest).exists());
    }

    #[test]
    fn verify_file_checks_size_and_hash() {
        let (b, sha) = body(10_000);
        let dir = tempdir();
        let p = dir.join("f");
        std::fs::write(&p, &b).unwrap();
        assert!(verify_file(&p, b.len() as u64, &sha).is_ok());
        assert!(verify_file(&p, b.len() as u64 + 1, &sha).is_err());
        assert!(verify_file(&p, b.len() as u64, &"0".repeat(64)).is_err());
    }

    #[test]
    fn pinned_files_add_up() {
        assert_eq!(total_size(), 847_013_896);
        assert!(FILES.iter().all(|f| f.sha256.len() == 64));
    }

    fn tempdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("tracer-dl-test-{}-{}", std::process::id(), rand_suffix()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn rand_suffix() -> u128 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    }
}
