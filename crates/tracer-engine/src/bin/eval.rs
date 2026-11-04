//! Score the eval sample folder and print predicted vs expected.
//!
//!   cargo run --release --bin eval                     # table over eval/samples
//!   cargo run --release --bin eval -- --verbose        # + per-sentence scores for mixed samples
//!   cargo run --release --bin eval -- --bench          # 500-word latency
//!   cargo run --release --bin eval -- --raw            # whole-sample P(AI), for parity with Python
//!
//! Options: --samples DIR  --model-dir DIR  --model ensemble|FILE (default ensemble = desklib + rasbt, max)
//!          --threads N  --overlap N  --no-doc
//!          --batch N  --budget TOKENS   (batching changes INT8 scores; see DetectorOptions)
//!
//! The model dir defaults to $TRACER_MODEL_DIR, then the app's model folder
//! (%APPDATA%\com.tracer.app\models), then tools/model-export/work/onnx.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracer_engine::{analyze, models, Analysis, Detector, DetectorOptions, EngineConfig, Gate, OnnxDetector, PieceKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

struct Args {
    samples: PathBuf,
    model_dir: PathBuf,
    model: String,
    threads: Option<usize>,
    batch: usize,
    overlap: Option<usize>,
    no_doc: bool,
    budget: Option<usize>,
    bench: bool,
    raw: bool,
    verbose: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        samples: Path::new(ROOT).join("eval/samples"),
        model_dir: default_model_dir(),
        model: "ensemble".into(),
        threads: None,
        batch: DetectorOptions::default().batch_size,
        overlap: None,
        no_doc: false,
        budget: None,
        bench: false,
        raw: false,
        verbose: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--samples" => a.samples = val()?.into(),
            "--model-dir" => a.model_dir = val()?.into(),
            "--model" => a.model = val()?,
            "--threads" => a.threads = Some(val()?.parse()?),
            "--batch" => a.batch = val()?.parse()?,
            "--overlap" => a.overlap = Some(val()?.parse()?),
            "--no-doc" => a.no_doc = true,
            "--budget" => a.budget = Some(val()?.parse()?),
            "--bench" => a.bench = true,
            "--raw" => a.raw = true,
            "--verbose" | "-v" => a.verbose = true,
            "--help" | "-h" => {
                println!("{}", include_str!("eval.rs").lines().take_while(|l| l.starts_with("//!")).map(|l| l.trim_start_matches("//!")).collect::<Vec<_>>().join("\n"));
                std::process::exit(0);
            }
            other => bail!("unknown argument {other} (try --help)"),
        }
    }
    Ok(a)
}

fn default_model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("TRACER_MODEL_DIR") {
        return d.into();
    }
    if let Ok(appdata) = std::env::var("APPDATA") {
        let d = Path::new(&appdata).join("com.tracer.app").join("models");
        if models::complete(&d) {
            return d;
        }
    }
    Path::new(ROOT).join("tools/model-export/work/onnx")
}

struct Sample {
    name: String,
    expected: String,
    expected_share: f32,
    body: String,
}

fn load_samples(dir: &Path) -> Result<Vec<Sample>> {
    let mut files = Vec::new();
    collect(dir, &mut files)?;
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let raw = std::fs::read_to_string(&f)?.replace("\r\n", "\n");
        let mut parts = raw.splitn(3, "---");
        let (Some(_), Some(meta), Some(body)) = (parts.next(), parts.next(), parts.next()) else {
            eprintln!("skipping {} (no front matter)", f.display());
            continue;
        };
        let field = |k: &str| {
            meta.lines()
                .find_map(|l| l.split_once(':').filter(|(key, _)| key.trim() == k))
                .map(|(_, v)| v.split('#').next().unwrap_or("").trim().to_string())
                .unwrap_or_default()
        };
        out.push(Sample {
            name: f.strip_prefix(dir).unwrap_or(&f).to_string_lossy().replace('\\', "/"),
            expected: field("expected"),
            expected_share: field("expected_ai_share").parse().unwrap_or(0.0),
            body: body.trim().to_string(),
        });
    }
    Ok(out)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for e in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let p = e?.path();
        if p.is_dir() {
            collect(&p, out)?;
        } else if p.extension().is_some_and(|x| x == "txt") {
            out.push(p);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let mut cfg = EngineConfig::default();
    if let Some(o) = args.overlap {
        cfg.window_overlap_sentences = o;
    }
    if args.no_doc {
        cfg.doc_check_max_words = 0;
    }
    let mut opts = DetectorOptions { intra_threads: args.threads, batch_size: args.batch, ..Default::default() };
    if let Some(b) = args.budget {
        opts.max_batch_tokens = b;
    }
    let t = Instant::now();
    let det: Box<dyn Detector> = if args.model == "ensemble" {
        if !models::complete(&args.model_dir) {
            bail!("model set incomplete in {} (need {:?}; run tools/model-export, or pass --model-dir)", args.model_dir.display(), models::ALL_FILES);
        }
        Box::new(models::load_ensemble(&args.model_dir, &opts)?)
    } else {
        // A single model, e.g. --model tracer-int8.onnx or --model rasbt-bf16w.onnx.
        let model = args.model_dir.join(&args.model);
        let tok = if args.model.starts_with("rasbt") { models::STYLE_TOKENIZER } else { models::DESKLIB_TOKENIZER };
        if !model.exists() {
            bail!("model not found at {}", model.display());
        }
        Box::new(OnnxDetector::load(&model, &args.model_dir.join(tok), &opts)?)
    };
    let det = det.as_ref();
    let threads = args.threads.unwrap_or_else(tracer_engine::detector::default_threads);
    println!("model {} | {} threads | batch {} | loaded in {:.2}s\n", args.model, threads, args.batch, t.elapsed().as_secs_f32());

    let samples = load_samples(&args.samples)?;
    if args.raw {
        let mut total = 0.0;
        for s in &samples {
            let t = Instant::now();
            let p = det.score(std::slice::from_ref(&s.body))?[0];
            total += t.elapsed().as_secs_f32();
            println!("{}\t{:.6}", s.name, p);
        }
        eprintln!("raw: {} samples scored one at a time in {total:.2}s", samples.len());
        return Ok(());
    }
    if args.bench {
        return bench(det, &samples, &cfg);
    }

    println!("{:<34} {:>6} {:>4} {:>5} {:>8} {:>5} {:>5} {:>6} {:>7}  result", "sample", "exp", "exp%", "words", "gate", "AI%", "doc", "conf", "ms");
    let (mut ok_clear, mut n_clear, mut mixed_err, mut n_mixed, mut gated) = (0, 0, 0f32, 0, 0);
    for s in &samples {
        let t = Instant::now();
        let a = analyze(&s.body, det, &cfg)?;
        let ms = t.elapsed().as_secs_f32() * 1000.0;
        let verdict = verdict(s, &a);
        match (&a.gate, s.expected.as_str()) {
            (Gate::TooShort, _) => gated += 1,
            (_, "mixed") => {
                n_mixed += 1;
                mixed_err += (a.ai_share.unwrap_or(0.0) - s.expected_share).abs();
            }
            _ if !s.name.starts_with("edge/") => {
                n_clear += 1;
                ok_clear += verdict.starts_with("ok") as usize;
            }
            _ => {}
        }
        println!(
            "{:<34} {:>6} {:>4.0} {:>5} {:>8} {:>5} {:>5} {:>6} {:>7.0}  {}{}",
            s.name,
            s.expected,
            s.expected_share * 100.0,
            a.words,
            format!("{:?}", a.gate),
            a.ai_share.map_or("-".into(), |x| format!("{:.0}", x * 100.0)),
            a.doc_score.map_or("-".into(), |x| format!("{x:.2}")),
            a.confidence.map_or("-".into(), |c| format!("{c:?}")),
            ms,
            verdict,
            flags(&a)
        );
        if args.verbose && (s.expected == "mixed" || verdict.starts_with("MISS")) && a.gate != Gate::TooShort {
            for seg in a.segments.iter().filter(|g| g.kind == PieceKind::Prose) {
                let p = seg.p_ai.map_or("  -  ".into(), |p| format!("{p:.2}"));
                let text: String = seg.text.chars().take(90).collect::<String>().replace('\n', " ");
                println!("      {p}  {text}");
            }
        }
    }
    println!(
        "\nclear human/AI correct: {ok_clear}/{n_clear} | mixed mean |AI% error|: {:.0} pts over {n_mixed} | gated too-short: {gated}",
        if n_mixed > 0 { mixed_err / n_mixed as f32 * 100.0 } else { 0.0 }
    );
    Ok(())
}

fn verdict(s: &Sample, a: &Analysis) -> String {
    let Some(share) = a.ai_share else { return "gated".into() };
    let ok = match s.expected.as_str() {
        "human" => share < 0.5,
        "ai" => share >= 0.5,
        "mixed" => (share - s.expected_share).abs() <= 0.25,
        _ => return "?".into(),
    };
    if ok { "ok".into() } else { "MISS".into() }
}

fn flags(a: &Analysis) -> String {
    let mut f = Vec::new();
    if a.non_english {
        f.push("non-english");
    }
    if a.disagreement {
        f.push("disagree");
    }
    if a.truncated {
        f.push("truncated");
    }
    if f.is_empty() { String::new() } else { format!("  [{}]", f.join(", ")) }
}

/// Time analyze() on ~150/300/500/1000-word texts built from the AI samples.
/// Each text is analysed 7 times after a warm-up; repeated shapes run a
/// little faster than fresh pastes, so treat these as best-case-ish.
fn bench(det: &dyn Detector, samples: &[Sample], cfg: &EngineConfig) -> Result<()> {
    let ai_text: String = samples.iter().filter(|s| s.expected == "ai").map(|s| s.body.as_str()).collect::<Vec<_>>().join("\n\n");
    let available = ai_text.split_whitespace().count();
    if available < 500 {
        bail!("not enough AI sample text for a 500-word bench");
    }
    for target in [150, 300, 500, 1000].into_iter().filter(|&t| t <= available) {
        let mut text = String::new();
        let mut n = 0;
        for word in ai_text.split_inclusive(char::is_whitespace) {
            if n >= target {
                break;
            }
            n += (!word.trim().is_empty()) as usize;
            text.push_str(word);
        }
        let a = analyze(&text, det, cfg)?; // warm-up
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            analyze(&text, det, cfg)?;
            times.push(t.elapsed().as_secs_f32());
        }
        times.sort_by(f32::total_cmp);
        println!(
            "bench: {:>3} words, {:>2} chunks{} -> median {:.2}s (min {:.2}s, max {:.2}s)",
            a.words,
            a.chunks_scored,
            if a.doc_score.is_some() { " incl. doc pass" } else { "               " },
            times[times.len() / 2],
            times[0],
            times[times.len() - 1]
        );
    }
    Ok(())
}
