//! Score the eval sample folder and print predicted vs expected.
//!
//!   cargo run --release --bin eval                     # table over eval/samples
//!   cargo run --release --bin eval -- --verbose        # + per-sentence scores for mixed samples
//!   cargo run --release --bin eval -- --bench          # 500-word latency
//!   cargo run --release --bin eval -- --raw            # whole-sample P(AI), for parity with Python
//!   cargo run --release --bin eval -- --dump FILE      # per-span scores (windows, paragraph groups, doc
//!                                                      #   chunks) as TSV, for offline rule sweeps
//!
//! Options: --samples DIR  --model-dir DIR  --model ensemble|max|FILE
//!            ensemble (default): desklib + rasbt, context gate + style lift (analyze_with)
//!            max: naive max of both per window (the old WIP baseline)
//!            FILE: one model, e.g. tracer-int8.onnx (desklib only)
//!          --no-gate  (context gate off: desklib only / max behave as before the ensemble rule)
//!          --threads N  --overlap N  --no-doc (no doc pass; implies --no-gate)
//!          --batch N  --budget TOKENS   (batching changes INT8 scores; see DetectorOptions)
//!
//! The model dir defaults to $TRACER_MODEL_DIR, then the app's model folder
//! (%APPDATA%\com.tracer.app\models), then tools/model-export/work/onnx.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use tracer_engine::text::{paragraph_ids, segment, span_text};
use tracer_engine::window::{build_doc_chunks, build_grouped_windows, build_groups};
use tracer_engine::window::build_group_chunks;
use tracer_engine::{analyze_with, models, Analysis, Detector, DetectorOptions, EngineConfig, Gate, MaxEnsemble, OnnxDetector, PieceKind};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

struct Args {
    samples: PathBuf,
    model_dir: PathBuf,
    model: String,
    threads: Option<usize>,
    batch: usize,
    overlap: Option<usize>,
    no_doc: bool,
    no_gate: bool,
    budget: Option<usize>,
    bench: bool,
    raw: bool,
    dump: Option<PathBuf>,
    verbose: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        samples: Path::new(ROOT).join("eval/samples"),
        model_dir: models::dev_model_dir(),
        model: "ensemble".into(),
        threads: None,
        batch: DetectorOptions::default().batch_size,
        overlap: None,
        no_doc: false,
        no_gate: false,
        budget: None,
        bench: false,
        raw: false,
        dump: None,
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
            "--no-gate" => a.no_gate = true,
            "--budget" => a.budget = Some(val()?.parse()?),
            "--bench" => a.bench = true,
            "--raw" => a.raw = true,
            "--dump" => a.dump = Some(val()?.into()),
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
        cfg.context_gate = 0.0;
    }
    if args.no_gate {
        cfg.context_gate = 0.0;
    }
    let mut opts = DetectorOptions { intra_threads: args.threads, batch_size: args.batch, ..Default::default() };
    if let Some(b) = args.budget {
        opts.max_batch_tokens = b;
    }
    let t = Instant::now();
    let (det, style): (Box<dyn Detector>, Option<Box<dyn Detector>>) = if args.model == "ensemble" || args.model == "max" {
        if !models::complete(&args.model_dir) {
            bail!("model set incomplete in {} (need {:?}; run tools/model-export, or pass --model-dir)", args.model_dir.display(), models::ALL_FILES);
        }
        let e = models::load_ensemble(&args.model_dir, &opts)?;
        if args.model == "max" {
            (Box::new(MaxEnsemble::new(vec![Box::new(e.desklib), Box::new(e.style)])), None)
        } else {
            (Box::new(e.desklib), Some(Box::new(e.style)))
        }
    } else {
        // A single model, e.g. --model tracer-int8.onnx or --model rasbt-bf16w.onnx.
        let model = args.model_dir.join(&args.model);
        let tok = if args.model.starts_with("rasbt") { models::STYLE_TOKENIZER } else { models::DESKLIB_TOKENIZER };
        if !model.exists() {
            bail!("model not found at {}", model.display());
        }
        (Box::new(OnnxDetector::load(&model, &args.model_dir.join(tok), &opts)?), None)
    };
    let det = det.as_ref();
    let style = style.as_deref();
    let threads = args.threads.unwrap_or_else(tracer_engine::detector::default_threads);
    println!(
        "model {} | gate {} | {} threads | batch {} | loaded in {:.2}s\n",
        args.model,
        if cfg.context_gate > 0.0 { format!("{}", cfg.context_gate) } else { "off".into() },
        threads,
        args.batch,
        t.elapsed().as_secs_f32()
    );

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
    if let Some(path) = &args.dump {
        if style.is_some() {
            bail!("--dump scores one model at a time: pass --model FILE");
        }
        return dump(det, &samples, &cfg, path);
    }
    if args.bench {
        return bench(det, style, &samples, &cfg);
    }

    println!("{:<34} {:>6} {:>4} {:>5} {:>8} {:>5} {:>5} {:>6} {:>7}  result", "sample", "exp", "exp%", "words", "gate", "AI%", "doc", "conf", "ms");
    let (mut ok_clear, mut n_clear, mut mixed_err, mut n_mixed, mut gated) = (0, 0, 0f32, 0, 0);
    // Tells per expected label: (samples, samples with any tell, total tells, samples with an artefact).
    let mut tell_stats: std::collections::BTreeMap<String, (usize, usize, usize, usize)> = Default::default();
    for s in &samples {
        let t = Instant::now();
        let a = analyze_with(&s.body, det, style, &cfg)?;
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
        let e = tell_stats.entry(s.expected.clone()).or_default();
        e.0 += 1;
        e.1 += (!a.tells.is_empty()) as usize;
        e.2 += a.tells.len();
        e.3 += a.tells.iter().any(|t| t.artefact) as usize;
        if args.verbose && !a.tells.is_empty() {
            let list: Vec<String> = a.tells.iter().map(|t| format!("{}x{}{}", t.id, t.count, if t.artefact { "!" } else { "" })).collect();
            println!("      tells: {}", list.join(", "));
        }
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
    for (exp, (n, any, total, art)) in &tell_stats {
        println!("tells on {exp} texts: {any}/{n} have any ({:.1} per text) | {art}/{n} have a scoring artefact", *total as f32 / (*n).max(1) as f32);
    }
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
    let art: Vec<&str> = a.tells.iter().filter(|t| t.artefact).map(|t| t.id.as_str()).collect();
    let art = (!art.is_empty()).then(|| format!("artefact: {}", art.join("+")));
    if let Some(x) = &art {
        f.push(x);
    }
    if f.is_empty() { String::new() } else { format!("  [{}]", f.join(", ")) }
}

/// Time analyze() on ~150/300/500/1000/2000-word texts built from the AI samples.
/// Each text is analysed 7 times after a warm-up; repeated shapes run a
/// little faster than fresh pastes, so treat these as best-case-ish.
fn bench(det: &dyn Detector, style: Option<&dyn Detector>, samples: &[Sample], cfg: &EngineConfig) -> Result<()> {
    let ai_text: String = samples.iter().filter(|s| s.expected == "ai").map(|s| s.body.as_str()).collect::<Vec<_>>().join("\n\n");
    let available = ai_text.split_whitespace().count();
    if available < 500 {
        bail!("not enough AI sample text for a 500-word bench");
    }
    for target in [150, 300, 500, 1000, 2000].into_iter().filter(|&t| t <= available) {
        let mut text = String::new();
        let mut n = 0;
        for word in ai_text.split_inclusive(char::is_whitespace) {
            if n >= target {
                break;
            }
            n += (!word.trim().is_empty()) as usize;
            text.push_str(word);
        }
        let a = analyze_with(&text, det, style, cfg)?; // warm-up
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            analyze_with(&text, det, style, cfg)?;
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

/// Per-span scores for offline combination-rule sweeps (tools/model-export/
/// sweep_rules.py). Mirrors analyze()'s spans: windows, paragraph groups
/// (chunked like the doc pass when long) and doc chunks. One TSV row per span:
///   sample  expected  expected_share  kind  start  end  words  p
/// plus one `sent` row per sample listing its sentence word counts.
fn dump(det: &dyn Detector, samples: &[Sample], cfg: &EngineConfig, path: &Path) -> Result<()> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for s in samples {
        let pieces = segment(&s.body, cfg);
        let prose: Vec<usize> = pieces.iter().enumerate().filter(|(_, p)| p.kind == PieceKind::Prose).map(|(i, _)| i).collect();
        let sw: Vec<usize> = prose.iter().map(|&i| pieces[i].words).collect();
        if sw.iter().sum::<usize>() < cfg.min_words || sw.iter().sum::<usize>() > cfg.max_scored_words {
            continue;
        }
        let para_of = paragraph_ids(&s.body, &pieces);
        let sp: Vec<usize> = prose.iter().map(|&i| para_of[i]).collect();
        let groups = build_groups(&sw, &sp, cfg);
        let mut spans: Vec<(&str, std::ops::Range<usize>)> = Vec::new();
        spans.extend(build_grouped_windows(&sw, &groups, cfg).into_iter().map(|r| ("win", r)));
        spans.extend(build_group_chunks(&sw, &groups, cfg).into_iter().map(|r| ("group", r)));
        spans.extend(build_doc_chunks(&sw, cfg).into_iter().map(|r| ("doc", r)));
        let texts: Vec<String> = spans.iter().map(|(_, r)| span_text(&s.body, &pieces, &prose[r.clone()])).collect();
        let probs = det.score(&texts)?;
        let sents = sw.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(",");
        writeln!(out, "{}\t{}\t{}\tsent\t0\t{}\t{}\t{sents}", s.name, s.expected, s.expected_share, sw.len(), sw.iter().sum::<usize>())?;
        for ((kind, r), p) in spans.iter().zip(probs) {
            writeln!(out, "{}\t{}\t{}\t{kind}\t{}\t{}\t{}\t{p:.6}", s.name, s.expected, s.expected_share, r.start, r.end, sw[r.clone()].iter().sum::<usize>())?;
        }
        eprint!(".");
    }
    std::fs::write(path, out)?;
    eprintln!("\nwrote {}", path.display());
    Ok(())
}
