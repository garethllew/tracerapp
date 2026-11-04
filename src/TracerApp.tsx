import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { analyzeText, engineStatus, onEngineStatus, showMainWindow } from "./lib/ipc";
import type { Analysis, EngineStatus } from "./lib/types";
import type { Side } from "./lib/score";
import { prefersReducedMotion } from "./lib/score";
import { TitleBar } from "./components/TitleBar";
import { ResultBar } from "./components/ResultBar";
import { HighlightedText } from "./components/HighlightedText";

/** Shimmer only when an analysis is slow enough to notice. */
const BUSY_AFTER_MS = 300;

export function TracerApp() {
  const [text, setText] = useState("");
  const [editing, setEditing] = useState(true);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analysedText, setAnalysedText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const [focus, setFocus] = useState<Side | null>(null);
  const [traceKey, setTraceKey] = useState(0);
  const [tracing, setTracing] = useState(false);

  const reqId = useRef(0);
  const pending = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const engineRef = useRef<EngineStatus | null>(null);
  engineRef.current = engine;

  useEffect(() => {
    // Subscribe first, then fetch: if the model finishes loading in between,
    // the fetch still sees it (fetching first could miss the only event).
    const un = onEngineStatus(setEngine);
    un.then(() => engineStatus()).then(setEngine).catch((e) => setError(String(e)));
    showMainWindow().catch(() => {});
    return () => { un.then((f) => f()); };
  }, []);

  const run = useCallback(async (t: string) => {
    if (!t.trim()) return;
    if (engineRef.current?.state !== "ready") {
      pending.current = t; // runs as soon as the model is loaded
      return;
    }
    const id = ++reqId.current;
    setError(null);
    const timer = window.setTimeout(() => { if (reqId.current === id) setBusy(true); }, BUSY_AFTER_MS);
    try {
      const a = await analyzeText(t);
      if (reqId.current !== id) return; // superseded or cleared
      setAnalysis(a);
      setAnalysedText(t);
      setFocus(null);
      setEditing(a.gate === "tooShort");
      setTraceKey((k) => k + 1);
      if (!prefersReducedMotion()) {
        setTracing(true);
        window.setTimeout(() => setTracing(false), 700);
      }
    } catch (e) {
      if (reqId.current === id) setError(String(e));
    } finally {
      window.clearTimeout(timer);
      if (reqId.current === id) setBusy(false);
    }
  }, []);

  // Model finished loading with a paste waiting.
  useEffect(() => {
    if (engine?.state === "ready" && pending.current) {
      const t = pending.current;
      pending.current = null;
      run(t);
    }
  }, [engine, run]);

  const clear = useCallback(() => {
    reqId.current++;
    pending.current = null;
    setText("");
    setAnalysis(null);
    setAnalysedText("");
    setError(null);
    setBusy(false);
    setFocus(null);
    setEditing(true);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, []);

  const edit = useCallback(() => {
    setEditing(true);
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, []);

  // Ctrl+Enter checks, Esc clears; pasting while viewing a result replaces the text.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(text); }
      else if (e.key === "Escape") { e.preventDefault(); clear(); }
    };
    const onPaste = (e: ClipboardEvent) => {
      if (editing) return; // the textarea handles its own paste
      const pasted = e.clipboardData?.getData("text/plain") ?? "";
      if (!pasted.trim()) return;
      e.preventDefault();
      setText(pasted);
      run(pasted);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("paste", onPaste);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("paste", onPaste); };
  }, [text, editing, run, clear]);

  const stale = editing && analysis !== null && text !== analysedText && analysis.gate !== "tooShort";
  const empty = !text && editing;

  return (
    <div className="app">
      <TitleBar />

      <ResultBar
        analysis={analysis}
        busy={busy}
        stale={stale}
        focus={focus}
        onFocus={setFocus}
        traceKey={traceKey}
      />

      <main className="panel">
        {editing || !analysis ? (
          <textarea
            ref={textareaRef}
            className="paste"
            value={text}
            placeholder="Paste text to check…"
            spellCheck={false}
            autoFocus
            onChange={(e) => setText(e.target.value)}
            onPaste={(e) => {
              const el = e.currentTarget;
              // Let the paste land, then check the whole box.
              window.setTimeout(() => run(el.value), 0);
            }}
          />
        ) : (
          <div className="scroll">
            <HighlightedText analysis={analysis} focus={focus} tracing={tracing} onClick={edit} />
          </div>
        )}

        {empty && <EmptyState engine={engine} />}
      </main>

      <footer className="foot">
        {error ? (
          <p className="footnote err">{error}</p>
        ) : engine && engine.state !== "ready" ? (
          <p className="footnote warn">{engineMessage(engine)}</p>
        ) : (
          <p className="footnote">A rough gauge, not proof. Short, formal or non-native-English writing can be misread as AI.</p>
        )}
        <div className="actions">
          <button className="rb xs" onClick={clear} disabled={!text && !analysis} title="Esc">Clear</button>
          <button className="rb xs a" onClick={() => run(text)} disabled={!text.trim() || (!stale && analysis !== null && !editing)} title="Ctrl+Enter">
            Check <kbd>Ctrl ↵</kbd>
          </button>
        </div>
      </footer>

      <div
        className="resize-grip"
        aria-hidden
        onMouseDown={(e) => {
          if (e.button !== 0) return;
          e.preventDefault();
          getCurrentWindow().startResizeDragging("South").catch(() => {});
        }}
      />
    </div>
  );
}

function engineMessage(s: EngineStatus): string {
  switch (s.state) {
    case "loading": return "Loading the detection model…";
    case "missing": return "Detection model not installed yet.";
    case "error": return `Model failed to load: ${s.message ?? "unknown error"}`;
    default: return "";
  }
}

function EmptyState({ engine }: { engine: EngineStatus | null }) {
  return (
    <div className="empty" aria-hidden>
      <p className="label">How it works</p>
      <p>
        Paste a Slack message, email, post or cover letter. Tracer estimates how much of it reads as
        AI-written, sentence by sentence, on this device. Nothing leaves your computer or is saved.
      </p>
      <p className="hint">
        <kbd>Ctrl</kbd> <kbd>Shift</kbd> <kbd>H</kbd> checks your clipboard from any app.
      </p>
      {engine?.state === "loading" && <p className="hint">Warming up the model…</p>}
    </div>
  );
}
