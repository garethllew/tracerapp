import { useEffect, useRef, useState } from "react";
import type { Settings, Theme } from "../lib/settings";
import { resumeHotkey, suspendHotkey } from "../lib/settings";
import { ModelInfo } from "./ModelInfo";

/* Settings view, after Vixer's SettingsPanel: replaces the main body, back
   button at the top, small uppercase section labels. */

interface Props {
  settings: Settings;
  /** Resolves null on success, or a message to show inline. */
  onChange: (patch: Partial<Settings>) => Promise<string | null>;
  /** Registration failure reported by the backend for the saved hotkey. */
  hotkeyError: string | null;
  onBack: () => void;
}

export function SettingsPanel({ settings, onChange, hotkeyError, onBack }: Props) {
  const [hkError, setHkError] = useState<string | null>(null);
  const [startupError, setStartupError] = useState<string | null>(null);

  return (
    <div className="settings">
      <div className="set-head">
        <button className="rb xs" onClick={onBack} title="Back (Esc)" aria-label="Back">‹ Back</button>
        <span className="label">Settings</span>
      </div>

      <div className="set-body">
        <section className="set-sec">
          <p className="label">Global hotkey</p>
          <p className="set-desc">Checks whatever text is on your clipboard, from any app.</p>
          <HotkeyInput
            value={settings.hotkey}
            error={hkError ?? hotkeyError}
            onCaptureStart={() => { setHkError(null); suspendHotkey().catch(() => {}); }}
            onCancel={() => { resumeHotkey().catch(() => {}); }}
            onCommit={async (combo) => {
              if (combo === settings.hotkey) { resumeHotkey().catch(() => {}); return; }
              setHkError(await onChange({ hotkey: combo }));
            }}
          />
        </section>

        <section className="set-sec">
          <div className="set-row">
            <div>
              <p className="label">Launch at startup</p>
              <p className="set-desc">Start in the tray when you sign in.</p>
            </div>
            <Toggle
              on={settings.launchAtStartup}
              label="Launch at startup"
              onChange={async (v) => setStartupError(await onChange({ launchAtStartup: v }))}
            />
          </div>
          {startupError && <p className="set-err">{startupError}</p>}
        </section>

        <section className="set-sec">
          <p className="label">Theme</p>
          <Segmented value={settings.theme} onChange={(theme) => { onChange({ theme }); }} />
        </section>

        <section className="set-sec">
          <p className="label">Model</p>
          <ModelInfo />
        </section>
      </div>
    </div>
  );
}

const THEMES: { id: Theme; label: string }[] = [
  { id: "system", label: "System" },
  { id: "dark", label: "Dark" },
  { id: "light", label: "Light" },
];

function Segmented({ value, onChange }: { value: Theme; onChange: (t: Theme) => void }) {
  return (
    <div className="segctl" role="radiogroup" aria-label="Theme">
      {THEMES.map((t) => (
        <button key={t.id} role="radio" aria-checked={value === t.id} className={value === t.id ? "on" : ""} onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* Vixer's Toggle look, with Tracer's violet for "on". */
function Toggle({ on, label, onChange }: { on: boolean; label: string; onChange: (v: boolean) => void }) {
  return (
    <button role="switch" aria-checked={on} aria-label={label} className={`toggle${on ? " on" : ""}`} onClick={() => onChange(!on)}>
      <span />
    </button>
  );
}

/** Press-to-capture field (Vixer's HotkeyInput): click, press the combo, it
 *  saves; Esc or clicking away cancels. Emits "Ctrl+Shift+H"-style strings,
 *  which is what the shortcut parser on the Rust side expects. */
function HotkeyInput({ value, error, onCaptureStart, onCancel, onCommit }: {
  value: string;
  error: string | null;
  onCaptureStart: () => void;
  onCancel: () => void;
  onCommit: (combo: string) => void;
}) {
  const [capturing, setCapturing] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const capturingRef = useRef(false);
  capturingRef.current = capturing;

  // Leaving the panel mid-capture must give the hotkey back.
  useEffect(() => () => { if (capturingRef.current) onCancel(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const start = () => { setCapturing(true); setHint(null); onCaptureStart(); };
  const cancel = () => { setCapturing(false); setHint(null); onCancel(); };

  const onKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>) => {
    if (!capturing) return;
    e.preventDefault();
    e.stopPropagation(); // keep Esc / Ctrl+Enter away from the window-level handlers
    if (e.code === "Escape") { cancel(); return; }
    if (/^(Control|Shift|Alt|Meta)/.test(e.code)) return; // wait for the main key
    const mods = [e.ctrlKey && "Ctrl", e.shiftKey && "Shift", e.altKey && "Alt", e.metaKey && "Super"].filter(Boolean) as string[];
    if (mods.length === 0) { setHint("Include Ctrl, Shift or Alt"); return; }
    let key = e.code; // already what the parser expects (Space, F5, ArrowUp, …)
    if (/^Key[A-Z]$/.test(key)) key = key.slice(3);
    else if (/^Digit[0-9]$/.test(key)) key = key.slice(5);
    setCapturing(false);
    setHint(null);
    onCommit([...mods, key].join("+"));
  };

  return (
    <div className="hk">
      <button
        className={`hk-btn${capturing ? " cap" : ""}${error && !capturing ? " bad" : ""}`}
        onClick={() => (capturing ? cancel() : start())}
        onKeyDown={onKeyDown}
        onBlur={() => { if (capturing) cancel(); }}
        title={capturing ? "Press the new key combination (Esc cancels)" : "Click to change"}
      >
        {capturing ? (
          <span className={`hk-prompt${hint ? " warn" : ""}`}>{hint ?? "Press new keys…"}</span>
        ) : (
          value.split("+").map((part, i) => (
            <span key={i} className="hk-part">
              {i > 0 && <span className="hk-plus">+</span>}
              <kbd>{part}</kbd>
            </span>
          ))
        )}
      </button>
      {error && !capturing && <p className="set-err">{error}</p>}
    </div>
  );
}
