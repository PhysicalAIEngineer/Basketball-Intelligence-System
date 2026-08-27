import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Download,
  FileVideo,
  Gauge,
  Layers3,
  Maximize2,
  Play,
  RotateCcw,
  Search,
  Shield,
  Sparkles,
  Target,
  Timer,
  Upload,
  Users,
  Video,
  X,
  Zap
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const demoStats = {
  frames: 194,
  fps: 24,
  players: 10,
  ballDetected: 93,
  teamAccuracy: 95.2,
  trackingIdf1: 88.7,
  trackingHota: 76.4,
  courtMae: 0.72,
  speedMae: 1.34,
  possessionF1: 91.1,
  passPrecision: 87.5,
  passRecall: 83.3,
  interceptionPrecision: 85.7,
  interceptionRecall: 75.0
};

const demoPlayers = [
  { id: 1, team: "Team 1", speed: 18.6, distance: 42.8, possession: false, x: 25, y: 34 },
  { id: 5, team: "Team 2", speed: 21.3, distance: 48.2, possession: false, x: 73, y: 29 },
  { id: 7, team: "Team 1", speed: 15.2, distance: 37.6, possession: true, x: 52, y: 55 },
  { id: 10, team: "Team 2", speed: 22.8, distance: 51.4, possession: false, x: 63, y: 70 },
  { id: 12, team: "Team 1", speed: 17.9, distance: 39.7, possession: false, x: 34, y: 72 }
];

const demoEvents = [
  { time: "00:01.80", type: "PASS", team: "Team 1", detail: "Player 10 → Player 7" },
  { time: "00:03.45", type: "POSSESSION", team: "Team 1", detail: "Player 7 control" },
  { time: "00:05.20", type: "INTERCEPTION", team: "Team 2", detail: "Player 5" },
  { time: "00:06.90", type: "PASS", team: "Team 2", detail: "Player 5 → Player 10" }
];

function App() {
  const [inputFile, setInputFile] = useState(null);
  const [inputUrl, setInputUrl] = useState("");
  const [outputUrl, setOutputUrl] = useState("/reference-output.mp4");
  const [activeTab, setActiveTab] = useState("overview");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisDone, setAnalysisDone] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState(demoPlayers[2]);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (inputUrl && inputUrl.startsWith("blob:")) URL.revokeObjectURL(inputUrl);
    };
  }, [inputUrl]);

  const stats = useMemo(() => demoStats, []);

  function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setError("Please upload an MP4, MOV, WebM, or another supported video file.");
      return;
    }
    setError("");
    setInputFile(file);
    const url = URL.createObjectURL(file);
    setInputUrl(url);
    setAnalysisDone(false);
    setProgress(0);
  }

  function onDrop(e) {
    e.preventDefault();
    handleFile(e.dataTransfer.files?.[0]);
  }

  async function analyze() {
    if (!inputFile) {
      setError("Upload a basketball video first.");
      return;
    }

    setError("");
    setIsAnalyzing(true);
    setAnalysisDone(false);
    setProgress(0);

    // If a backend is available, send the real file to it.
    if (API_BASE) {
      try {
        const form = new FormData();
        form.append("file", inputFile);
        const response = await fetch(`${API_BASE}/api/analyze`, {
          method: "POST",
          body: form
        });

        if (!response.ok) throw new Error(`Analysis API returned ${response.status}`);
        const result = await response.json();

        if (result.output_video_url) {
          setOutputUrl(result.output_video_url);
        }
        if (result.metrics) {
          Object.assign(demoStats, result.metrics);
        }
        setProgress(100);
        setAnalysisDone(true);
        setIsAnalyzing(false);
        return;
      } catch (err) {
        console.warn(err);
        setError("Backend analysis failed, so the frontend is showing the reference/demo analytics.");
      }
    }

    // Demo mode: animate a realistic processing pipeline.
    let value = 0;
    const timer = setInterval(() => {
      value += Math.floor(Math.random() * 8) + 4;
      if (value >= 100) {
        clearInterval(timer);
        setProgress(100);
        setIsAnalyzing(false);
        setAnalysisDone(true);
      } else {
        setProgress(value);
      }
    }, 180);
  }

  function reset() {
    setInputFile(null);
    if (inputUrl.startsWith("blob:")) URL.revokeObjectURL(inputUrl);
    setInputUrl("");
    setOutputUrl("/reference-output.mp4");
    setProgress(0);
    setAnalysisDone(false);
    setIsAnalyzing(false);
    setError("");
  }

  function exportJSON() {
    const payload = {
      project: "Basketball Vision Analytics",
      source_video: inputFile?.name || "demo",
      generated_at: new Date().toISOString(),
      metrics: stats,
      players: demoPlayers,
      events: demoEvents
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "basketball-analysis.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><CircleDot size={20} /></div>
          <div>
            <strong>Basketball Vision</strong>
            <span>AI Performance Analytics</span>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="status-pill">
            <span className="status-dot" />
            {analysisDone ? "Analysis ready" : "System ready"}
          </div>
          <button className="icon-btn" title="Reset" onClick={reset}>
            <RotateCcw size={17} />
          </button>
        </div>
      </header>

      <main className="page">
        <section className="hero">
          <div>
            <div className="eyebrow"><Sparkles size={14} /> COMPUTER VISION PIPELINE</div>
            <h1>Basketball Video <span>Analytics</span></h1>
            <p>
              Upload a game video and inspect detection, tracking, team identity,
              tactical court mapping, player movement, possession and events.
            </p>
          </div>

          <div className="hero-actions">
            <button className="primary-btn" onClick={() => inputRef.current?.click()}>
              <Upload size={17} /> Upload video
            </button>
            <button className="secondary-btn" onClick={analyze} disabled={isAnalyzing}>
              <Zap size={17} /> {isAnalyzing ? "Analyzing..." : "Analyze video"}
            </button>
            <input
              ref={inputRef}
              hidden
              type="file"
              accept="video/*"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>
        </section>

        {error && (
          <div className="error-banner">
            <X size={17} />
            <span>{error}</span>
          </div>
        )}

        <section
          className="upload-card"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
        >
          <div className="upload-icon"><FileVideo size={28} /></div>
          <div>
            <strong>{inputFile ? inputFile.name : "Drop your basketball video here"}</strong>
            <p>
              {inputFile
                ? `${(inputFile.size / 1024 / 1024).toFixed(2)} MB · ${inputFile.type || "video"}`
                : "MP4 / MOV / WebM · 720p or higher recommended"}
            </p>
          </div>
          <button className="ghost-btn" onClick={() => inputRef.current?.click()}>
            Browse
          </button>
        </section>

        {(isAnalyzing || analysisDone) && (
          <section className="pipeline-card">
            <div className="pipeline-header">
              <div>
                <span className="section-kicker">ANALYSIS PIPELINE</span>
                <h2>{isAnalyzing ? "Processing video" : "Analysis complete"}</h2>
              </div>
              <span className="progress-label">{progress}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <div className="pipeline-steps">
              {[
                ["Detection", "Players + ball"],
                ["Tracking", "IDs + trajectories"],
                ["Team ID", "Jersey classifier"],
                ["Court", "Homography"],
                ["Analytics", "Speed + distance"],
                ["Events", "Pass + interception"]
              ].map(([title, sub], i) => (
                <div className={`pipeline-step ${progress >= (i + 1) * 16 ? "done" : ""}`} key={title}>
                  <span>{progress >= (i + 1) * 16 ? "✓" : i + 1}</span>
                  <div><b>{title}</b><small>{sub}</small></div>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="video-grid">
          <VideoPanel
            title="Input video"
            subtitle={inputFile ? inputFile.name : "Upload a video to begin"}
            icon={<Video size={16} />}
            src={inputUrl}
            empty={!inputUrl}
          />
          <VideoPanel
            title="AI analysis output"
            subtitle="Reference-style annotated result"
            icon={<Target size={16} />}
            src={outputUrl}
            output
          />
        </section>

        <nav className="tabs">
          {[
            ["overview", "Overview", BarChart3],
            ["players", "Players", Users],
            ["court", "Tactical Court", Layers3],
            ["events", "Events", Activity]
          ].map(([key, label, Icon]) => (
            <button
              key={key}
              className={activeTab === key ? "tab active" : "tab"}
              onClick={() => setActiveTab(key)}
            >
              <Icon size={16} /> {label}
            </button>
          ))}
        </nav>

        {activeTab === "overview" && (
          <>
            <section className="metrics-grid">
              <Metric title="Players tracked" value={stats.players} suffix="" icon={<Users />} />
              <Metric title="Tracking IDF1" value={stats.trackingIdf1} suffix="%" icon={<Activity />} />
              <Metric title="Tracking HOTA" value={stats.trackingHota} suffix="%" icon={<Shield />} />
              <Metric title="Team accuracy" value={stats.teamAccuracy} suffix="%" icon={<Target />} />
              <Metric title="Court error" value={stats.courtMae} suffix=" m" icon={<Layers3 />} />
              <Metric title="Speed MAE" value={stats.speedMae} suffix=" km/h" icon={<Gauge />} />
            </section>

            <section className="two-column">
              <div className="panel">
                <PanelHeader title="Possession" icon={<CircleDot />} />
                <div className="possession-wrap">
                  <div className="donut"><span>91.1%</span><small>F1</small></div>
                  <div className="possession-bars">
                    <Bar label="Team 1" value={58} />
                    <Bar label="Team 2" value={42} />
                    <div className="mini-stat"><span>Possession F1</span><b>{stats.possessionF1}%</b></div>
                  </div>
                </div>
              </div>

              <div className="panel">
                <PanelHeader title="Event detection" icon={<Activity />} />
                <div className="event-metrics">
                  <EventMetric title="Pass Precision" value={stats.passPrecision} />
                  <EventMetric title="Pass Recall" value={stats.passRecall} />
                  <EventMetric title="INT Precision" value={stats.interceptionPrecision} />
                  <EventMetric title="INT Recall" value={stats.interceptionRecall} />
                </div>
              </div>
            </section>
          </>
        )}

        {activeTab === "players" && (
          <section className="panel">
            <PanelHeader title="Tracked players" icon={<Users />} />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>ID</th><th>Team</th><th>Speed</th><th>Distance</th><th>Possession</th></tr>
                </thead>
                <tbody>
                  {demoPlayers.map((p) => (
                    <tr
                      key={p.id}
                      className={selectedPlayer.id === p.id ? "selected-row" : ""}
                      onClick={() => setSelectedPlayer(p)}
                    >
                      <td><span className="player-id">#{p.id}</span></td>
                      <td><span className={p.team === "Team 1" ? "team-one" : "team-two"}>{p.team}</span></td>
                      <td>{p.speed} km/h</td>
                      <td>{p.distance} m</td>
                      <td>{p.possession ? <span className="live-badge">ACTIVE</span> : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {activeTab === "court" && (
          <section className="court-panel">
            <div className="court-toolbar">
              <div>
                <span className="section-kicker">IMAGE → WORLD COORDINATES</span>
                <h2>Tactical Court View</h2>
              </div>
              <span className="accuracy-chip">MAE {stats.courtMae} m</span>
            </div>

            <div className="court">
              <div className="half-line" />
              <div className="center-circle" />
              <div className="key left-key" />
              <div className="key right-key" />
              <div className="hoop left-hoop" />
              <div className="hoop right-hoop" />

              {demoPlayers.map((p) => (
                <button
                  key={p.id}
                  className={`court-player ${p.team === "Team 1" ? "one" : "two"} ${p.possession ? "possessor" : ""}`}
                  style={{ left: `${p.x}%`, top: `${p.y}%` }}
                  onClick={() => setSelectedPlayer(p)}
                  title={`Player ${p.id}`}
                >
                  {p.id}
                </button>
              ))}

              <div className="court-ball" style={{ left: "55%", top: "57%" }} />
            </div>

            <div className="selected-player">
              <div className="selected-avatar">{selectedPlayer.id}</div>
              <div>
                <b>Player #{selectedPlayer.id}</b>
                <span>{selectedPlayer.team} · {selectedPlayer.speed} km/h · {selectedPlayer.distance} m</span>
              </div>
              {selectedPlayer.possession && <span className="live-badge">BALL CONTROL</span>}
            </div>
          </section>
        )}

        {activeTab === "events" && (
          <section className="panel">
            <PanelHeader title="Event timeline" icon={<Timer />} />
            <div className="timeline">
              {demoEvents.map((e, i) => (
                <div className="timeline-item" key={`${e.time}-${i}`}>
                  <div className="timeline-time">{e.time}</div>
                  <div className="timeline-dot" />
                  <div className="timeline-content">
                    <div className="timeline-title">
                      <b>{e.type}</b>
                      <span>{e.team}</span>
                    </div>
                    <p>{e.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="export-panel">
          <div>
            <span className="section-kicker">EXPORT</span>
            <h2>Take the analysis to your pipeline</h2>
            <p>Export structured JSON or use the annotated MP4 as the frontend result.</p>
          </div>
          <div className="export-actions">
            <button className="secondary-btn" onClick={exportJSON}><Download size={16} /> Export JSON</button>
            <a className="primary-btn link-btn" href={outputUrl} download>
              <Download size={16} /> Output MP4
            </a>
          </div>
        </section>
      </main>
    </div>
  );
}

function VideoPanel({ title, subtitle, icon, src, empty, output }) {
  return (
    <div className={`video-panel ${output ? "output-panel" : ""}`}>
      <div className="video-header">
        <div className="video-title">
          <span className="panel-icon">{icon}</span>
          <div><b>{title}</b><small>{subtitle}</small></div>
        </div>
        {src && <button className="icon-btn"><Maximize2 size={15} /></button>}
      </div>

      <div className="video-stage">
        {src ? (
          <>
            <video src={src} controls playsInline />
            {output && (
              <div className="overlay-demo">
                <span>AI OUTPUT</span>
                <span>TRACKING: ON</span>
                <span>BALL: DETECTED</span>
              </div>
            )}
          </>
        ) : (
          <div className="video-empty">
            <FileVideo size={34} />
            <b>No input video</b>
            <span>Upload a basketball video above</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({ title, value, suffix, icon }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <span>{title}</span>
        <strong>{value}{suffix}</strong>
      </div>
      <ChevronRight className="metric-arrow" size={16} />
    </div>
  );
}

function PanelHeader({ title, icon }) {
  return (
    <div className="panel-header">
      <div className="panel-heading"><span className="panel-icon">{icon}</span><h2>{title}</h2></div>
      <button className="ghost-icon"><Search size={16} /></button>
    </div>
  );
}

function Bar({ label, value }) {
  return (
    <div className="bar-row">
      <div><span>{label}</span><b>{value}%</b></div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${value}%` }} /></div>
    </div>
  );
}

function EventMetric({ title, value }) {
  return (
    <div className="event-metric">
      <span>{title}</span>
      <strong>{value}%</strong>
      <div className="tiny-track"><div style={{ width: `${value}%` }} /></div>
    </div>
  );
}

export default App;