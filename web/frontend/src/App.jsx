import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Database,
  Dna,
  FileArchive,
  FileText,
  GitBranch,
  Play,
  ShieldCheck,
  Square,
  UploadCloud,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const tabs = [
  ["upload", "Upload", UploadCloud],
  ["status", "Run Status", Activity],
  ["summary", "Summary", BarChart3],
  ["tree", "Tree", GitBranch],
  ["clade", "Clade", Dna],
  ["antigenic", "Antigenic", ShieldCheck],
  ["qc", "QC/Outliers", Database],
  ["files", "Files", FileArchive],
];

const fileFields = [
  ["target", "Target FASTA", true],
  ["reference", "Reference FASTA", true],
  ["background", "Background FASTA", false],
  ["vaccine", "Vaccine FASTA", false],
  ["tree_date_metadata", "Tree date metadata", false],
  ["nextclade_results", "Nextclade results", false],
  ["tree_outlier_file", "Tree outlier file", false],
];

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [files, setFiles] = useState({});
  const [options, setOptions] = useState({
    tree_method: "auto",
    tree_plot_style: "figtree",
    tree_display_max_tips: 0,
    tree_display_branch_cap: 0.65,
    max_tree_sequences: 0,
    target_date: "",
    iqtree_model: "GTR+G",
    iqtree_threads: "AUTO",
    iqtree_fast: true,
    treetime_remove_outliers: false,
    treetime_outlier_max_passes: 6,
    tree_clade_bar: false,
    clade_method: "auto",
    allow_rule_clade_fallback: true,
    iqtree_exe: "",
    treetime_exe: "",
  });
  const [runId, setRunId] = useState("");
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const resultFiles = results?.files || [];
  const manifest = results?.manifest || {};
  const statusValue = status?.status || results?.status || "idle";

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const nextStatus = await getJson(`/analyses/${runId}/status`);
        if (cancelled) return;
        setStatus(nextStatus);
        if (["completed", "failed", "cancelled"].includes(nextStatus.status)) {
          const nextResults = await getJson(`/analyses/${runId}/results`);
          if (!cancelled) setResults(nextResults);
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };
    tick();
    const timer = window.setInterval(tick, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [runId]);

  const fileMap = useMemo(() => {
    const map = new Map();
    resultFiles.forEach((file) => map.set(file.name, file));
    return map;
  }, [resultFiles]);

  async function submitRun(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    setResults(null);
    setStatus(null);
    const form = new FormData();
    fileFields.forEach(([name]) => {
      if (files[name]) form.append(name, files[name]);
    });
    Object.entries(options).forEach(([key, value]) => {
      form.append(key, value);
    });
    try {
      const response = await fetch(`${API_BASE}/analyses`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setRunId(payload.run_id);
      setActiveTab("status");
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function cancelRun() {
    if (!runId) return;
    const payload = await postJson(`/analyses/${runId}/cancel`);
    setStatus(payload);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Influmatics</h1>
          <p>H3N2 HA retrospective analysis dashboard</p>
        </div>
        <div className={`status-pill status-${statusValue}`}>{statusValue}</div>
      </header>

      <nav className="tabs" aria-label="Analysis sections">
        {tabs.map(([id, label, Icon]) => (
          <button
            key={id}
            className={activeTab === id ? "active" : ""}
            type="button"
            onClick={() => setActiveTab(id)}
            title={label}
          >
            <Icon size={17} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      {error && <div className="notice error">{error}</div>}

      {activeTab === "upload" && (
        <UploadTab
          files={files}
          setFiles={setFiles}
          options={options}
          setOptions={setOptions}
          submitting={submitting}
          submitRun={submitRun}
        />
      )}
      {activeTab === "status" && (
        <StatusTab
          runId={runId}
          status={status}
          submitting={submitting}
          cancelRun={cancelRun}
        />
      )}
      {activeTab === "summary" && <SummaryTab manifest={manifest} results={results} />}
      {activeTab === "tree" && <TreeTab runId={runId} fileMap={fileMap} />}
      {activeTab === "clade" && (
        <CladeTab runId={runId} manifest={manifest} />
      )}
      {activeTab === "antigenic" && <AntigenicTab runId={runId} fileMap={fileMap} />}
      {activeTab === "qc" && <QCTab runId={runId} fileMap={fileMap} />}
      {activeTab === "files" && <FilesTab files={resultFiles} />}
    </main>
  );
}

function UploadTab({ files, setFiles, options, setOptions, submitting, submitRun }) {
  return (
    <form className="workflow-grid" onSubmit={submitRun}>
      <section className="panel upload-panel">
        <h2>Inputs</h2>
        <div className="file-grid">
          {fileFields.map(([name, label, required]) => (
            <label className="file-row" key={name}>
              <span>
                {label}
                {required && <b>*</b>}
              </span>
              <input
                type="file"
                onChange={(event) =>
                  setFiles((current) => ({
                    ...current,
                    [name]: event.target.files?.[0],
                  }))
                }
              />
              <small>{files[name]?.name || "No file selected"}</small>
            </label>
          ))}
        </div>
      </section>
      <section className="panel options-panel">
        <h2>Options</h2>
        <div className="options-grid">
          <SelectControl
            label="Tree method"
            value={options.tree_method}
            onChange={(value) => setOptions({ ...options, tree_method: value })}
            values={["auto", "nj", "fast-upgma", "iqtree", "iqtree-treetime"]}
          />
          <SelectControl
            label="Plot style"
            value={options.tree_plot_style}
            onChange={(value) => setOptions({ ...options, tree_plot_style: value })}
            values={["figtree", "dashboard"]}
          />
          <TextControl
            label="Target date"
            value={options.target_date}
            onChange={(value) => setOptions({ ...options, target_date: value })}
            placeholder="2023 or 2023-01-20"
          />
          <NumberControl
            label="Display tips"
            value={options.tree_display_max_tips}
            onChange={(value) => setOptions({ ...options, tree_display_max_tips: value })}
          />
          <NumberControl
            label="Max tree sequences"
            value={options.max_tree_sequences}
            onChange={(value) => setOptions({ ...options, max_tree_sequences: value })}
          />
          <TextControl
            label="IQ-TREE model"
            value={options.iqtree_model}
            onChange={(value) => setOptions({ ...options, iqtree_model: value })}
          />
          <TextControl
            label="IQ-TREE threads"
            value={options.iqtree_threads}
            onChange={(value) => setOptions({ ...options, iqtree_threads: value })}
          />
          <NumberControl
            label="Outlier passes"
            value={options.treetime_outlier_max_passes}
            onChange={(value) => setOptions({ ...options, treetime_outlier_max_passes: value })}
          />
          <ToggleControl
            label="IQ-TREE fast"
            checked={options.iqtree_fast}
            onChange={(value) => setOptions({ ...options, iqtree_fast: value })}
          />
          <ToggleControl
            label="Remove TreeTime outliers"
            checked={options.treetime_remove_outliers}
            onChange={(value) => setOptions({ ...options, treetime_remove_outliers: value })}
          />
          <ToggleControl
            label="Tree clade bar"
            checked={options.tree_clade_bar}
            onChange={(value) => setOptions({ ...options, tree_clade_bar: value })}
          />
          <ToggleControl
            label="Allow rule fallback"
            checked={options.allow_rule_clade_fallback}
            onChange={(value) => setOptions({ ...options, allow_rule_clade_fallback: value })}
          />
        </div>
        <button className="primary-action" disabled={submitting} type="submit">
          <Play size={18} />
          <span>{submitting ? "Starting" : "Start Analysis"}</span>
        </button>
      </section>
    </form>
  );
}

function StatusTab({ runId, status, submitting, cancelRun }) {
  return (
    <section className="panel full-panel">
      <div className="panel-title-row">
        <h2>Run Status</h2>
        {runId && (
          <button className="quiet-action" type="button" onClick={cancelRun}>
            <Square size={16} />
            <span>Cancel</span>
          </button>
        )}
      </div>
      <div className="metrics-row">
        <Metric label="Run ID" value={runId || "Not started"} />
        <Metric label="Status" value={status?.status || (submitting ? "queued" : "idle")} />
        <Metric label="Return code" value={status?.return_code ?? "-"} />
      </div>
      <pre className="log-view">{status?.log_tail || "No log output yet."}</pre>
    </section>
  );
}

function SummaryTab({ manifest, results }) {
  const counts = manifest.counts || {};
  const parameters = manifest.parameters || {};
  return (
    <section className="panel full-panel">
      <h2>Summary</h2>
      <div className="metrics-row wrap">
        {Object.entries(counts).map(([key, value]) => (
          <Metric key={key} label={key.replaceAll("_", " ")} value={String(value)} />
        ))}
      </div>
      <h3>Parameters</h3>
      <KeyValueTable values={parameters} />
      {results?.warnings?.length > 0 && (
        <div className="notice error">{results.warnings.join(" ")}</div>
      )}
    </section>
  );
}

function TreeTab({ runId, fileMap }) {
  const tree = fileMap.get("phylogenetic_tree.png");
  const newick = fileMap.get("phylogenetic_tree.newick");
  return (
    <section className="panel full-panel">
      <div className="panel-title-row">
        <h2>Tree</h2>
        {newick && <FileLink file={newick} />}
      </div>
      {tree ? (
        <img className="tree-image" alt="Phylogenetic tree" src={absoluteUrl(tree.url)} />
      ) : (
        <EmptyState text={runId ? "Tree image is not available yet." : "Start a run first."} />
      )}
    </section>
  );
}

function CladeTab({ runId, manifest }) {
  const counts = manifest.counts || {};
  return (
    <section className="panel full-panel">
      <h2>Clade</h2>
      <div className="count-panels">
        <CountList title="Clades" values={counts.clade_counts || {}} />
        <CountList title="Subclades" values={counts.subclade_counts || {}} />
      </div>
      <CsvTable runId={runId} filename="clade_assignments.csv" />
    </section>
  );
}

function AntigenicTab({ runId, fileMap }) {
  const cartography = fileMap.get("antigenic_cartography.png");
  return (
    <section className="panel full-panel">
      <h2>Antigenic</h2>
      {cartography && (
        <img
          className="chart-image"
          alt="Antigenic cartography"
          src={absoluteUrl(cartography.url)}
        />
      )}
      <CsvTable runId={runId} filename="antigenic_distance_to_vaccine.csv" />
      <CsvTable runId={runId} filename="antigenic_site_mutations.csv" />
    </section>
  );
}

function QCTab({ runId, fileMap }) {
  const names = [
    "tree_outliers_removed.csv",
    "tree_metadata_outliers_removed.csv",
    "treetime_outliers_removed.csv",
  ];
  return (
    <section className="panel full-panel">
      <h2>QC/Outliers</h2>
      {names.map((name) =>
        fileMap.has(name) ? <CsvTable key={name} runId={runId} filename={name} /> : null,
      )}
      {!names.some((name) => fileMap.has(name)) && <EmptyState text="No outlier files yet." />}
    </section>
  );
}

function FilesTab({ files }) {
  return (
    <section className="panel full-panel">
      <h2>Files</h2>
      <div className="file-list">
        {files.map((file) => (
          <a key={file.name} className="download-row" href={absoluteUrl(file.url)}>
            <FileText size={17} />
            <span>{file.name}</span>
            <small>{formatBytes(file.size)}</small>
          </a>
        ))}
      </div>
      {!files.length && <EmptyState text="Result files will appear after a run completes." />}
    </section>
  );
}

function CsvTab({ runId, filename, title }) {
  return (
    <section className="panel full-panel">
      <h2>{title}</h2>
      <CsvTable runId={runId} filename={filename} />
    </section>
  );
}

function CountList({ title, values }) {
  const entries = Object.entries(values).sort((a, b) => Number(b[1]) - Number(a[1]));
  return (
    <div className="count-list">
      <h3>{title}</h3>
      {entries.length ? (
        entries.map(([label, value]) => (
          <div className="count-row" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))
      ) : (
        <div className="empty-mini">No counts yet</div>
      )}
    </div>
  );
}

function CsvTable({ runId, filename }) {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    fetch(absoluteUrl(`/analyses/${runId}/files/${filename}`))
      .then((response) => {
        if (!response.ok) throw new Error(`${filename} is not available.`);
        return response.text();
      })
      .then((text) => {
        if (!cancelled) setRows(parseCsv(text));
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, filename]);

  if (!runId) return <EmptyState text="Start a run first." />;
  if (error) return <div className="notice muted">{error}</div>;
  if (!rows.length) return <div className="notice muted">{filename}: no rows</div>;

  const columns = Object.keys(rows[0]);
  return (
    <div className="table-block">
      <div className="table-title">{filename}</div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 100).map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => (
                  <td key={column}>{row[column]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > 100 && <small>Showing 100 of {rows.length} rows</small>}
    </div>
  );
}

function SelectControl({ label, value, onChange, values }) {
  return (
    <label className="control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {values.map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
      </select>
    </label>
  );
}

function TextControl({ label, value, onChange, placeholder = "" }) {
  return (
    <label className="control">
      <span>{label}</span>
      <input
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function NumberControl({ label, value, onChange }) {
  return (
    <label className="control">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ToggleControl({ label, checked, onChange }) {
  return (
    <label className="toggle-control">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function KeyValueTable({ values }) {
  const entries = Object.entries(values || {});
  if (!entries.length) return <EmptyState text="No manifest data yet." />;
  return (
    <div className="table-scroll compact">
      <table>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <th>{key}</th>
              <td>{String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FileLink({ file }) {
  return (
    <a className="quiet-link" href={absoluteUrl(file.url)}>
      {file.name}
    </a>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}

async function getJson(path) {
  const response = await fetch(absoluteUrl(path));
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function postJson(path) {
  const response = await fetch(absoluteUrl(path), { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function absoluteUrl(path) {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const delimiter = lines[0].includes("\t") ? "\t" : ",";
  const header = splitDelimited(lines[0], delimiter);
  return lines.slice(1).map((line) => {
    const cells = splitDelimited(line, delimiter);
    return Object.fromEntries(header.map((key, index) => [key, cells[index] || ""]));
  });
}

function splitDelimited(line, delimiter) {
  const cells = [];
  let current = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === delimiter && !quoted) {
      cells.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default App;
