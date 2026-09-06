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
  RotateCcw,
  Search,
  ShieldCheck,
  Square,
  UploadCloud,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const DEFAULT_BACKGROUND_DATASET = "h3n2_ha_demo_reference";
const DEFAULT_TREE_METHOD = "iqtree-treetime";
const DEFAULT_TREE_ZOOM = { x: 0.72, y: 0.16 };

const tabs = [
  ["upload", "Upload", UploadCloud],
  ["status", "Run Status", Activity],
  ["summary", "Summary", BarChart3],
  ["tree", "Tree", GitBranch],
  ["clade", "Clade", Dna],
  ["antigenic", "Antigenic", ShieldCheck],
  ["variability", "Variability", BarChart3],
  ["qc", "QC/Outliers", Database],
  ["files", "Files", FileArchive],
];

const primaryFileFields = [
  ["target", "Target FASTA", true],
];

const advancedFileFields = [
  ["background", "Background FASTA", false],
  ["tree_date_metadata", "Tree date metadata", false],
  ["nextclade_results", "Nextclade results", false],
  ["tree_outlier_file", "Tree outlier file", false],
];

const allFileFields = [...primaryFileFields, ...advancedFileFields];

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [files, setFiles] = useState({});
  const [targetInputMode, setTargetInputMode] = useState("file");
  const [targetFastaText, setTargetFastaText] = useState("");
  const [options, setOptions] = useState({
    background_dataset: DEFAULT_BACKGROUND_DATASET,
    tree_method: DEFAULT_TREE_METHOD,
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
  const [backgroundDatasets, setBackgroundDatasets] = useState([]);
  const [backgroundDatasetError, setBackgroundDatasetError] = useState("");

  const resultFiles = results?.files || [];
  const manifest = results?.manifest || {};
  const statusValue = status?.status || results?.status || "idle";

  useEffect(() => {
    let cancelled = false;
    getJson("/background-datasets")
      .then((payload) => {
        if (cancelled) return;
        const datasets = payload.datasets || [];
        setBackgroundDatasets(datasets);
        setOptions((current) => ({
          ...current,
          background_dataset: resolveBackgroundDataset(
            current.background_dataset,
            payload.default_dataset,
            datasets,
          ),
        }));
      })
      .catch((err) => {
        if (!cancelled) setBackgroundDatasetError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
    const targetFile = prepareTargetFile({
      mode: targetInputMode,
      file: files.target,
      pastedText: targetFastaText,
    });
    if (!targetFile) {
      setError(
        targetInputMode === "paste"
          ? "Paste a target nucleotide or protein sequence before starting the analysis."
          : "Choose a target FASTA file before starting the analysis.",
      );
      setSubmitting(false);
      return;
    }
    form.append("target", targetFile);
    allFileFields.forEach(([name]) => {
      if (name !== "target" && files[name]) form.append(name, files[name]);
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
          targetInputMode={targetInputMode}
          setTargetInputMode={setTargetInputMode}
          targetFastaText={targetFastaText}
          setTargetFastaText={setTargetFastaText}
          options={options}
          setOptions={setOptions}
          backgroundDatasets={backgroundDatasets}
          backgroundDatasetError={backgroundDatasetError}
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
      {activeTab === "variability" && <VariabilityTab runId={runId} fileMap={fileMap} />}
      {activeTab === "qc" && <QCTab runId={runId} fileMap={fileMap} />}
      {activeTab === "files" && <FilesTab files={resultFiles} />}
    </main>
  );
}

function UploadTab({
  files,
  setFiles,
  targetInputMode,
  setTargetInputMode,
  targetFastaText,
  setTargetFastaText,
  options,
  setOptions,
  backgroundDatasets,
  backgroundDatasetError,
  submitting,
  submitRun,
}) {
  const selectedDataset = backgroundDatasets.find(
    (dataset) => dataset.id === options.background_dataset,
  );
  const datasetValues = backgroundDatasets.length
    ? backgroundDatasets.map((dataset) => dataset.id)
    : [options.background_dataset || DEFAULT_BACKGROUND_DATASET];
  const treeTimeNotice = getTreeTimeNotice(options, selectedDataset, files);

  return (
    <form className="workflow-grid" onSubmit={submitRun}>
      <section className="panel upload-panel">
        <h2>Start a run</h2>
        <p className="panel-note">
          Upload the target sequence and choose a curated background. Dates and
          background inputs are prepared automatically when the preset has them.
        </p>
        <TargetSequenceControl
          mode={targetInputMode}
          setMode={setTargetInputMode}
          file={files.target}
          setFile={(file) =>
            setFiles((current) => ({
              ...current,
              target: file,
            }))
          }
          pastedText={targetFastaText}
          setPastedText={setTargetFastaText}
        />
        <TextControl
          label="Collection date"
          value={options.target_date}
          onChange={(value) => setOptions({ ...options, target_date: value })}
          placeholder="2024-01-20 or 2024"
        />
        <SelectControl
          label="Background preset"
          value={options.background_dataset}
          onChange={(value) => setOptions({ ...options, background_dataset: value })}
          values={datasetValues}
          labels={Object.fromEntries(
            backgroundDatasets.map((dataset) => [dataset.id, dataset.label]),
          )}
        />
        {selectedDataset && <DatasetCard dataset={selectedDataset} />}
        {treeTimeNotice && <div className="notice warning">{treeTimeNotice}</div>}
        {backgroundDatasetError && (
          <div className="notice muted">
            Background presets could not be loaded. Use Advanced overrides if needed.
          </div>
        )}
        <details className="advanced-inputs">
          <summary>Advanced overrides</summary>
          <div className="file-grid">
            {advancedFileFields.map(([name, label, required]) => (
              <FileControl
                key={name}
                name={name}
                label={label}
                required={required}
                files={files}
                setFiles={setFiles}
              />
            ))}
          </div>
        </details>
      </section>
      <section className="panel options-panel">
        <h2>Options</h2>
        <div className="options-grid">
          <SelectControl
            label="Tree method"
            value={options.tree_method}
            onChange={(value) => setOptions({ ...options, tree_method: value })}
            values={["auto", "nj", "fast-upgma", "iqtree", "iqtree-treetime"]}
            labels={{
              "iqtree-treetime": "IQ-TREE + TreeTime",
              iqtree: "IQ-TREE only",
              "fast-upgma": "Fast UPGMA",
              nj: "Neighbor joining",
              auto: "Auto preview",
            }}
          />
          <SelectControl
            label="Plot style"
            value={options.tree_plot_style}
            onChange={(value) => setOptions({ ...options, tree_plot_style: value })}
            values={["figtree", "dashboard"]}
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

function TargetSequenceControl({
  mode,
  setMode,
  file,
  setFile,
  pastedText,
  setPastedText,
}) {
  const pastedRecords = countPastedFastaRecords(pastedText);
  return (
    <div className="target-sequence-control">
      <div className="target-input-heading">
        <span>Target sequence<b>*</b></span>
        <div className="segmented-control" role="group" aria-label="Target sequence input method">
          <button
            type="button"
            className={mode === "file" ? "active" : ""}
            aria-pressed={mode === "file"}
            onClick={() => setMode("file")}
          >
            Upload file
          </button>
          <button
            type="button"
            className={mode === "paste" ? "active" : ""}
            aria-pressed={mode === "paste"}
            onClick={() => setMode("paste")}
          >
            Paste sequence
          </button>
        </div>
      </div>
      {mode === "file" ? (
        <label className="target-file-drop">
          <UploadCloud size={20} />
          <span>{file?.name || "Choose a FASTA file"}</span>
          <small>.fasta, .fa, .fas, .fna, or .txt</small>
          <input
            type="file"
            accept=".fasta,.fa,.fas,.fna,.txt"
            onChange={(event) => setFile(event.target.files?.[0])}
          />
        </label>
      ) : (
        <label className="target-paste-field">
          <span>FASTA or raw sequence</span>
          <textarea
            value={pastedText}
            onChange={(event) => setPastedText(event.target.value)}
            placeholder={">sample_name\nATGAAAGCAAAACTACTGGTCCTGTTATGTGCA..."}
            spellCheck="false"
            rows={11}
          />
          <small>
            {pastedText.trim()
              ? `${pastedRecords || 1} sequence${(pastedRecords || 1) === 1 ? "" : "s"} ready`
              : "Multiple FASTA records are supported. A header is added automatically for a raw sequence."}
          </small>
        </label>
      )}
    </div>
  );
}

function FileControl({ name, label, required, files, setFiles }) {
  return (
    <label className="file-row">
      <span>
        {label}
        {required && <b>*</b>}
      </span>
      <input
        type="file"
        required={required}
        accept=".fasta,.fa,.fas,.fna,.txt"
        onChange={(event) =>
          setFiles((current) => ({
            ...current,
            [name]: event.target.files?.[0],
          }))
        }
      />
      <small>{files[name]?.name || "No file selected"}</small>
    </label>
  );
}

function prepareTargetFile({ mode, file, pastedText }) {
  if (mode === "file") return file || null;
  const fasta = normalizePastedFasta(pastedText);
  if (!fasta) return null;
  return new File([fasta], "pasted_target.fasta", { type: "text/plain" });
}

function normalizePastedFasta(value) {
  const normalized = String(value || "").replace(/\r\n?/g, "\n").trim();
  if (!normalized) return "";
  if (normalized.startsWith(">")) return `${normalized}\n`;
  const sequence = normalized.replace(/\s+/g, "");
  return sequence ? `>pasted_target\n${sequence}\n` : "";
}

function countPastedFastaRecords(value) {
  const text = String(value || "").trim();
  if (!text) return 0;
  if (!text.startsWith(">")) return 1;
  return text.split(/\n/).filter((line) => line.trim().startsWith(">")).length;
}

function getTreeTimeNotice(options, dataset, files) {
  if (options.tree_method !== "iqtree-treetime") return "";
  if (files.tree_date_metadata) return "TreeTime will use the uploaded date metadata file.";
  const sequenceCount = Number(dataset?.sequence_count || 0);
  if (sequenceCount > 0 && sequenceCount < 3) {
    return (
      "The selected background preset is too small for TreeTime. Choose a full "
      + "background dataset or upload Tree date metadata before running IQ-TREE + TreeTime."
    );
  }
  if (!String(options.target_date || "").trim()) {
    return "Add the target collection date so TreeTime can date the uploaded target tip.";
  }
  return "";
}

function DatasetCard({ dataset }) {
  const dateRange = (dataset.date_range || []).filter(Boolean).join(" to ");
  return (
    <div className="dataset-card">
      <div>
        <strong>{dataset.label}</strong>
        <span>version {dataset.version}</span>
      </div>
      <p>{dataset.description}</p>
      <dl>
        <div>
          <dt>Sequences</dt>
          <dd>{dataset.sequence_count || "Unknown"}</dd>
        </div>
        <div>
          <dt>Date range</dt>
          <dd>{dateRange || "Unknown"}</dd>
        </div>
      </dl>
    </div>
  );
}

function resolveBackgroundDataset(currentValue, defaultDataset, datasets) {
  const datasetIds = new Set(datasets.map((dataset) => dataset.id));
  if (defaultDataset && (!datasetIds.size || datasetIds.has(defaultDataset))) {
    return defaultDataset;
  }
  if (currentValue && (!datasetIds.size || datasetIds.has(currentValue))) {
    return currentValue;
  }
  return datasets[0]?.id || defaultDataset || DEFAULT_BACKGROUND_DATASET;
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
  const newick = findResultFile(fileMap, [
    "phylogenetic_tree.newick",
    "phylogenetic_tree.nwk",
    "iqtree_treetime/treetime/timetree.nwk",
    "iqtree_treetime/treetime/timetree.newick",
    "iqtree_treetime/treetime/timetree.nexus",
    "iqtree_treetime/treetime/annotated_tree.nexus",
  ], { allowEmpty: false });
  const metadata = findResultFile(fileMap, [
    "tree_tip_metadata.json",
    "metadata/tree_tip_metadata.json",
  ]);
  return (
    <section className="panel full-panel">
      <div className="panel-title-row">
        <h2>Tree</h2>
        <div className="link-row">
          {newick && <FileLink file={newick} />}
          {tree && <FileLink file={tree} />}
        </div>
      </div>
      {newick ? (
        <InteractiveTree newickFile={newick} metadataFile={metadata} />
      ) : (
        <EmptyState text={runId ? "Tree file is not available yet." : "Start a run first."} />
      )}
      {tree && (
        <details className="static-tree-details">
          <summary>Static figure</summary>
          <img className="tree-image" alt="Phylogenetic tree" src={absoluteUrl(tree.url)} />
        </details>
      )}
    </section>
  );
}

function InteractiveTree({ newickFile, metadataFile }) {
  const [newickText, setNewickText] = useState("");
  const [metadataMap, setMetadataMap] = useState(new Map());
  const [error, setError] = useState("");
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [showLabels, setShowLabels] = useState(false);
  const [zoom, setZoom] = useState(DEFAULT_TREE_ZOOM);

  useEffect(() => {
    let cancelled = false;
    setError("");
    setNewickText("");
    fetch(absoluteUrl(newickFile.url))
      .then((response) => {
        if (!response.ok) throw new Error("Tree Newick file is not available.");
        return response.text();
      })
      .then((text) => {
        if (!cancelled) setNewickText(text);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [newickFile.url]);

  useEffect(() => {
    let cancelled = false;
    setMetadataMap(new Map());
    if (!metadataFile) return () => {};
    fetch(absoluteUrl(metadataFile.url))
      .then((response) => {
        if (!response.ok) throw new Error("Tree metadata is not available.");
        return response.json();
      })
      .then((payload) => {
        if (!cancelled) setMetadataMap(buildTipMetadataMap(payload));
      })
      .catch(() => {
        if (!cancelled) setMetadataMap(new Map());
      });
    return () => {
      cancelled = true;
    };
  }, [metadataFile?.url]);

  const layout = useMemo(() => {
    if (!newickText.trim()) return null;
    try {
      return layoutNewickTree(parseNewick(newickText), metadataMap, zoom);
    } catch (err) {
      return { error: err.message };
    }
  }, [newickText, metadataMap, zoom]);

  const normalizedSearch = searchTerm.trim().toLowerCase();
  const activeNode = hovered || selected;

  if (error) return <div className="notice error">{error}</div>;
  if (!newickText.trim()) return <EmptyState text="Loading interactive tree." />;
  if (layout?.error) return <div className="notice error">{layout.error}</div>;
  if (!layout) return <EmptyState text="Preparing interactive tree." />;

  return (
    <div className="interactive-tree">
      <div className="tree-toolbar">
        <label className="tree-search">
          <Search size={16} />
          <input
            value={searchTerm}
            placeholder="Search sequence"
            onChange={(event) => setSearchTerm(event.target.value)}
          />
        </label>
        <button type="button" onClick={() => setZoom((current) => ({ ...current, x: Math.min(current.x + 0.25, 5) }))}>
          <ZoomIn size={16} />
          <span>Wide</span>
        </button>
        <button type="button" onClick={() => setZoom((current) => ({ ...current, x: Math.max(current.x - 0.25, 0.45) }))}>
          <ZoomOut size={16} />
          <span>Narrow</span>
        </button>
        <button type="button" onClick={() => setZoom((current) => ({ ...current, y: Math.min(current.y + 0.35, 6) }))}>
          <ZoomIn size={16} />
          <span>Tall</span>
        </button>
        <button type="button" onClick={() => setZoom((current) => ({ ...current, y: Math.max(current.y - 0.1, 0.12) }))}>
          <ZoomOut size={16} />
          <span>Short</span>
        </button>
        <button type="button" onClick={() => setZoom(DEFAULT_TREE_ZOOM)}>
          <RotateCcw size={16} />
          <span>Reset</span>
        </button>
        <label className="tree-label-toggle">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(event) => setShowLabels(event.target.checked)}
          />
          <span>Labels</span>
        </label>
      </div>
      <div className="tree-meta-row">
        <Metric label="tips" value={String(layout.leaves.length)} />
        <Metric label="nodes" value={String(layout.nodes.length)} />
        <Metric label="max branch depth" value={formatNumber(layout.maxDepth)} />
      </div>
      <div className="interactive-tree-grid">
        <div className="tree-svg-scroll">
          <svg
            className="tree-svg"
            role="img"
            aria-label="Interactive phylogenetic tree"
            width={layout.width}
            height={layout.height}
            viewBox={`0 0 ${layout.width} ${layout.height}`}
          >
            <g>
              {layout.edges.map((edge) => (
                <g key={edge.id}>
                  <line
                    className="tree-edge tree-edge-vertical"
                    x1={edge.parent.x}
                    x2={edge.parent.x}
                    y1={edge.parent.y}
                    y2={edge.child.y}
                  />
                  <line
                    className="tree-edge"
                    x1={edge.parent.x}
                    x2={edge.child.x}
                    y1={edge.child.y}
                    y2={edge.child.y}
                  />
                </g>
              ))}
            </g>
            <g>
              {layout.leaves.map((node) => {
                const searchable = `${node.name} ${node.meta.clade} ${node.meta.subclade}`.toLowerCase();
                const matched = normalizedSearch && searchable.includes(normalizedSearch);
                const highlighted = matched || selected?.id === node.id;
                const showNodeLabel = showLabels || matched || selected?.id === node.id;
                const radius = Math.max(0.55, Math.min(4.8, layout.leafGap * 0.48));
                return (
                  <g
                    key={node.id}
                    className={`tree-tip group-${node.meta.group || "background"}${highlighted ? " highlighted" : ""}`}
                    transform={`translate(${node.x}, ${node.y})`}
                    onMouseEnter={() => setHovered(node)}
                    onMouseLeave={() => setHovered(null)}
                    onClick={() => setSelected(node)}
                  >
                    <circle r={highlighted ? Math.max(radius + 2, 4.5) : radius} />
                    <title>{tooltipText(node)}</title>
                    {showNodeLabel && (
                      <text x="8" y="4">
                        {node.name}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        </div>
        <aside className="tree-detail-panel">
          <h3>{activeNode ? "Node detail" : "Hover a node"}</h3>
          {activeNode ? (
            <dl>
              <div>
                <dt>Name</dt>
                <dd>{activeNode.name || "(internal node)"}</dd>
              </div>
              <div>
                <dt>Group</dt>
                <dd>{activeNode.meta.group || "background"}</dd>
              </div>
              <div>
                <dt>Clade</dt>
                <dd>{activeNode.meta.clade || "unassigned"}</dd>
              </div>
              <div>
                <dt>Subclade</dt>
                <dd>{activeNode.meta.subclade || activeNode.meta.clade || "unassigned"}</dd>
              </div>
              <div>
                <dt>Collection date</dt>
                <dd>{activeNode.meta.collection_date || "-"}</dd>
              </div>
              <div>
                <dt>Branch depth</dt>
                <dd>{formatNumber(activeNode.depth)}</dd>
              </div>
            </dl>
          ) : (
            <p>
              Move the cursor over a tip to inspect its name, group, clade,
              subclade, and collection date.
            </p>
          )}
          <div className="tree-legend">
            <span><i className="legend-dot target" />target</span>
            <span><i className="legend-dot background" />background</span>
            <span><i className="legend-dot vaccine" />vaccine</span>
            <span><i className="legend-dot reference" />reference</span>
          </div>
        </aside>
      </div>
    </div>
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
        <FigureImageViewer
          file={cartography}
          alt="Antigenic cartography"
          initialScale={1.45}
        />
      )}
      {!cartography && (
        <EmptyState text={runId ? "Antigenic figure is not available yet." : "Start a run first."} />
      )}
      <CsvTable runId={runId} filename="antigenic_distance_to_vaccine.csv" />
      <CsvTable runId={runId} filename="antigenic_site_mutations.csv" />
    </section>
  );
}

function VariabilityTab({ runId, fileMap }) {
  const variabilityFigure = findResultFile(fileMap, [
    "codon_variability.png",
    "figures/codon_variability.png",
  ]);
  return (
    <section className="panel full-panel">
      <h2>Variability</h2>
      <div className="interpretation-card">
        <h3>Codon Variability Summary</h3>
        <p>
          This educational view summarizes observed historical amino-acid
          changing codon variation across public H3N2 HA sequences.
        </p>
        <ul>
          <li>It describes past sequence variability by HA position or region.</li>
          <li>It is not mutation prediction, ranking, fitness estimation, or design.</li>
          <li>It should be interpreted as retrospective surveillance education.</li>
        </ul>
      </div>
      {variabilityFigure ? (
        <FigureImageViewer
          file={variabilityFigure}
          alt="Retrospective codon variability summary"
          initialScale={1.25}
        />
      ) : (
        <EmptyState
          text={
            runId
              ? "Codon variability figure is not available for this run."
              : "Start a run first."
          }
        />
      )}
      <CsvTable runId={runId} filename="codon_variability_regions.csv" />
      <CsvTable runId={runId} filename="codon_variability_sites.csv" />
    </section>
  );
}

function findResultFile(fileMap, names, options = {}) {
  const allowEmpty = options.allowEmpty ?? true;
  for (const name of names) {
    const file = fileMap.get(name);
    if (file && (allowEmpty || Number(file.size || 0) > 0)) return file;
  }
  return null;
}

function FigureImageViewer({ file, alt, initialScale = 1 }) {
  const [scale, setScale] = useState(initialScale);
  return (
    <div className="figure-viewer">
      <div className="figure-toolbar">
        <a className="quiet-link" href={absoluteUrl(file.url)}>
          {file.name}
        </a>
        <div className="figure-actions">
          <button
            type="button"
            onClick={() => setScale((current) => Math.min(current + 0.2, 2.8))}
          >
            <ZoomIn size={16} />
            <span>Zoom in</span>
          </button>
          <button
            type="button"
            onClick={() => setScale((current) => Math.max(current - 0.2, 0.8))}
          >
            <ZoomOut size={16} />
            <span>Zoom out</span>
          </button>
          <button type="button" onClick={() => setScale(initialScale)}>
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
          <span className="figure-scale">{Math.round(scale * 100)}%</span>
        </div>
      </div>
      <div className="figure-scroll">
        <img
          className="chart-image enlarged"
          alt={alt}
          src={absoluteUrl(file.url)}
          style={{ width: `${scale * 100}%` }}
        />
      </div>
    </div>
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

function SelectControl({ label, value, onChange, values, labels = {} }) {
  return (
    <label className="control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {values.map((item) => (
          <option key={item} value={item}>
            {labels[item] || item}
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

function parseNewick(text) {
  const source = extractNewickSource(text).trim().replace(/;+\s*$/, "");
  let index = 0;
  let nextId = 0;

  function skipSpaceAndComments() {
    while (index < source.length) {
      if (/\s/.test(source[index])) {
        index += 1;
      } else if (source[index] === "[") {
        let depth = 1;
        index += 1;
        while (index < source.length && depth > 0) {
          if (source[index] === "[") depth += 1;
          if (source[index] === "]") depth -= 1;
          index += 1;
        }
      } else {
        break;
      }
    }
  }

  function readLabel() {
    skipSpaceAndComments();
    let label = "";
    if (source[index] === "'") {
      index += 1;
      while (index < source.length) {
        const char = source[index];
        if (char === "'" && source[index + 1] === "'") {
          label += "'";
          index += 2;
        } else if (char === "'") {
          index += 1;
          break;
        } else {
          label += char;
          index += 1;
        }
      }
      skipSpaceAndComments();
      return label.trim();
    }
    while (index < source.length && ![":", ",", "(", ")"].includes(source[index])) {
      if (source[index] === "[") {
        skipSpaceAndComments();
      } else {
        label += source[index];
        index += 1;
      }
    }
    return label.trim();
  }

  function readLength() {
    skipSpaceAndComments();
    if (source[index] !== ":") return 0;
    index += 1;
    skipSpaceAndComments();
    let token = "";
    while (index < source.length && ![",", "(", ")"].includes(source[index])) {
      if (source[index] === "[") {
        skipSpaceAndComments();
      } else {
        token += source[index];
        index += 1;
      }
    }
    const value = Number.parseFloat(token.trim());
    return Number.isFinite(value) ? Math.max(value, 0) : 0;
  }

  function node(children = []) {
    return {
      id: `n${nextId += 1}`,
      name: "",
      branchLength: 0,
      children,
      depth: 0,
      x: 0,
      y: 0,
      meta: {},
    };
  }

  function parseNode() {
    skipSpaceAndComments();
    if (source[index] === "(") {
      index += 1;
      const children = [];
      while (index < source.length) {
        children.push(parseNode());
        skipSpaceAndComments();
        if (source[index] === ",") {
          index += 1;
          continue;
        }
        if (source[index] === ")") {
          index += 1;
          break;
        }
      }
      const current = node(children);
      current.name = readLabel();
      current.branchLength = readLength();
      return current;
    }
    const current = node([]);
    current.name = readLabel();
    current.branchLength = readLength();
    return current;
  }

  const root = parseNode();
  if (!root.children.length && !root.name) {
    throw new Error("Could not parse the Newick tree.");
  }
  return root;
}

function extractNewickSource(text) {
  const source = String(text || "").trim();
  if (!source) return "";
  if (source.startsWith("(")) return source;
  const treeMatch = source.match(/\btree\s+[^=]+=\s*(?:\[[^\]]*\]\s*)?([\s\S]*?);/i);
  if (treeMatch?.[1]) return `${treeMatch[1]};`;
  const firstTree = source.indexOf("(");
  const lastSemi = source.lastIndexOf(";");
  if (firstTree >= 0 && lastSemi > firstTree) {
    return source.slice(firstTree, lastSemi + 1);
  }
  return source;
}

function layoutNewickTree(root, metadataMap, zoom) {
  const nodes = [];
  const leaves = [];
  const edges = [];

  function annotate(node, parent = null, depth = 0, topologicalDepth = 0) {
    node.parent = parent;
    node.depth = depth;
    node.topologicalDepth = topologicalDepth;
    node.meta = metadataForTip(node.name, metadataMap);
    nodes.push(node);
    if (!node.children.length) leaves.push(node);
    node.children.forEach((child) => {
      edges.push({ id: `${node.id}-${child.id}`, parent: node, child });
      annotate(child, node, depth + (child.branchLength || 0), topologicalDepth + 1);
    });
  }

  annotate(root);

  let leafIndex = 0;
  function assignY(node) {
    if (!node.children.length) {
      node.leafIndex = leafIndex;
      leafIndex += 1;
      return node.leafIndex;
    }
    const childY = node.children.map(assignY);
    node.leafIndex = childY.reduce((sum, value) => sum + value, 0) / childY.length;
    return node.leafIndex;
  }
  assignY(root);

  const maxDepth = Math.max(...nodes.map((item) => item.depth), 0);
  const maxTopologicalDepth = Math.max(...nodes.map((item) => item.topologicalDepth), 1);
  const useBranchDepth = maxDepth > 0;
  const leafGap = Math.max(4, Math.min(24, leaves.length > 0 ? 2600 / leaves.length : 24)) * zoom.y;
  const topPad = 44;
  const leftPad = 40;
  const rightPad = 220;
  const bottomPad = 54;
  const plotWidth = Math.max(940, 1450 * zoom.x);
  const width = leftPad + plotWidth + rightPad;
  const height = Math.max(720, topPad + bottomPad + Math.max(1, leaves.length - 1) * leafGap);

  nodes.forEach((node) => {
    const xValue = useBranchDepth ? node.depth / maxDepth : node.topologicalDepth / maxTopologicalDepth;
    node.x = leftPad + xValue * plotWidth;
    node.y = topPad + node.leafIndex * leafGap;
  });

  return { root, nodes, leaves, edges, width, height, maxDepth, leafGap };
}

function buildTipMetadataMap(payload) {
  const map = new Map();
  (payload?.tips || []).forEach((item) => {
    [item.name, item.label_key, item.normalized_id].filter(Boolean).forEach((key) => {
      map.set(String(key).toLowerCase(), item);
    });
  });
  return map;
}

function metadataForTip(name, metadataMap) {
  const fallback = {
    group: "background",
    clade: "unassigned",
    subclade: "unassigned",
    collection_date: "",
  };
  if (!name) return fallback;
  for (const key of treeNameKeys(name)) {
    const item = metadataMap.get(key);
    if (item) return { ...fallback, ...item };
  }
  return fallback;
}

function treeNameKeys(name) {
  const raw = String(name || "").trim();
  return [raw, normalizeTreeName(raw), treeLabelKey(raw)]
    .filter(Boolean)
    .map((item) => item.toLowerCase());
}

function normalizeTreeName(name) {
  return String(name || "")
    .trim()
    .replace(/[\s/|:;,()[\]']+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "") || "seq";
}

function treeLabelKey(name) {
  return String(name || "")
    .trim()
    .replace(/["'()[\]]/g, "")
    .replace(/[\s/|:;,.\-]+/g, "_")
    .replace(/(__[A-Za-z0-9]+_)+$/g, "")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function tooltipText(node) {
  return [
    node.name,
    `group: ${node.meta.group || "background"}`,
    `clade: ${node.meta.clade || "unassigned"}`,
    `subclade: ${node.meta.subclade || node.meta.clade || "unassigned"}`,
    `collection date: ${node.meta.collection_date || "-"}`,
  ].join("\n");
}

function formatNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "-";
  if (Math.abs(numeric) >= 10) return numeric.toFixed(1);
  if (Math.abs(numeric) >= 1) return numeric.toFixed(2);
  return numeric.toPrecision(3);
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
