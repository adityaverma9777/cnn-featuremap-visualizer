import React, { Suspense, lazy, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ModelReasoning from "./ModelReasoning";
const Network3D = lazy(() => import("./Network3D"));

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const API_URL = `${API_BASE_URL.replace(/\/$/, "")}/predict`;
const VIEW_MODES = {
  ORIGINAL: "original",
  OVERLAY: "overlay",
  SIDE_BY_SIDE: "side-by-side",
};

function avg(values = []) {
  if (!Array.isArray(values) || values.length === 0) return 0;
  return values.reduce((sum, value) => sum + Number(value || 0), 0) / values.length;
}

function buildProbabilityData(result) {
  if (!result) return [];

  if (Array.isArray(result.class_probabilities)) {
    return result.class_probabilities
      .map((item) => ({
        className: item.class ?? item.label ?? "unknown",
        probability: Number(item.probability ?? 0),
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
  }

  if (result.probabilities && typeof result.probabilities === "object") {
    return Object.entries(result.probabilities)
      .map(([className, probability]) => ({ className, probability: Number(probability || 0) }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
  }

  if (result.prediction) {
    return [{ className: result.prediction, probability: Number(result.confidence || 0) }];
  }

  return [];
}

function buildActivationData(result) {
  if (!result?.activations) return [];

  return [
    { layer: "conv1", value: avg(result.activations.conv1) },
    { layer: "conv2", value: avg(result.activations.conv2) },
    { layer: "conv3", value: avg(result.activations.conv3) },
  ];
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState(VIEW_MODES.OVERLAY);
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const [pulseKey, setPulseKey] = useState(0);
  const [selectedFeature, setSelectedFeature] = useState(null);

  const probabilityData = useMemo(() => buildProbabilityData(result), [result]);
  const activationData = useMemo(() => buildActivationData(result), [result]);
  const hasResult = Boolean(result);

  const heatmapSrc = useMemo(() => {
    if (!result?.heatmap) return "";
    return `data:image/png;base64,${result.heatmap}`;
  }, [result]);

  function onFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setResult(null);
    setError("");
    setSelectedFeature(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);
    setError("");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const details = await response.text();
        throw new Error(details || "Prediction request failed.");
      }

      const payload = await response.json();
      setResult(payload);
      setPulseKey((value) => value + 1);
      setSelectedFeature(null);
    } catch (err) {
      setError(err.message || "Unexpected error during inference.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="panel top-panel">
        <h1>Real-Time Explainable AI Interface</h1>
        <p>
          Upload an image and run local inference. Prediction, Grad-CAM, charts, and 3D network update from the same
          backend response.
        </p>

        <form className="upload-form" onSubmit={onSubmit}>
          <input type="file" accept="image/*" onChange={onFileChange} />
          <button type="submit" disabled={loading}>
            {loading ? "Run" : "Run"}
          </button>
        </form>

        <p className="local-note">Frontend API target: {API_BASE_URL}</p>
        {error && <p className="error">{error}</p>}

        <div className="top-layout">
          <article className="card top-card">
            <h2>Uploaded Image</h2>
            {previewUrl ? <img src={previewUrl} alt="Uploaded preview" className="image" /> : <p>No image selected yet.</p>}
          </article>

          <article className="card top-card">
            <h2>Prediction Result</h2>
            {hasResult ? (
              <div className="result-summary">
                <p className="result-main">{result.prediction}</p>
                <p>Confidence: {(Number(result.confidence || 0) * 100).toFixed(2)}%</p>
                <p>Top classes available: {probabilityData.length}</p>
              </div>
            ) : (
              <p>Run prediction to view class, confidence, and explainability views.</p>
            )}
          </article>
        </div>
      </section>

      <section className="card middle-card">
        <h2>Grad-CAM Heatmap</h2>
        <div className="mode-controls" role="tablist" aria-label="Visualization mode">
          <button
            type="button"
            className={viewMode === VIEW_MODES.ORIGINAL ? "mode-btn active" : "mode-btn"}
            onClick={() => setViewMode(VIEW_MODES.ORIGINAL)}
          >
            Original image
          </button>
          <button
            type="button"
            className={viewMode === VIEW_MODES.OVERLAY ? "mode-btn active" : "mode-btn"}
            onClick={() => setViewMode(VIEW_MODES.OVERLAY)}
          >
            Heatmap overlay
          </button>
          <button
            type="button"
            className={viewMode === VIEW_MODES.SIDE_BY_SIDE ? "mode-btn active" : "mode-btn"}
            onClick={() => setViewMode(VIEW_MODES.SIDE_BY_SIDE)}
          >
            Side-by-side
          </button>
        </div>

        {viewMode === VIEW_MODES.OVERLAY && (
          <div className="opacity-wrap">
            <label htmlFor="opacity-range">Heatmap opacity: {Math.round(overlayOpacity * 100)}%</label>
            <input
              id="opacity-range"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={overlayOpacity}
              onChange={(event) => setOverlayOpacity(Number(event.target.value))}
              disabled={!heatmapSrc || !previewUrl}
            />
          </div>
        )}

        {!previewUrl && <p>No image available yet.</p>}
        {previewUrl && viewMode === VIEW_MODES.ORIGINAL && (
          <img src={previewUrl} alt="Original uploaded" className="image" />
        )}

        {previewUrl && viewMode === VIEW_MODES.OVERLAY && (
          <div className="overlay-stage">
            <img src={previewUrl} alt="Original uploaded" className="overlay-image" />
            {heatmapSrc ? (
              <img
                src={heatmapSrc}
                alt="Grad-CAM heatmap overlay"
                className="overlay-image heatmap"
                style={{ opacity: overlayOpacity }}
              />
            ) : (
              <div className="overlay-placeholder">Heatmap appears after prediction.</div>
            )}
            {selectedFeature?.heatmap_anchor && (
              <div
                className="heatmap-anchor-marker"
                style={{
                  left: `${selectedFeature.heatmap_anchor.x * 100}%`,
                  top: `${selectedFeature.heatmap_anchor.y * 100}%`,
                }}
              />
            )}
          </div>
        )}

        {previewUrl && viewMode === VIEW_MODES.SIDE_BY_SIDE && (
          <div className="compare-grid">
            <div>
              <p className="compare-label">Original</p>
              <img src={previewUrl} alt="Original uploaded" className="image" />
            </div>
            <div>
              <p className="compare-label">Heatmap</p>
              {heatmapSrc ? (
                <div className="heatmap-anchor-stage">
                  <img src={heatmapSrc} alt="Grad-CAM heatmap" className="image" />
                  {selectedFeature?.heatmap_anchor && (
                    <div
                      className="heatmap-anchor-marker"
                      style={{
                        left: `${selectedFeature.heatmap_anchor.x * 100}%`,
                        top: `${selectedFeature.heatmap_anchor.y * 100}%`,
                      }}
                    />
                  )}
                </div>
              ) : (
                <div className="image placeholder">Heatmap appears after prediction.</div>
              )}
            </div>
          </div>
        )}
      </section>

      <section className="charts-layout">
        <article className="card chart-card">
          <h2>Top Class Probabilities</h2>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={probabilityData} margin={{ top: 10, right: 20, left: 0, bottom: 15 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="className" interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis domain={[0, 1]} />
                <Tooltip formatter={(value) => Number(value).toFixed(4)} />
                <Legend />
                <Bar dataKey="probability" fill="#2f6f4f" name="Probability" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="card chart-card">
          <h2>Average Layer Activations</h2>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activationData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="layer" />
                <YAxis />
                <Tooltip formatter={(value) => Number(value).toFixed(4)} />
                <Legend />
                <Bar dataKey="value" fill="#d78533" name="Avg Activation" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </section>

      <section className="card network-card">
        <h2>Interactive 3D Neural Network</h2>
        <p className="network-note">
          Layered view of input, conv1, conv2, conv3, and output. Signals animate from input to output after each
          prediction.
        </p>
        {hasResult ? (
          <Suspense
            fallback={
              <div className="network3d-wrap network3d-fallback">
                <p>Loading 3D visualization...</p>
              </div>
            }
          >
            <Network3D
              result={result}
              pulseKey={pulseKey}
              inputImageSrc={previewUrl}
              highlightedMap={
                selectedFeature
                  ? {
                      layer: "conv3",
                      channel: selectedFeature.conv3_channel,
                    }
                  : null
              }
            />
          </Suspense>
        ) : (
          <div className="network3d-wrap network3d-fallback">
            <p>Run a prediction to render the interactive 3D network.</p>
          </div>
        )}
      </section>

      <ModelReasoning
        result={result}
        selectedFeature={selectedFeature}
        onFeatureSelect={(feature) => setSelectedFeature(feature)}
      />
    </main>
  );
}
