import React, { useMemo } from "react";

function formatNumber(value, digits = 4) {
  const number = Number(value);
  if (Number.isNaN(number)) return "0";
  return number.toFixed(digits);
}

function MatrixTable({ title, matrix }) {
  if (!Array.isArray(matrix) || matrix.length === 0) return null;

  return (
    <div className="matrix-block">
      <p className="matrix-title">{title}</p>
      <table className="matrix-table">
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={`${title}-${rowIndex}`}>
              {row.map((cell, colIndex) => (
                <td key={`${title}-${rowIndex}-${colIndex}`}>{formatNumber(cell, 3)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FeatureMapStrip({ layerName, maps }) {
  const subset = Array.isArray(maps) ? maps.slice(0, 8) : [];
  if (subset.length === 0) return null;

  return (
    <div className="reasoning-map-strip">
      <p className="reasoning-subtitle">{layerName.toUpperCase()} sample feature maps</p>
      <div className="reasoning-map-grid">
        {subset.map((map) => (
          <figure key={`${layerName}-${map.channel}`} className="reasoning-map-item">
            <img src={`data:image/png;base64,${map.preview_image}`} alt={`${layerName} channel ${map.channel}`} />
            <figcaption>ch {map.channel}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

export default function ModelReasoning({
  result,
  selectedFeature,
  onFeatureSelect,
}) {
  const explainability = result?.explainability;

  const topTerms = explainability?.fully_connected?.top_contributions ?? [];
  const linkedFeatures = explainability?.fully_connected?.linked_features ?? [];

  const fcScoreCheck = useMemo(() => {
    if (!explainability?.fully_connected) return null;
    const sum = Number(explainability.fully_connected.sum_products || 0);
    const bias = Number(explainability.fully_connected.bias || 0);
    return sum + bias;
  }, [explainability]);

  if (!explainability) {
    return (
      <section className="card reasoning-card">
        <h2>Model Reasoning and Mathematical Breakdown</h2>
        <p>Run a prediction to view the full step-by-step mathematical reasoning.</p>
      </section>
    );
  }

  return (
    <section className="card reasoning-card">
      <h2>Model Reasoning and Mathematical Breakdown</h2>
      <p className="reasoning-intro">
        This section follows the actual forward pass and uses real tensors produced by the model.
      </p>

      <details open>
        <summary>1. Input Representation</summary>
        <div className="reasoning-content">
          <img
            src={`data:image/png;base64,${explainability.input_representation.resized_image}`}
            alt="Resized 32x32 input"
            className="reasoning-image"
          />
          <p>
            Shape: {explainability.input_representation.shape.join(" x ")} | Normalized range: [
            {formatNumber(explainability.input_representation.normalized_range.min, 3)}, {" "}
            {formatNumber(explainability.input_representation.normalized_range.max, 3)}]
          </p>
          <MatrixTable title="8x8 normalized intensity matrix" matrix={explainability.input_representation.pixel_matrix} />
        </div>
      </details>

      <details>
        <summary>2. Convolution Operation (Conv1, Conv2, Conv3)</summary>
        <div className="reasoning-content">
          <FeatureMapStrip layerName="conv1" maps={result?.feature_maps?.conv1} />
          <FeatureMapStrip layerName="conv2" maps={result?.feature_maps?.conv2} />
          <FeatureMapStrip layerName="conv3" maps={result?.feature_maps?.conv3} />

          {Object.entries(explainability.convolution_examples).map(([layer, sample]) => (
            <article key={layer} className="reasoning-block">
              <h3>{layer.toUpperCase()} sampled activation computation</h3>
              <p>{sample.equation}</p>
              <p>
                Output channel {sample.selected_output_channel}, input channel {sample.selected_input_channel},
                location ({sample.location.x}, {sample.location.y})
              </p>
              <div className="matrix-row">
                <MatrixTable title="Input patch" matrix={sample.input_patch} />
                <MatrixTable title="Kernel" matrix={sample.kernel} />
                <MatrixTable title="Kernel * Patch" matrix={sample.elementwise_product} />
              </div>
              <p>
                Single channel contribution: {formatNumber(sample.single_channel_sum)} | Bias: {formatNumber(sample.bias)}
              </p>
              <p>
                Pre-activation output: {formatNumber(sample.pre_activation)} | Post-ReLU output: {formatNumber(sample.relu_output)}
              </p>
            </article>
          ))}
        </div>
      </details>

      <details>
        <summary>3. Activation Function (ReLU)</summary>
        <div className="reasoning-content">
          <p>ReLU(x) = max(0, x)</p>
          {Object.entries(explainability.relu_examples).map(([layer, pair]) => (
            <p key={layer}>
              {layer.toUpperCase()}: ReLU({formatNumber(pair.before)}) = {formatNumber(pair.after)}
            </p>
          ))}
        </div>
      </details>

      <details>
        <summary>4. Pooling</summary>
        <div className="reasoning-content">
          <p>
            Max pooling reduces shape from {explainability.pooling.input_shape.join(" x ")} to {" "}
            {explainability.pooling.output_shape.join(" x ")}
          </p>
          <MatrixTable title="Example 2x2 pooling window" matrix={explainability.pooling.example_window} />
          <p>max(window) = {formatNumber(explainability.pooling.example_max)}</p>
        </div>
      </details>

      <details>
        <summary>5. Flattening</summary>
        <div className="reasoning-content">
          <p>
            Tensor {explainability.flattening.from_shape.join(" x ")} {"->"} Vector length {explainability.flattening.to_length}
          </p>
          <p className="mono-wrap">
            Sample vector values: {explainability.flattening.sample_values.map((value) => formatNumber(value, 3)).join(", ")}
          </p>
        </div>
      </details>

      <details>
        <summary>6. Fully Connected Layer (Critical)</summary>
        <div className="reasoning-content">
          <p>{explainability.fully_connected.equation}</p>
          <p>
            Target class: <strong>{explainability.fully_connected.target_class}</strong>
          </p>
          <div className="reasoning-terms">
            {topTerms.map((term) => (
              <div key={term.feature_index} className="reasoning-term-row">
                <span>
                  ({formatNumber(term.feature_value)} x {formatNumber(term.weight)}) = {formatNumber(term.product)}
                </span>
              </div>
            ))}
          </div>
          <p>
            Sum products = {formatNumber(explainability.fully_connected.sum_products)} | Bias = {formatNumber(
              explainability.fully_connected.bias
            )}
          </p>
          <p>
            Score ({explainability.fully_connected.target_class}) = {formatNumber(fcScoreCheck)} ≈ {formatNumber(
              explainability.fully_connected.score
            )}
          </p>
        </div>
      </details>

      <details>
        <summary>7. Softmax Conversion</summary>
        <div className="reasoning-content">
          <p>{explainability.softmax.equation}</p>
          <p>
            Numerator exp(score) = {formatNumber(explainability.softmax.numerator, 6)} | Denominator = {" "}
            {formatNumber(explainability.softmax.denominator, 6)}
          </p>
          <p>
            P({explainability.softmax.predicted_class}) = {formatNumber(explainability.softmax.probability, 6)}
          </p>
        </div>
      </details>

      <details>
        <summary>8. Final Output Mapping</summary>
        <div className="reasoning-content">
          <p>
            Computed probability for <strong>{explainability.final_output.predicted_class}</strong> = {" "}
            {formatNumber(explainability.final_output.probability, 6)}
          </p>
          <p>
            Highest probability class: <strong>{explainability.final_output.max_probability_class}</strong>
          </p>
        </div>
      </details>

      <details>
        <summary>9. Visual Linking (click to highlight)</summary>
        <div className="reasoning-content">
          <p>Click a linked feature to highlight its conv3 feature map and anchor region in the heatmap.</p>
          <div className="reasoning-terms">
            {linkedFeatures.map((feature) => (
              <button
                type="button"
                key={feature.flatten_index}
                className={
                  selectedFeature?.flatten_index === feature.flatten_index
                    ? "reasoning-link-btn active"
                    : "reasoning-link-btn"
                }
                onClick={() => onFeatureSelect(feature)}
              >
                f[{feature.flatten_index}] = {formatNumber(feature.value)} {"->"} conv3 ch {feature.conv3_channel} @ (
                {feature.conv3_position.row}, {feature.conv3_position.col})
              </button>
            ))}
          </div>
        </div>
      </details>
    </section>
  );
}
