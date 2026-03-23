import React, { useMemo } from "react";
import { BlockMath, InlineMath } from "react-katex";

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

function MathBlock({ latex }) {
  return <BlockMath math={latex} />;
}

function StepBlock({ title, children }) {
  return (
    <div className="reasoning-step-block">
      <p className="reasoning-step-title">{title}</p>
      {children}
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
  const L = String.raw;
  const explainability = result?.explainability;

  const linkedFeatures = explainability?.fully_connected?.linked_features ?? [];
  const flattenTerms = explainability?.fully_connected?.flatten_top_contributions ?? [];

  const selectedTerm = useMemo(() => {
    if (!explainability?.fully_connected) return null;
    if (selectedFeature?.flatten_index !== undefined) {
      const byIndex = flattenTerms.find((item) => item.flatten_index === selectedFeature.flatten_index);
      if (byIndex) return byIndex;
      const linked = linkedFeatures.find((item) => item.flatten_index === selectedFeature.flatten_index);
      if (linked) return linked;
    }
    return flattenTerms[0] ?? linkedFeatures[0] ?? null;
  }, [explainability, flattenTerms, linkedFeatures, selectedFeature]);

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
      <h2>Mathematical Derivation</h2>
      <p className="reasoning-intro">
        All equations below are rendered in LaTeX and instantiated using values from the current model inference.
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
          <MathBlock latex={L`x \in \mathbb{R}^{3 \times 32 \times 32}`} />
          <MathBlock latex={L`x_{\mathrm{norm}} = \frac{x - \mu}{\sigma}, \quad \mu = 0.5, \quad \sigma = 0.5`} />
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
              <MathBlock latex={L`y_{x,y}^{(k)} = \sum_{c=1}^{C}\sum_{i=1}^{3}\sum_{j=1}^{3} W_{c,i,j}^{(k)} X_{c,x+i,y+j} + b^{(k)}`} />
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
              <MathBlock
                latex={L`z = ${formatNumber(sample.single_channel_sum)} + ${formatNumber(sample.bias)} = ${formatNumber(
                  sample.pre_activation
                )}, \quad a = \max(0, z) = ${formatNumber(sample.relu_output)}`}
              />
            </article>
          ))}
        </div>
      </details>

      <details>
        <summary>3. Activation Function (ReLU)</summary>
        <div className="reasoning-content">
          <MathBlock latex={L`\mathrm{ReLU}(z) = \max(0, z)`} />
          {Object.entries(explainability.relu_examples).map(([layer, pair]) => (
            <MathBlock
              key={layer}
              latex={L`${layer}:\ \mathrm{ReLU}(${formatNumber(pair.before)}) = ${formatNumber(pair.after)}`}
            />
          ))}
        </div>
      </details>

      <details>
        <summary>4. Pooling</summary>
        <div className="reasoning-content">
          <MathBlock latex={L`p_{x,y}^{(k)} = \max\{a_{2x,2y}^{(k)}, a_{2x+1,2y}^{(k)}, a_{2x,2y+1}^{(k)}, a_{2x+1,2y+1}^{(k)}\}`} />
          <p>
            Max pooling reduces shape from {explainability.pooling.input_shape.join(" x ")} to {" "}
            {explainability.pooling.output_shape.join(" x ")}
          </p>
          <MatrixTable title="Example 2x2 pooling window" matrix={explainability.pooling.example_window} />
          <MathBlock latex={L`\max(\text{window}) = ${formatNumber(explainability.pooling.example_max)}`} />
        </div>
      </details>

      <details>
        <summary>5. Flattening</summary>
        <div className="reasoning-content">
          <MathBlock latex={L`\mathrm{vec}:\ \mathbb{R}^{${explainability.flattening.from_shape.join(" \\\\times ")}} \to \mathbb{R}^{${explainability.flattening.to_length}}`} />
          <p className="mono-wrap">
            Sample vector values: {explainability.flattening.sample_values.map((value) => formatNumber(value, 3)).join(", ")}
          </p>
        </div>
      </details>

      <details>
        <summary>6. Fully Connected Layer (Critical)</summary>
        <div className="reasoning-content">
          <MathBlock latex={L`s_{${explainability.fully_connected.target_class}} = \sum_{i=1}^{n} \tilde{w}_i x_i + \tilde{b}`} />
          {selectedTerm && (
            <>
              <StepBlock title="Inputs">
                <MathBlock latex={L`x_{${selectedTerm.flatten_index}} = ${formatNumber(selectedTerm.feature_value ?? selectedTerm.value)}`} />
                <MathBlock latex={L`\tilde{w}_{${selectedTerm.flatten_index}} = ${formatNumber(selectedTerm.effective_weight)}`} />
              </StepBlock>
              <StepBlock title="Formula">
                <MathBlock latex={L`t_i = \tilde{w}_i x_i`} />
              </StepBlock>
              <StepBlock title="Substitution">
                <MathBlock
                  latex={L`t_{${selectedTerm.flatten_index}} = ${formatNumber(selectedTerm.effective_weight)} \times ${formatNumber(
                    selectedTerm.feature_value ?? selectedTerm.value
                  )}`}
                />
              </StepBlock>
              <StepBlock title="Simplification">
                <MathBlock latex={L`t_{${selectedTerm.flatten_index}} = ${formatNumber(selectedTerm.product)}`} />
              </StepBlock>
              <StepBlock title="Final Result">
                <MathBlock
                  latex={L`s = ${formatNumber(explainability.fully_connected.effective_sum_products)} + ${formatNumber(
                    explainability.fully_connected.effective_bias
                  )} = ${formatNumber(explainability.fully_connected.score)}`}
                />
              </StepBlock>
            </>
          )}

          <div className="reasoning-terms">
            {(explainability.fully_connected.flatten_top_contributions ?? []).slice(0, 10).map((term) => (
              <div key={term.flatten_index} className="reasoning-term-row">
                <InlineMath
                  math={L`(${formatNumber(term.feature_value)} \times ${formatNumber(term.effective_weight)}) = ${formatNumber(
                    term.product
                  )}`}
                />
              </div>
            ))}
          </div>
        </div>
      </details>

      <details>
        <summary>7. Softmax Conversion</summary>
        <div className="reasoning-content">
          <MathBlock latex={L`P(c) = \frac{e^{s_c}}{\sum_{j} e^{s_j}}`} />
          <MathBlock
            latex={L`P(${explainability.softmax.predicted_class}) = \frac{${formatNumber(
              explainability.softmax.numerator,
              6
            )}}{${formatNumber(explainability.softmax.denominator, 6)}} = ${formatNumber(
              explainability.softmax.probability,
              6
            )}`}
          />
        </div>
      </details>

      <details>
        <summary>8. Final Output Mapping</summary>
        <div className="reasoning-content">
          <MathBlock
            latex={L`\hat{y} = \arg\max_{c}\ P(c) = ${explainability.final_output.max_probability_class}, \quad P(\hat{y}) = ${formatNumber(
              explainability.final_output.probability,
              6
            )}`}
          />
        </div>
      </details>

      <details>
        <summary>9. Dynamic Linking and Derivatives</summary>
        <div className="reasoning-content">
          {selectedTerm && (
            <>
              <MathBlock
                latex={L`\frac{\partial s}{\partial x_{${selectedTerm.flatten_index}}} = \tilde{w}_{${selectedTerm.flatten_index}} = ${formatNumber(
                  selectedTerm.derivative_dscore_dxi ?? selectedTerm.effective_weight
                )}`}
              />
              <MathBlock latex={L`\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}`} />
            </>
          )}
          <p>Click a linked feature to update the derivation for that node only.</p>
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
                <InlineMath
                  math={L`x_{${feature.flatten_index}} = ${formatNumber(feature.value)} \Rightarrow \text{ch } ${feature.conv3_channel}(${feature.conv3_position.row}, ${feature.conv3_position.col})`}
                />
              </button>
            ))}
          </div>
        </div>
      </details>
    </section>
  );
}
