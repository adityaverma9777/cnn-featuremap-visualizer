import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const DEBUG_RENDER_HELPERS = false;
const MAX_CONNECTIONS = 900;
const WEIGHT_THRESHOLD = 0.4;
const TOPK_PIXEL_TO_CONV1 = 3;
const TOPK_ADJACENT = 3;
const TOPK_CONV3_TO_OUTPUT = 5;
const DEBUG_SINGLE_TRACE = false;

const NETWORK_TARGET = new THREE.Vector3(0, 0, 1);
const DEFAULT_CAMERA_AZIMUTH = THREE.MathUtils.degToRad(72);
const DEFAULT_CAMERA_POLAR = THREE.MathUtils.degToRad(78);
const DEFAULT_CAMERA_DISTANCE = 26;
const MIN_CAMERA_DISTANCE = 10;
const MAX_CAMERA_DISTANCE = 42;
const ZOOM_STEP = 2.5;

const LAYER_DEPTH = {
  input: -8,
  conv1: -2,
  conv2: 2,
  conv3: 6,
  output: 10,
};

const REVEAL_START = {
  input: 0.0,
  conv1: 0.2,
  conv2: 0.45,
  conv3: 0.7,
  output: 0.9,
};

function hashSeed(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function seedToRange(seed, min, max) {
  const value = (Math.sin(seed * 12.9898) * 43758.5453) % 1;
  const normalized = value - Math.floor(value);
  return min + normalized * (max - min);
}

function createTextureFromValues(values, pixelScale = 10) {
  const rows = Array.isArray(values) ? values : [[0]];
  const height = Math.max(rows.length, 1);
  const width = Math.max(Array.isArray(rows[0]) ? rows[0].length : 1, 1);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);
  let offset = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = Number(rows[y]?.[x] ?? 0);
      const shade = Math.max(0, Math.min(255, Math.round(value * 255)));
      imageData.data[offset] = shade;
      imageData.data[offset + 1] = shade;
      imageData.data[offset + 2] = shade;
      imageData.data[offset + 3] = 255;
      offset += 4;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  const scaled = document.createElement("canvas");
  scaled.width = width * pixelScale;
  scaled.height = height * pixelScale;
  const scaledCtx = scaled.getContext("2d");
  scaledCtx.imageSmoothingEnabled = false;
  scaledCtx.drawImage(canvas, 0, 0, scaled.width, scaled.height);

  const texture = new THREE.CanvasTexture(scaled);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.needsUpdate = true;

  return {
    texture,
    dataUrl: scaled.toDataURL("image/png"),
  };
}

function createOutputTexture(classProbabilities) {
  const bars = Array.isArray(classProbabilities) ? classProbabilities.slice(0, 10) : [];
  const width = 320;
  const height = 120;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#101218";
  ctx.fillRect(0, 0, width, height);

  const barWidth = (width - 40) / Math.max(bars.length, 1);
  bars.forEach((entry, index) => {
    const p = Math.max(0, Math.min(1, Number(entry?.probability ?? 0)));
    const h = Math.round((height - 30) * p);
    const x = 20 + index * barWidth;
    const y = height - 15 - h;
    ctx.fillStyle = "#59d18c";
    ctx.fillRect(x, y, Math.max(3, barWidth - 4), h);
  });

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

function buildGridPositions(count, cols = 4, spacing = 1.3) {
  const rows = Math.ceil(count / cols);
  return Array.from({ length: count }, (_, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    const x = (col - (cols - 1) / 2) * spacing;
    const y = ((rows - 1) / 2 - row) * spacing;
    const u = cols > 1 ? col / (cols - 1) : 0.5;
    const v = rows > 1 ? row / (rows - 1) : 0.5;
    return { x, y, u, v, row, col };
  });
}

function createCurvePoints(start, end, seedText) {
  const seed = hashSeed(seedText);
  const midZ = (start[2] + end[2]) / 2;
  const lift = seedToRange(seed, 0.1, 0.9);
  const spreadX = seedToRange(seed + 17, -0.35, 0.35);
  const spreadY = seedToRange(seed + 41, -0.35, 0.35);

  const c1 = new THREE.Vector3(
    THREE.MathUtils.lerp(start[0], end[0], 0.28) + spreadX,
    THREE.MathUtils.lerp(start[1], end[1], 0.28) + spreadY,
    THREE.MathUtils.lerp(start[2], midZ, 0.6) + lift
  );
  const c2 = new THREE.Vector3(
    THREE.MathUtils.lerp(start[0], end[0], 0.72) - spreadX,
    THREE.MathUtils.lerp(start[1], end[1], 0.72) - spreadY,
    THREE.MathUtils.lerp(midZ, end[2], 0.6) - lift * 0.4
  );

  const curve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(...start),
    c1,
    c2,
    new THREE.Vector3(...end),
  ]);

  return curve.getPoints(22);
}

function ConnectionCurve({ connection, isActive, progressRef }) {
  const lineRef = useRef(null);
  const points = useMemo(
    () => createCurvePoints(connection.start, connection.end, connection.id),
    [connection.end, connection.id, connection.start]
  );

  useFrame((_, delta) => {
    if (!lineRef.current?.material) return;

    const reveal = THREE.MathUtils.smoothstep(
      progressRef.current,
      connection.revealStart,
      connection.revealStart + 0.22
    );

    const pulse = 0.82 + 0.18 * Math.sin(performance.now() * 0.0014 + connection.seed);
    const targetOpacity = reveal * connection.opacity * pulse * (isActive ? 1 : 0.1);
    lineRef.current.material.opacity = THREE.MathUtils.lerp(
      lineRef.current.material.opacity,
      targetOpacity,
      Math.min(1, delta * 9)
    );

    if (Object.prototype.hasOwnProperty.call(lineRef.current.material, "dashOffset")) {
      lineRef.current.material.dashOffset -= delta * (0.7 + Math.abs(connection.strength) * 0.6);
    }
  });

  return (
    <Line
      ref={lineRef}
      points={points}
      color={connection.color}
      transparent
      opacity={0}
      lineWidth={connection.width}
      dashed
      dashScale={1}
      dashSize={0.28}
      gapSize={0.22}
    />
  );
}

function FeatureMapPlane({ map, mapId, layerKey, position, progressRef, hovered, highlighted, onHover, onOut, onSelect }) {
  const meshRef = useRef(null);
  const materialRef = useRef(null);

  const { texture, highResUrl } = useMemo(() => {
    const low = createTextureFromValues(map.values, 10);
    const high = createTextureFromValues(map.values, 28);
    return { texture: low.texture, highResUrl: high.dataUrl };
  }, [map.values]);

  useEffect(() => {
    return () => {
      texture.dispose();
    };
  }, [texture]);

  useFrame((_, delta) => {
    if (!meshRef.current || !materialRef.current) return;
    const reveal = THREE.MathUtils.smoothstep(
      progressRef.current,
      REVEAL_START[layerKey],
      REVEAL_START[layerKey] + 0.22
    );

    const targetScale = hovered || highlighted ? 1.22 : 1;
    const currentScale = meshRef.current.scale.x;
    const smoothed = THREE.MathUtils.lerp(currentScale, targetScale, Math.min(1, delta * 9));
    meshRef.current.scale.setScalar(smoothed);
    materialRef.current.opacity = reveal * 0.98;
    materialRef.current.color.set(highlighted ? "#ffe680" : "#ffffff");
  });

  return (
    <mesh
      ref={meshRef}
      frustumCulled={false}
      position={position}
      onPointerOver={() => onHover(mapId)}
      onPointerOut={onOut}
      onClick={() => onSelect(`${layerKey.toUpperCase()} channel ${map.channel}`, highResUrl)}
    >
      <planeGeometry args={[0.95, 0.95]} />
      <meshBasicMaterial
        ref={materialRef}
        map={texture}
        transparent
        opacity={0}
        toneMapped={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function InputPlane({ inputImageSrc, progressRef }) {
  const texture = useLoader(THREE.TextureLoader, inputImageSrc);
  const materialRef = useRef(null);

  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
  }, [texture]);

  useFrame(() => {
    if (!materialRef.current) return;
    const reveal = THREE.MathUtils.smoothstep(progressRef.current, REVEAL_START.input, REVEAL_START.input + 0.12);
    materialRef.current.opacity = reveal;
  });

  return (
    <mesh position={[0, 0, LAYER_DEPTH.input]} frustumCulled={false}>
      <planeGeometry args={[3.4, 3.4]} />
      <meshBasicMaterial
        ref={materialRef}
        map={texture}
        transparent
        opacity={0}
        toneMapped={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function OutputPlane({ classProbabilities, progressRef, onOutputHover, onOutputOut }) {
  const materialRef = useRef(null);
  const texture = useMemo(() => createOutputTexture(classProbabilities), [classProbabilities]);

  useEffect(() => {
    return () => {
      texture.dispose();
    };
  }, [texture]);

  useFrame(() => {
    if (!materialRef.current) return;
    const reveal = THREE.MathUtils.smoothstep(progressRef.current, REVEAL_START.output, 1);
    materialRef.current.opacity = reveal;
  });

  return (
    <mesh
      position={[0, 0, LAYER_DEPTH.output]}
      frustumCulled={false}
      onPointerOver={onOutputHover}
      onPointerOut={onOutputOut}
    >
      <planeGeometry args={[3.8, 1.5]} />
      <meshBasicMaterial
        ref={materialRef}
        map={texture}
        transparent
        opacity={0}
        toneMapped={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function CameraPoseController({ controlsRef, desiredDistance }) {
  const { camera } = useThree();
  const initialized = useRef(false);
  const spherical = useMemo(() => new THREE.Spherical(), []);
  const vector = useMemo(() => new THREE.Vector3(), []);

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (!controls) return;

    if (!initialized.current) {
      controls.target.copy(NETWORK_TARGET);
      spherical.set(DEFAULT_CAMERA_DISTANCE, DEFAULT_CAMERA_POLAR, DEFAULT_CAMERA_AZIMUTH);
      vector.setFromSpherical(spherical);
      camera.position.copy(NETWORK_TARGET).add(vector);
      camera.lookAt(NETWORK_TARGET);
      controls.update();
      initialized.current = true;
      return;
    }

    vector.copy(camera.position).sub(controls.target);
    spherical.setFromVector3(vector);
    spherical.radius = THREE.MathUtils.lerp(
      spherical.radius,
      THREE.MathUtils.clamp(desiredDistance, MIN_CAMERA_DISTANCE, MAX_CAMERA_DISTANCE),
      Math.min(1, delta * 6)
    );

    vector.setFromSpherical(spherical);
    camera.position.copy(controls.target).add(vector);
    camera.lookAt(controls.target);
    controls.update();
  });

  return null;
}

function buildConnectionGraph({ pixelAnchors, mapNodesByLayer, explainability }) {
  const candidates = [];

  function addCandidate({ sourceId, targetId, start, end, strength, revealStart }) {
    const normalizedStrength = Math.tanh(Number(strength || 0));
    const absStrength = Math.abs(normalizedStrength);
    if (absStrength < WEIGHT_THRESHOLD) return;

    const id = `${sourceId}->${targetId}`;
    const width = 0.2 + absStrength * 0.85;
    const opacity = 0.15 + absStrength * 0.75;

    candidates.push({
      id,
      sourceId,
      targetId,
      start,
      end,
      strength: normalizedStrength,
      weightMagnitude: absStrength,
      opacity,
      width,
      color: normalizedStrength >= 0 ? "#35c978" : "#de5757",
      revealStart,
      seed: hashSeed(id) % 1000,
    });
  }

  const conv1 = mapNodesByLayer.conv1;
  const conv2 = mapNodesByLayer.conv2;
  const conv3 = mapNodesByLayer.conv3;

  pixelAnchors.forEach((pixel) => {
    const candidates = conv1
      .map((node) => {
        const dist = Math.hypot(pixel.u - node.u, pixel.v - node.v);
        return { node, dist, score: 1.15 - dist + node.activation * 0.55 };
      })
      .filter((entry) => entry.dist < 0.4)
      .sort((a, b) => b.score - a.score)
      .slice(0, TOPK_PIXEL_TO_CONV1);

    (candidates.length ? candidates : conv1.slice(0, 1).map((node) => ({ node, dist: 0.8, score: 0 }))).forEach((entry) => {
      const influence = Math.max(0.05, 1 - entry.dist);
      addCandidate({
        sourceId: pixel.id,
        targetId: entry.node.id,
        start: pixel.position,
        end: entry.node.position,
        strength: entry.node.signedMean * influence * pixel.activation,
        revealStart: REVEAL_START.conv1,
      });
    });
  });

  function connectAdjacent(leftNodes, rightNodes, revealStart) {
    leftNodes.forEach((left) => {
      const candidates = rightNodes
        .map((right) => {
          const dist = Math.hypot(left.u - right.u, left.v - right.v);
          const sim = 1 - Math.min(1, dist / 0.9);
          const signed = (left.signedMean + right.signedMean) / 2;
          return {
            right,
            dist,
            sim,
            signed,
            score: sim * 0.8 + Math.abs(right.activation) * 0.2,
          };
        })
        .filter((entry) => entry.dist < 0.62)
        .sort((a, b) => b.score - a.score)
        .slice(0, TOPK_ADJACENT);

      candidates.forEach((entry) => {
        addCandidate({
          sourceId: left.id,
          targetId: entry.right.id,
          start: left.position,
          end: entry.right.position,
          strength: entry.signed * Math.max(0.05, entry.sim),
          revealStart,
        });
      });
    });
  }

  connectAdjacent(conv1, conv2, REVEAL_START.conv2);
  connectAdjacent(conv2, conv3, REVEAL_START.conv3);

  const linkedFeatures = explainability?.fully_connected?.linked_features ?? [];
  const outputId = "output-main";
  const usedChannels = new Set();

  linkedFeatures
    .slice()
    .sort((a, b) => Math.abs(Number(b.value || 0)) - Math.abs(Number(a.value || 0)))
    .slice(0, TOPK_CONV3_TO_OUTPUT)
    .forEach((feature) => {
    const channel = Number(feature.conv3_channel);
    const node = conv3.find((item) => Number(item.channel) === channel);
    if (!node) return;

    usedChannels.add(channel);
    const signal = Number(feature.value || 0);
    const signedStrength = node.signedMean * Math.max(0.1, Math.abs(signal));

    addCandidate({
      sourceId: node.id,
      targetId: outputId,
      start: node.position,
      end: [0, 0, LAYER_DEPTH.output],
      strength: signedStrength,
      revealStart: REVEAL_START.output,
    });
  });

  if (usedChannels.size === 0) {
    conv3
      .slice()
      .sort((a, b) => Math.abs(b.signedMean) - Math.abs(a.signedMean))
      .slice(0, 8)
      .forEach((node) => {
        addCandidate({
          sourceId: node.id,
          targetId: outputId,
          start: node.position,
          end: [0, 0, LAYER_DEPTH.output],
          strength: node.signedMean,
          revealStart: REVEAL_START.output,
        });
      });
  }

  let ranked = candidates
    .sort((a, b) => b.weightMagnitude - a.weightMagnitude)
    .slice(0, MAX_CONNECTIONS);

  if (DEBUG_SINGLE_TRACE && ranked.length > 0) {
    const strongestToOutput = ranked
      .filter((conn) => conn.targetId === "output-main")
      .sort((a, b) => b.weightMagnitude - a.weightMagnitude)[0];

    if (strongestToOutput) {
      const keep = new Set([strongestToOutput.id]);
      const frontier = new Set([strongestToOutput.sourceId]);
      for (let depth = 0; depth < 4; depth += 1) {
        ranked.forEach((conn) => {
          if (frontier.has(conn.targetId)) {
            keep.add(conn.id);
          }
        });
        ranked.forEach((conn) => {
          if (keep.has(conn.id)) {
            frontier.add(conn.sourceId);
          }
        });
      }
      ranked = ranked.filter((conn) => keep.has(conn.id));
    }
  }

  return ranked;
}

function LayeredFeatureMaps({ result, pulseKey, inputImageSrc, onSelect, highlightedMap }) {
  const progressRef = useRef(1);
  const [hoveredMapId, setHoveredMapId] = useState("");
  const [hoveredPixelId, setHoveredPixelId] = useState("");
  const [hoveredOutput, setHoveredOutput] = useState(false);

  useEffect(() => {
    progressRef.current = 0;
  }, [pulseKey]);

  useFrame((_, delta) => {
    progressRef.current = Math.min(1, progressRef.current + delta / 1.7);
  });

  const pixelAnchors = useMemo(() => {
    const matrix = result?.explainability?.input_representation?.pixel_matrix;
    if (!Array.isArray(matrix) || matrix.length === 0) return [];

    const rows = matrix.length;
    const cols = matrix[0].length;
    const width = 3.2;
    const height = 3.2;

    const anchors = [];
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const u = cols > 1 ? c / (cols - 1) : 0.5;
        const v = rows > 1 ? r / (rows - 1) : 0.5;
        anchors.push({
          id: `pixel-${r}-${c}`,
          row: r,
          col: c,
          u,
          v,
          activation: Math.max(0, Math.min(1, (Number(matrix[r][c]) + 1) / 2)),
          position: [
            (u - 0.5) * width,
            (0.5 - v) * height,
            LAYER_DEPTH.input + 0.05,
          ],
        });
      }
    }
    return anchors;
  }, [result]);

  const mapNodesByLayer = useMemo(() => {
    const out = { conv1: [], conv2: [], conv3: [] };

    ["conv1", "conv2", "conv3"].forEach((layerKey) => {
      const maps = Array.isArray(result?.feature_maps?.[layerKey]) ? result.feature_maps[layerKey].slice(0, 12) : [];
      const positions = buildGridPositions(maps.length, 4, 1.25);
      out[layerKey] = maps.map((map, index) => ({
        id: `${layerKey}-${map.channel}-${index}`,
        layerKey,
        channel: map.channel,
        activation: Math.max(0, Number(map.mean_activation || 0)),
        signedMean: Number(map.signed_mean ?? map.mean_activation ?? 0),
        position: [positions[index].x, positions[index].y, LAYER_DEPTH[layerKey]],
        u: positions[index].u,
        v: positions[index].v,
        map,
      }));
    });

    return out;
  }, [result]);

  const connections = useMemo(
    () =>
      buildConnectionGraph({
        pixelAnchors,
        mapNodesByLayer,
        explainability: result?.explainability,
      }),
    [pixelAnchors, mapNodesByLayer, result]
  );

  const activeConnectionIds = useMemo(() => {
    if (!hoveredPixelId && !hoveredOutput) {
      return null;
    }

    const active = new Set();

    if (hoveredPixelId) {
      let frontier = new Set([hoveredPixelId]);
      for (let depth = 0; depth < 5; depth += 1) {
        const next = new Set();
        connections.forEach((conn) => {
          if (frontier.has(conn.sourceId)) {
            active.add(conn.id);
            next.add(conn.targetId);
          }
        });
        frontier = next;
      }
    }

    if (hoveredOutput) {
      const strongest = connections
        .filter((conn) => conn.targetId === "output-main")
        .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
        .slice(0, 10);

      const frontier = new Set(strongest.map((conn) => conn.sourceId));
      strongest.forEach((conn) => active.add(conn.id));

      for (let depth = 0; depth < 3; depth += 1) {
        const next = new Set();
        connections.forEach((conn) => {
          if (frontier.has(conn.targetId)) {
            active.add(conn.id);
            next.add(conn.sourceId);
          }
        });
        next.forEach((item) => frontier.add(item));
      }
    }

    return active;
  }, [connections, hoveredOutput, hoveredPixelId]);

  const allMapNodes = [...mapNodesByLayer.conv1, ...mapNodesByLayer.conv2, ...mapNodesByLayer.conv3];

  return (
    <group>
      {inputImageSrc && <InputPlane inputImageSrc={inputImageSrc} progressRef={progressRef} />}

      {pixelAnchors.map((pixel) => (
        <mesh
          key={pixel.id}
          position={pixel.position}
          onPointerOver={() => setHoveredPixelId(pixel.id)}
          onPointerOut={() => setHoveredPixelId("")}
          frustumCulled={false}
        >
          <circleGeometry args={[0.05, 10]} />
          <meshBasicMaterial
            color={hoveredPixelId === pixel.id ? "#fff08a" : "#a2a8b5"}
            transparent
            opacity={hoveredPixelId === pixel.id ? 0.95 : 0.25}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}

      {connections.map((connection) => (
        <ConnectionCurve
          key={connection.id}
          connection={connection}
          progressRef={progressRef}
          isActive={!activeConnectionIds || activeConnectionIds.has(connection.id)}
        />
      ))}

      {allMapNodes.map((node) => (
        <FeatureMapPlane
          key={node.id}
          map={node.map}
          mapId={node.id}
          layerKey={node.layerKey}
          position={node.position}
          progressRef={progressRef}
          hovered={hoveredMapId === node.id}
          highlighted={
            highlightedMap?.layer === node.layerKey && Number(highlightedMap?.channel) === Number(node.channel)
          }
          onHover={setHoveredMapId}
          onOut={() => setHoveredMapId("")}
          onSelect={onSelect}
        />
      ))}

      <OutputPlane
        classProbabilities={result?.class_probabilities}
        progressRef={progressRef}
        onOutputHover={() => setHoveredOutput(true)}
        onOutputOut={() => setHoveredOutput(false)}
      />
    </group>
  );
}

export default function Network3D({ result, pulseKey, inputImageSrc, highlightedMap }) {
  const hasData = Boolean(result?.activations || result?.class_probabilities);
  const [selectedMap, setSelectedMap] = useState(null);
  const [desiredDistance, setDesiredDistance] = useState(DEFAULT_CAMERA_DISTANCE);
  const controlsRef = useRef(null);
  const webglAvailable = useMemo(() => {
    try {
      const canvas = document.createElement("canvas");
      return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
    } catch (error) {
      return false;
    }
  }, []);

  if (!webglAvailable) {
    return (
      <div className="network3d-wrap network3d-fallback">
        <p>WebGL is not available in this browser session, so the 3D view is disabled.</p>
      </div>
    );
  }

  return (
    <div className="network3d-wrap network3d-feature-wrap" role="img" aria-label="3D feature map visualization">
      <Canvas
        camera={{ position: [0, 0.4, 17], fov: 68, near: 0.1, far: 1000 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        onCreated={({ camera }) => {
          camera.lookAt(NETWORK_TARGET);
          camera.updateProjectionMatrix();
        }}
      >
        <color attach="background" args={["#faf8f1"]} />
        <ambientLight intensity={0.92} />
        {DEBUG_RENDER_HELPERS && <axesHelper args={[6]} />}

        <CameraPoseController controlsRef={controlsRef} desiredDistance={desiredDistance} />

        <LayeredFeatureMaps
          result={result}
          pulseKey={pulseKey}
          inputImageSrc={inputImageSrc}
          highlightedMap={highlightedMap}
          onSelect={(title, imageUrl) => setSelectedMap({ title, imageUrl })}
        />

        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          enableZoom={false}
          minDistance={MIN_CAMERA_DISTANCE}
          maxDistance={MAX_CAMERA_DISTANCE}
          maxPolarAngle={Math.PI}
          target={[NETWORK_TARGET.x, NETWORK_TARGET.y, NETWORK_TARGET.z]}
          autoRotate={!hasData}
          autoRotateSpeed={0.45}
        />
      </Canvas>

      <div className="network-zoom-controls" aria-label="Network zoom controls">
        <button
          type="button"
          className="network-zoom-btn"
          onClick={() => setDesiredDistance((distance) => Math.max(MIN_CAMERA_DISTANCE, distance - ZOOM_STEP))}
        >
          +
        </button>
        <button
          type="button"
          className="network-zoom-btn"
          onClick={() => setDesiredDistance((distance) => Math.min(MAX_CAMERA_DISTANCE, distance + ZOOM_STEP))}
        >
          -
        </button>
      </div>

      {selectedMap && (
        <div className="network3d-modal" onClick={() => setSelectedMap(null)}>
          <div className="network3d-modal-card" onClick={(event) => event.stopPropagation()}>
            <h3>{selectedMap.title}</h3>
            <img src={selectedMap.imageUrl} alt={selectedMap.title} className="network3d-modal-image" />
            <button type="button" className="network3d-modal-close" onClick={() => setSelectedMap(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
