import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const DEBUG_RENDER_HELPERS = false;

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
    return [x, y];
  });
}

function FeatureMapPlane({ map, layerKey, position, progressRef, hovered, highlighted, onHover, onOut, onSelect }) {
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
      onPointerOver={onHover}
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

function OutputPlane({ classProbabilities, progressRef }) {
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
    <mesh position={[0, 0, LAYER_DEPTH.output]} frustumCulled={false}>
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

function LayeredFeatureMaps({ result, pulseKey, inputImageSrc, onSelect, highlightedMap }) {
  const progressRef = useRef(1);
  const [hoveredId, setHoveredId] = useState("");

  useEffect(() => {
    progressRef.current = 0;
  }, [pulseKey]);

  useFrame((_, delta) => {
    progressRef.current = Math.min(1, progressRef.current + delta / 1.6);
  });

  const layers = useMemo(() => {
    const conv1 = Array.isArray(result?.feature_maps?.conv1) ? result.feature_maps.conv1.slice(0, 12) : [];
    const conv2 = Array.isArray(result?.feature_maps?.conv2) ? result.feature_maps.conv2.slice(0, 12) : [];
    const conv3 = Array.isArray(result?.feature_maps?.conv3) ? result.feature_maps.conv3.slice(0, 12) : [];
    return [
      { key: "conv1", z: LAYER_DEPTH.conv1, maps: conv1 },
      { key: "conv2", z: LAYER_DEPTH.conv2, maps: conv2 },
      { key: "conv3", z: LAYER_DEPTH.conv3, maps: conv3 },
    ];
  }, [result]);

  return (
    <group>
      {inputImageSrc && <InputPlane inputImageSrc={inputImageSrc} progressRef={progressRef} />}

      {layers.map((layer) => {
        const positions = buildGridPositions(layer.maps.length, 4, 1.25);
        return layer.maps.map((map, index) => {
          const id = `${layer.key}-${map.channel}-${index}`;
          return (
            <FeatureMapPlane
              key={id}
              map={map}
              layerKey={layer.key}
              position={[positions[index][0], positions[index][1], layer.z]}
              progressRef={progressRef}
              hovered={hoveredId === id}
              highlighted={
                highlightedMap?.layer === layer.key &&
                Number(highlightedMap?.channel) === Number(map.channel)
              }
              onHover={() => setHoveredId(id)}
              onOut={() => setHoveredId("")}
              onSelect={onSelect}
            />
          );
        });
      })}

      <OutputPlane classProbabilities={result?.class_probabilities} progressRef={progressRef} />
    </group>
  );
}

export default function Network3D({ result, pulseKey, inputImageSrc, highlightedMap }) {
  const hasData = Boolean(result?.activations || result?.class_probabilities);
  const [selectedMap, setSelectedMap] = useState(null);
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
          camera.lookAt(0, 0, 0);
          camera.updateProjectionMatrix();
        }}
      >
        <color attach="background" args={["#faf8f1"]} />
        <ambientLight intensity={0.92} />
        {DEBUG_RENDER_HELPERS && <axesHelper args={[6]} />}
        <LayeredFeatureMaps
          result={result}
          pulseKey={pulseKey}
          inputImageSrc={inputImageSrc}
          highlightedMap={highlightedMap}
          onSelect={(title, imageUrl) => setSelectedMap({ title, imageUrl })}
        />
        <OrbitControls
          enablePan={false}
          minDistance={12}
          maxDistance={24}
          maxPolarAngle={Math.PI}
          autoRotate={!hasData}
          autoRotateSpeed={0.45}
        />
      </Canvas>

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
