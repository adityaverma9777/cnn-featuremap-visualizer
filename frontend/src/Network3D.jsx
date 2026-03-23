import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

const LAYER_LAYOUT = [
  { key: "input", label: "Input", x: -8, count: 24 },
  { key: "conv1", label: "Conv1", x: -4, count: 24 },
  { key: "conv2", label: "Conv2", x: 0, count: 24 },
  { key: "conv3", label: "Conv3", x: 4, count: 24 },
  { key: "output", label: "Output", x: 8, count: 10 },
];

function normalizeValues(values, count) {
  const source = Array.isArray(values) ? values.map((value) => Number(value || 0)) : [];
  const fallback = source.length > 0 ? source : new Array(count).fill(0);

  const sampled = Array.from({ length: count }, (_, index) => fallback[index % fallback.length]);
  const min = Math.min(...sampled);
  const max = Math.max(...sampled);
  const span = max - min || 1;

  return sampled.map((value) => (value - min) / span);
}

function buildLayerSignals(result) {
  const conv1 = result?.activations?.conv1 ?? [];
  const conv2 = result?.activations?.conv2 ?? [];
  const conv3 = result?.activations?.conv3 ?? [];
  const output = Array.isArray(result?.class_probabilities)
    ? result.class_probabilities.map((item) => Number(item.probability || 0))
    : [];

  const inputSeed = conv1.length > 0 ? conv1 : output.length > 0 ? output : [0.25, 0.4, 0.6, 0.35];

  return {
    input: normalizeValues(inputSeed, 24),
    conv1: normalizeValues(conv1, 24),
    conv2: normalizeValues(conv2, 24),
    conv3: normalizeValues(conv3, 24),
    output: normalizeValues(output.length > 0 ? output : [0.1], 10),
  };
}

function buildNodesAndEdges(layerSignals) {
  const nodes = [];
  const edges = [];

  LAYER_LAYOUT.forEach((layer, layerIndex) => {
    const values = layerSignals[layer.key];
    const rows = Math.ceil(Math.sqrt(layer.count));
    const cols = Math.ceil(layer.count / rows);

    for (let i = 0; i < layer.count; i += 1) {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const y = (row - (rows - 1) / 2) * 0.8;
      const z = (col - (cols - 1) / 2) * 0.8;

      nodes.push({
        id: `${layer.key}-${i}`,
        layerIndex,
        layerKey: layer.key,
        value: values[i],
        position: new THREE.Vector3(layer.x, y, z),
      });
    }
  });

  const nodesByLayer = LAYER_LAYOUT.map((layer, index) =>
    nodes.filter((node) => node.layerIndex === index && node.layerKey === layer.key)
  );

  for (let i = 0; i < nodesByLayer.length - 1; i += 1) {
    const left = nodesByLayer[i];
    const right = nodesByLayer[i + 1];
    const degree = Math.min(4, right.length);

    left.forEach((node, nodeIndex) => {
      for (let k = 0; k < degree; k += 1) {
        const target = right[(nodeIndex + k * 3) % right.length];
        const influence = 0.2 + 0.8 * ((node.value + target.value) / 2);
        edges.push({ from: node.position, to: target.position, layerIndex: i, influence });
      }
    });
  }

  return { nodes, edges };
}

function Nodes({ nodes, pulseKey }) {
  const meshRef = useRef(null);
  const color = new THREE.Color();
  const matrix = new THREE.Matrix4();
  const baseScale = new THREE.Vector3();
  const quat = new THREE.Quaternion();

  useFrame(({ clock }) => {
    if (!meshRef.current) return;

    const time = clock.getElapsedTime();
    nodes.forEach((node, index) => {
      const wave = 0.75 + 0.25 * Math.sin(time * 3.2 - node.layerIndex * 0.9 + pulseKey * 0.7);
      const strength = node.value * wave;
      const radius = 0.12 + strength * 0.27;
      baseScale.set(radius, radius, radius);
      matrix.compose(node.position, quat, baseScale);
      meshRef.current.setMatrixAt(index, matrix);

      color.setHSL(0.62 - strength * 0.55, 0.85, 0.52);
      meshRef.current.setColorAt(index, color);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, nodes.length]}>
      <sphereGeometry args={[1, 14, 14]} />
      <meshStandardMaterial vertexColors roughness={0.35} metalness={0.05} />
    </instancedMesh>
  );
}

function Edges({ edges, pulseKey }) {
  const meshRef = useRef(null);
  const tmp = new THREE.Vector3();
  const axis = new THREE.Vector3(0, 1, 0);
  const quat = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();

  useFrame(({ clock }) => {
    if (!meshRef.current) return;

    const time = clock.getElapsedTime();
    edges.forEach((edge, index) => {
      tmp.subVectors(edge.to, edge.from);
      const length = tmp.length();
      const direction = tmp.clone().normalize();

      pos.copy(edge.from).add(edge.to).multiplyScalar(0.5);
      quat.setFromUnitVectors(axis, direction);

      const pulse = 0.8 + 0.2 * Math.sin(time * 4 - edge.layerIndex * 1.15 + pulseKey * 0.8);
      const thickness = 0.012 + edge.influence * 0.03 * pulse;
      scale.set(thickness, length * 0.5, thickness);
      matrix.compose(pos, quat, scale);
      meshRef.current.setMatrixAt(index, matrix);

      color.setHSL(0.62 - edge.influence * 0.5, 0.7, 0.42 + edge.influence * 0.2);
      meshRef.current.setColorAt(index, color);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, edges.length]}>
      <cylinderGeometry args={[1, 1, 1, 8]} />
      <meshStandardMaterial vertexColors transparent opacity={0.8} roughness={0.7} />
    </instancedMesh>
  );
}

function LayeredNetwork({ result, pulseKey }) {
  const { nodes, edges } = useMemo(() => {
    const signals = buildLayerSignals(result);
    return buildNodesAndEdges(signals);
  }, [result]);

  return (
    <group>
      <Edges edges={edges} pulseKey={pulseKey} />
      <Nodes nodes={nodes} pulseKey={pulseKey} />
    </group>
  );
}

export default function Network3D({ result, pulseKey }) {
  const hasData = Boolean(result?.activations || result?.class_probabilities);
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
    <div className="network3d-wrap" role="img" aria-label="3D neural network visualization">
      <Canvas
        camera={{ position: [0, 2, 18], fov: 42 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
      >
        <color attach="background" args={["#faf8f1"]} />
        <ambientLight intensity={0.7} />
        <directionalLight intensity={0.9} position={[8, 10, 6]} />
        <pointLight intensity={0.55} position={[-9, -6, -2]} />
        <LayeredNetwork result={result} pulseKey={pulseKey} />
        <OrbitControls
          enablePan={false}
          maxPolarAngle={Math.PI * 0.85}
          minDistance={11}
          maxDistance={27}
          autoRotate={!hasData}
          autoRotateSpeed={0.8}
        />
      </Canvas>
    </div>
  );
}
