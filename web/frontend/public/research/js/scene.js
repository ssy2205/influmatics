/**
 * Influmatics — procedural molecular sculpture.
 * A graphic interpretation, NOT an experimental or predicted protein structure.
 * All positions are generated locally from trigonometric curves; no biological
 * coordinates, network requests, user data, or model inference are involved.
 * Three.js r180 · MIT (see assets/vendor/THREE-LICENSE.txt).
 */
import * as THREE from "three";

const TAU = Math.PI * 2;
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
const mix = (a, b, t) => a + (b - a) * t;
const smooth = (t) => t * t * (3 - 2 * t);
function createScene({
  container,
  canvas,
  onReady = () => {},
  onFallback = () => {},
} = {}) {
  if (!container || !canvas) return null;
  const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const narrowQuery = window.matchMedia("(max-width: 760px)");
  let reducedMotion = reducedQuery.matches;
  let smallScreen = narrowQuery.matches;
  let disposed = false;
  let contextLost = false;
  let paused = false;
  let inView = true;
  let mode = "structure";
  let quality = "auto";
  let raf = 0;
  let previousTime = 0;
  let age = 0;
  let cloudMix = 0;
  let mapMix = 0;
  let rnaMix = 0;
  let dnaMix = 0;
  let scrollProgress = 0;
  let scrollTarget = 0;
  let explosion = 0;
  let explosionTarget = 0;
  let targetCloud = 0;
  let targetMap = 0;
  let targetRna = 0;
  let targetDna = 0;
  let userRotationX = -0.07;
  let userRotationY = -0.36;
  let autoRotation = 0;
  let isDragging = false;
  let dragCandidate = null;
  let renderer;
  const cleanups = [];
  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  const trackGeometry = (g) => (geometries.add(g), g);
  const trackMaterial = (m) => (materials.add(m), m);

  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: !smallScreen,
      powerPreference: "default",
    });
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.25;
  } catch (error) {
    onFallback({ reason: "webgl-unavailable" });
    return null;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(37, 1, 0.1, 60);
  camera.position.set(0, 0.15, 10.6);
  camera.lookAt(0, 0.05, 0);
  const specimen = new THREE.Group();
  specimen.rotation.set(userRotationX, userRotationY, -0.16);
  scene.add(specimen);
  const environment = new THREE.Group();
  scene.add(environment);

  scene.add(new THREE.AmbientLight(0xc5e6db, 1.05));
  const key = new THREE.DirectionalLight(0xebffca, 3.9);
  key.position.set(-3, 5, 6);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xa6ff33, 4.2);
  rim.position.set(4, 0, -3);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0x6ca9c8, 2.1);
  fill.position.set(-4, -2, 2);
  scene.add(fill);
  const top = new THREE.PointLight(0xe6ffa4, 30, 15, 2);
  top.position.set(0, 4, 1);
  scene.add(top);

  // A seeded visual generator keeps the sculpture identical on every visit.
  let seed = 80231;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const perChain = smallScreen ? 204 : 286;
  const count = perChain * 3;
  const base = new Float32Array(count * 3);
  const cloud = new Float32Array(count * 3);
  const map = new Float32Array(count * 3);
  const rna = new Float32Array(count * 3);
  const dna = new Float32Array(count * 3);
  const rnaColors = new Float32Array(count * 3);
  const dnaColors = new Float32Array(count * 3);
  const rnaRadii = new Float32Array(count);
  const dnaRadii = new Float32Array(count);
  const rungCount = Math.ceil(perChain / 9);
  const positions = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const colors = new Float32Array(count * 3);
  const palette = [
    new THREE.Color("#c4ff47"),
    new THREE.Color("#b3daca"),
    new THREE.Color("#89a9ab"),
  ];
  const pointColor = new THREE.Color();
  const helixGroups = [];
  const helixMaterials = [];
  const branchDirections = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  const pale = new THREE.Color("#ecffd7");

  for (let branch = 0; branch < 3; branch++) {
    const angle = (branch * TAU) / 3 + 0.4;
    const radial = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle));
    const tangent = new THREE.Vector3(-Math.sin(angle), 0, Math.cos(angle));
    branchDirections.push(radial);
    const group = new THREE.Group();
    specimen.add(group);
    helixGroups.push(group);
    const chain = [];
    for (let j = 0; j < perChain; j++) {
      const t = j / (perChain - 1);
      const i = branch * perChain + j;
      const offset = i * 3;
      let r, y, side;
      if (t < 0.21) {
        const q = t / 0.21;
        r = 0.23 + 0.5 * smooth(q) + 0.065 * Math.sin(q * TAU * 3);
        side = 0.11 * Math.cos(q * TAU * 3 + branch);
        y = -2.25 + q * 1.64;
      } else {
        const q = (t - 0.21) / 0.79;
        const vertical = q * 2 - 1;
        const width = Math.sqrt(Math.max(0, 1 - vertical * vertical));
        const winding = q * TAU * 9 + branch * 0.7;
        r = 0.73 + Math.cos(winding) * width * (0.64 + 0.08 * Math.sin(q * 37));
        side = Math.sin(winding) * width * (0.65 + 0.08 * Math.cos(q * 31));
        y = 0.79 + 1.4 * vertical + 0.055 * Math.sin(winding * 2.8) * width;
      }
      const x = radial.x * r + tangent.x * side;
      const z = radial.z * r + tangent.z * side;
      base[offset] = x;
      base[offset + 1] = y;
      base[offset + 2] = z;
      chain.push(new THREE.Vector3(x, y, z));
      radii[i] = 0.037 + random() * 0.026 + (j % 17 === 0 ? 0.019 : 0);
      pointColor.copy(palette[branch]).lerp(pale, random() * 0.46);
      colors[offset] = pointColor.r;
      colors[offset + 1] = pointColor.g;
      colors[offset + 2] = pointColor.b;
      // A Fibonacci cloud and three graphic clusters are illustrative states.
      const sy = 1 - (i / Math.max(1, count - 1)) * 2;
      const sr = Math.sqrt(1 - sy * sy);
      const theta = golden * i;
      const radius = 1.8 + random() * 0.95;
      cloud[offset] = Math.cos(theta) * sr * radius;
      cloud[offset + 1] = sy * radius;
      cloud[offset + 2] = Math.sin(theta) * sr * radius;
      const clusterTheta = golden * j;
      const spread = Math.sqrt((j + 0.5) / perChain);
      const clusterX = [-1.47, 0.2, 1.67][branch];
      const clusterY = [0.58, -0.85, 0.71][branch];
      map[offset] = clusterX + Math.cos(clusterTheta) * spread * 0.76;
      map[offset + 1] = clusterY + Math.sin(clusterTheta) * spread * 0.6;
      map[offset + 2] = (random() - 0.5) * 0.2;

      // Comparative nucleic-acid graphics: a single RNA strand and a DNA
      // double helix. These are abstract topology, without bases or sequence.
      const rungT = (Math.floor(j / 9) + 0.5) / rungCount;
      const rungQ = (j % 9) / 8;
      const dnaT = branch < 2 ? t : rungT;
      const dnaAngle = dnaT * TAU * 2.3 + (branch === 1 ? Math.PI : 0);
      const dnaRadius = branch < 2 ? 1.05 : 1.05 * (1 - 2 * rungQ);
      dna[offset] = Math.cos(dnaAngle) * dnaRadius;
      dna[offset + 1] = -2.5 + dnaT * 5;
      dna[offset + 2] = Math.sin(dnaAngle) * dnaRadius;
      dnaRadii[i] = branch < 2 ? 0.063 : 0.035;
      pointColor.set(
        branch === 1 || (branch === 2 && rungQ > 0.5) ? "#73eaff" : "#c4ff47",
      );
      pointColor.toArray(dnaColors, offset);
      const rnaT = branch === 0 ? t : rungT;
      const rnaAngle = rnaT * TAU * 2.65;
      const rnaRadius = branch === 0 ? 0.94 : 0.94 - rungQ * 0.5;
      rna[offset] = Math.cos(rnaAngle) * rnaRadius;
      rna[offset + 1] = -2.5 + rnaT * 5;
      rna[offset + 2] = Math.sin(rnaAngle) * rnaRadius;
      rnaRadii[i] = branch === 0 ? 0.064 : branch === 1 ? 0.031 : 0.0001;
      pointColor.set(branch === 0 ? "#c4ff47" : "#e0ffc6");
      pointColor.toArray(rnaColors, offset);
    }
    const curve = new THREE.CatmullRomCurve3(chain, false, "centripetal");
    const tubeGeometry = trackGeometry(
      new THREE.TubeGeometry(
        curve,
        perChain * 2,
        0.017,
        smallScreen ? 4 : 6,
        false,
      ),
    );
    const tubeMaterial = trackMaterial(
      new THREE.MeshStandardMaterial({
        color: palette[branch],
        metalness: 0.56,
        roughness: 0.32,
        emissive: palette[branch],
        emissiveIntensity: 0.1,
        transparent: true,
        opacity: 0.7,
        depthWrite: false,
      }),
    );
    helixMaterials.push(tubeMaterial);
    group.add(new THREE.Mesh(tubeGeometry, tubeMaterial));
    // Thin, occasional bonds add molecular depth without drawing a wire cage.
    const bondPositions = [];
    for (let j = Math.floor(perChain * 0.25); j < perChain - 16; j += 3) {
      const a = chain[j],
        b = chain[j + 12];
      if (a.distanceTo(b) > 1.1) continue;
      bondPositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
    const bondGeometry = trackGeometry(new THREE.BufferGeometry());
    bondGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(bondPositions, 3),
    );
    const bondMaterial = trackMaterial(
      new THREE.LineBasicMaterial({
        color: palette[branch],
        transparent: true,
        opacity: 0.19,
        depthWrite: false,
      }),
    );
    helixMaterials.push(bondMaterial);
    group.add(new THREE.LineSegments(bondGeometry, bondMaterial));
  }

  const baseColors = new Float32Array(colors);
  const nucleicMaterials = { rna: [], dna: [] };
  const rnaGroup = new THREE.Group();
  const dnaGroup = new THREE.Group();
  specimen.add(rnaGroup, dnaGroup);
  const nucleicMaterial = (kind, color, opacity = 0.9) => {
    const material = trackMaterial(
      new THREE.MeshPhysicalMaterial({
        color,
        metalness: 0.38,
        roughness: 0.17,
        clearcoat: 1,
        clearcoatRoughness: 0.11,
        emissive: color,
        emissiveIntensity: 0.15,
        transparent: true,
        opacity: 0,
        depthWrite: false,
      }),
    );
    nucleicMaterials[kind].push({ material, opacity });
    return material;
  };
  function addNucleicRail(kind, phase, radius, turns, color) {
    const path = [];
    const steps = smallScreen ? 160 : 240;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps,
        a = t * TAU * turns + phase;
      path.push(
        new THREE.Vector3(
          Math.cos(a) * radius,
          -2.5 + t * 5,
          Math.sin(a) * radius,
        ),
      );
    }
    const geometry = trackGeometry(
      new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(path),
        steps,
        0.049,
        smallScreen ? 6 : 8,
        false,
      ),
    );
    const mesh = new THREE.Mesh(geometry, nucleicMaterial(kind, color));
    (kind === "dna" ? dnaGroup : rnaGroup).add(mesh);
  }
  addNucleicRail("dna", 0, 1.05, 2.3, "#c4ff47");
  addNucleicRail("dna", Math.PI, 1.05, 2.3, "#73eaff");
  addNucleicRail("rna", 0, 0.94, 2.65, "#c4ff47");
  const rungGeometry = trackGeometry(
    new THREE.CylinderGeometry(1, 1, 1, smallScreen ? 6 : 8),
  );
  const rungDummy = new THREE.Object3D();
  const axis = new THREE.Vector3(0, 1, 0);
  const segmentDirection = new THREE.Vector3();
  function addNucleicRungs(kind, side) {
    const mesh = new THREE.InstancedMesh(
      rungGeometry,
      nucleicMaterial(
        kind,
        kind === "dna" && side === 1 ? "#73eaff" : "#d4ff8b",
        0.82,
      ),
      rungCount,
    );
    for (let i = 0; i < rungCount; i++) {
      const t = (i + 0.5) / rungCount;
      const angle = t * TAU * (kind === "dna" ? 2.3 : 2.65);
      const radial = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle));
      const startRadius = kind === "rna" ? 0.94 : side === 0 ? 1.05 : 0;
      const endRadius = kind === "rna" ? 0.44 : side === 0 ? 0 : -1.05;
      const start = radial.clone().multiplyScalar(startRadius);
      const end = radial.clone().multiplyScalar(endRadius);
      start.y = end.y = -2.5 + t * 5;
      segmentDirection.subVectors(end, start);
      const length = segmentDirection.length();
      rungDummy.position.copy(start).add(end).multiplyScalar(0.5);
      rungDummy.quaternion.setFromUnitVectors(
        axis,
        segmentDirection.normalize(),
      );
      rungDummy.scale.set(0.026, length, 0.026);
      rungDummy.updateMatrix();
      mesh.setMatrixAt(i, rungDummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
    (kind === "dna" ? dnaGroup : rnaGroup).add(mesh);
  }
  addNucleicRungs("dna", 0);
  addNucleicRungs("dna", 1);
  addNucleicRungs("rna", 0);
  rnaGroup.visible = false;
  dnaGroup.visible = false;

  const atomGeometry = trackGeometry(
    new THREE.IcosahedronGeometry(1, smallScreen ? 0 : 1),
  );
  const atomMaterial = trackMaterial(
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.26,
      metalness: 0.42,
      emissive: 0x6f8c34,
      emissiveIntensity: 0.13,
    }),
  );
  const atoms = new THREE.InstancedMesh(atomGeometry, atomMaterial, count);
  atoms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  atoms.frustumCulled = false;
  specimen.add(atoms);
  for (let i = 0; i < count; i++)
    atoms.setColorAt(i, pointColor.fromArray(colors, i * 3));
  atoms.instanceColor.needsUpdate = true;

  // One inexpensive additive pass creates the glow, without a postprocess chain.
  const glowGeometry = trackGeometry(new THREE.BufferGeometry());
  glowGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage),
  );
  glowGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const glowMaterial = trackMaterial(
    new THREE.ShaderMaterial({
      uniforms: {
        uSize: { value: 8 },
        uAlpha: { value: 0.22 },
        uDpr: { value: 1 },
      },
      vertexShader: `attribute vec3 color; varying vec3 vColor; uniform float uSize; uniform float uDpr;
      void main(){ vColor=color; vec4 viewPosition=modelViewMatrix*vec4(position,1.0);
      gl_Position=projectionMatrix*viewPosition; gl_PointSize=clamp(uSize*uDpr*7.0/-viewPosition.z,1.0,45.0); }`,
      fragmentShader: `precision mediump float; varying vec3 vColor; uniform float uAlpha;
      void main(){float radius=length(gl_PointCoord-0.5)*2.0; if(radius>1.0) discard;
      float glow=pow(1.0-radius,3.0); gl_FragColor=vec4(vColor,glow*uAlpha);}`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }),
  );
  const glows = new THREE.Points(glowGeometry, glowMaterial);
  glows.frustumCulled = false;
  specimen.add(glows);

  const orbitMaterials = [];
  const orbitGroup = new THREE.Group();
  environment.add(orbitGroup);
  const makeRing = (
    radius,
    squash,
    rotation,
    tint,
    opacity,
    dashed = false,
  ) => {
    const coords = [];
    const n = 240;
    for (let i = 0; i <= n; i++) {
      const angle = (i / n) * TAU;
      coords.push(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius * squash,
        0,
      );
    }
    const geom = trackGeometry(new THREE.BufferGeometry());
    geom.setAttribute("position", new THREE.Float32BufferAttribute(coords, 3));
    const mat = trackMaterial(
      dashed
        ? new THREE.LineDashedMaterial({
            color: tint,
            transparent: true,
            opacity,
            dashSize: 0.025,
            gapSize: 0.07,
            depthWrite: false,
          })
        : new THREE.LineBasicMaterial({
            color: tint,
            transparent: true,
            opacity,
            depthWrite: false,
          }),
    );
    orbitMaterials.push({ material: mat, opacity });
    const ring = new THREE.Line(geom, mat);
    if (dashed) ring.computeLineDistances();
    ring.rotation.set(...rotation);
    orbitGroup.add(ring);
    return ring;
  };
  makeRing(3.05, 1, [0.99, 0.15, -0.55], 0xa8d45b, 0.36);
  makeRing(3.3, 1, [0.99, 0.15, -0.55], 0x5f8975, 0.15, true);
  makeRing(2.85, 1, [1.13, -0.7, 0.8], 0x8ebfb0, 0.18);
  makeRing(3.35, 1, [0.2, 0.8, 0.42], 0xb5ed5b, 0.1, true);

  const gridGeometry = trackGeometry(new THREE.IcosahedronGeometry(2.65, 1));
  const gridMaterial = trackMaterial(
    new THREE.MeshBasicMaterial({
      color: 0x91bd8e,
      wireframe: true,
      transparent: true,
      opacity: 0.035,
      depthWrite: false,
    }),
  );
  const globe = new THREE.Mesh(gridGeometry, gridMaterial);
  environment.add(globe);

  // Quiet fixed-depth motes help distinguish the sculpture from a flat image.
  const dustCoords = [],
    dustColors = [];
  const dustCount = smallScreen ? 110 : 210;
  for (let i = 0; i < dustCount; i++) {
    const angle = random() * TAU,
      y = random() * 2 - 1;
    const width = Math.sqrt(1 - y * y),
      radius = 3 + random() * 2.4;
    dustCoords.push(
      Math.cos(angle) * width * radius,
      y * radius,
      Math.sin(angle) * width * radius - 1.8,
    );
    pointColor.set(i % 5 ? 0x87aa89 : 0xc4ff47);
    dustColors.push(pointColor.r, pointColor.g, pointColor.b);
  }
  const dustGeometry = trackGeometry(new THREE.BufferGeometry());
  dustGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(dustCoords, 3),
  );
  dustGeometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(dustColors, 3),
  );
  const dustMaterial = trackMaterial(
    new THREE.PointsMaterial({
      size: 0.016,
      vertexColors: true,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      sizeAttenuation: true,
    }),
  );
  const dust = new THREE.Points(dustGeometry, dustMaterial);
  environment.add(dust);

  const dotGeometry = trackGeometry(new THREE.SphereGeometry(0.045, 8, 8));
  const dotMaterial = trackMaterial(
    new THREE.MeshBasicMaterial({ color: 0xd5ff83 }),
  );
  const satellite = new THREE.Mesh(dotGeometry, dotMaterial);
  orbitGroup.add(satellite);

  const dummy = new THREE.Object3D();
  function updateGeometry() {
    const structureMix = Math.max(0, 1 - cloudMix - mapMix - rnaMix - dnaMix);
    const nucleicScale = 1 + explosion * 0.09;
    for (let i = 0; i < count; i++) {
      const o = i * 3;
      const branch = Math.floor(i / perChain);
      const dir = branchDirections[branch];
      const displacement =
        explosion * Math.max(0, 1 - mapMix - rnaMix - dnaMix) * 0.83;
      const x =
        base[o] * structureMix +
        cloud[o] * cloudMix +
        map[o] * mapMix +
        (rna[o] * rnaMix + dna[o] * dnaMix) * nucleicScale +
        dir.x * displacement;
      const y =
        base[o + 1] * structureMix +
        cloud[o + 1] * cloudMix +
        map[o + 1] * mapMix +
        (rna[o + 1] * rnaMix + dna[o + 1] * dnaMix) * nucleicScale;
      const z =
        base[o + 2] * structureMix +
        cloud[o + 2] * cloudMix +
        map[o + 2] * mapMix +
        (rna[o + 2] * rnaMix + dna[o + 2] * dnaMix) * nucleicScale +
        dir.z * displacement;
      positions[o] = x;
      positions[o + 1] = y;
      positions[o + 2] = z;
      dummy.position.set(x, y, z);
      dummy.scale.setScalar(
        radii[i] * (structureMix + cloudMix * 0.69 + mapMix * 0.8) +
          rnaRadii[i] * rnaMix +
          dnaRadii[i] * dnaMix,
      );
      dummy.updateMatrix();
      atoms.setMatrixAt(i, dummy.matrix);
      for (let component = 0; component < 3; component++) {
        colors[o + component] =
          baseColors[o + component] * (1 - rnaMix - dnaMix) +
          rnaColors[o + component] * rnaMix +
          dnaColors[o + component] * dnaMix;
      }
      atoms.setColorAt(i, pointColor.fromArray(colors, o));
    }
    atoms.instanceMatrix.needsUpdate = true;
    atoms.instanceColor.needsUpdate = true;
    glowGeometry.attributes.position.needsUpdate = true;
    glowGeometry.attributes.color.needsUpdate = true;
    const structureOpacity = Math.pow(structureMix, 2);
    helixMaterials.forEach((mat, i) => {
      mat.opacity = (i % 2 === 0 ? 0.7 : 0.19) * structureOpacity;
    });
    helixGroups.forEach((group, i) => {
      const d = branchDirections[i];
      group.position.set(d.x * explosion * 0.83, 0, d.z * explosion * 0.83);
      group.visible = structureOpacity > 0.001;
    });
    rnaGroup.visible = rnaMix > 0.002;
    dnaGroup.visible = dnaMix > 0.002;
    rnaGroup.scale.setScalar(nucleicScale);
    dnaGroup.scale.setScalar(nucleicScale);
    nucleicMaterials.rna.forEach(({ material, opacity }) => {
      material.opacity = opacity * Math.pow(rnaMix, 1.5);
    });
    nucleicMaterials.dna.forEach(({ material, opacity }) => {
      material.opacity = opacity * Math.pow(dnaMix, 1.5);
    });
    glowMaterial.uniforms.uSize.value =
      8 + cloudMix * 5 + mapMix * 2 + (rnaMix + dnaMix) * 2;
    glowMaterial.uniforms.uAlpha.value =
      0.28 + cloudMix * 0.38 + (rnaMix + dnaMix) * 0.1;
  }
  updateGeometry();

  function fit() {
    if (disposed) return;
    const box = container.getBoundingClientRect();
    if (box.width < 1 || box.height < 1) return;
    smallScreen = narrowQuery.matches;
    const cap =
      quality === "low"
        ? 1
        : quality === "high"
          ? smallScreen
            ? 1.2
            : 1.75
          : smallScreen
            ? 1.2
            : 1.5;
    const dpr = Math.min(window.devicePixelRatio || 1, cap);
    renderer.setPixelRatio(dpr);
    renderer.setSize(Math.round(box.width), Math.round(box.height), false);
    glowMaterial.uniforms.uDpr.value = dpr;
    camera.aspect = box.width / box.height;
    // On tall/narrow containers, horizontal fitting avoids clipping the orbit.
    const verticalFov = (camera.fov * Math.PI) / 180;
    const limitingFov = Math.min(
      verticalFov,
      2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect),
    );
    camera.position.z = 3.75 / Math.tan(limitingFov / 2);
    camera.updateProjectionMatrix();
    requestRender();
  }
  function canAnimate() {
    return (
      !disposed &&
      !contextLost &&
      !paused &&
      !reducedMotion &&
      inView &&
      !document.hidden
    );
  }
  function stopLoop() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    previousTime = 0;
  }
  function draw() {
    if (disposed || contextLost) return;
    try {
      renderer.render(scene, camera);
    } catch (error) {
      stopLoop();
      contextLost = true;
      onFallback({ reason: "render-error" });
    }
  }
  function settle() {
    cloudMix = targetCloud;
    mapMix = targetMap;
    rnaMix = targetRna;
    dnaMix = targetDna;
    scrollProgress = scrollTarget;
    explosion = explosionTarget;
    updateGeometry();
    specimen.rotation.set(
      mode === "map"
        ? 0
        : userRotationX + Math.sin(scrollProgress * Math.PI) * 0.22,
      mode === "map"
        ? 0
        : userRotationY + autoRotation + scrollProgress * Math.PI * 1.6,
      mode === "map" ? 0 : -0.16 + Math.sin(scrollProgress * TAU) * 0.09,
    );
    draw();
  }
  function frame(time) {
    raf = 0;
    if (!canAnimate()) return;
    const interval =
      quality === "low" || (smallScreen && quality !== "high")
        ? 1000 / 30
        : 1000 / 60;
    if (previousTime && time - previousTime < interval - 1) {
      raf = requestAnimationFrame(frame);
      return;
    }
    const delta = previousTime
      ? Math.min((time - previousTime) / 1000, 0.05)
      : 1 / 60;
    previousTime = time;
    age += delta;
    const lerp = 1 - Math.exp(-delta * 5);
    const needsGeometry =
      Math.abs(targetCloud - cloudMix) +
        Math.abs(targetMap - mapMix) +
        Math.abs(targetRna - rnaMix) +
        Math.abs(targetDna - dnaMix) +
        Math.abs(explosionTarget - explosion) >
      0.0002;
    if (needsGeometry) {
      cloudMix = mix(cloudMix, targetCloud, lerp);
      mapMix = mix(mapMix, targetMap, lerp);
      rnaMix = mix(rnaMix, targetRna, lerp);
      dnaMix = mix(dnaMix, targetDna, lerp);
      explosion = mix(explosion, explosionTarget, lerp);
      updateGeometry();
    }
    scrollProgress = mix(
      scrollProgress,
      scrollTarget,
      1 - Math.exp(-delta * 8),
    );
    if (!isDragging && mode !== "map") autoRotation += delta * 0.085;
    const rotationMix = 1 - Math.exp(-delta * 7);
    specimen.rotation.x = mix(
      specimen.rotation.x,
      mode === "map"
        ? 0
        : userRotationX + Math.sin(scrollProgress * Math.PI) * 0.22,
      rotationMix,
    );
    specimen.rotation.y = mix(
      specimen.rotation.y,
      mode === "map"
        ? 0
        : userRotationY + autoRotation + scrollProgress * Math.PI * 1.6,
      rotationMix,
    );
    specimen.rotation.z = mix(
      specimen.rotation.z,
      mode === "map" ? 0 : -0.16 + Math.sin(scrollProgress * TAU) * 0.09,
      rotationMix,
    );
    specimen.position.y = mode === "map" ? 0 : Math.sin(age * 0.65) * 0.055;
    orbitGroup.rotation.y = Math.sin(age * 0.09) * 0.13;
    globe.rotation.y = age * 0.025;
    dust.rotation.y = age * 0.012;
    const theta = age * 0.21;
    satellite.position.set(
      Math.cos(theta) * 3.05,
      Math.sin(theta) * 1.65,
      Math.sin(theta) * 2.5,
    );
    orbitMaterials.forEach(({ material, opacity }) => {
      material.opacity = opacity * (1 - mapMix * 0.7);
    });
    draw();
    if (canAnimate()) raf = requestAnimationFrame(frame);
  }
  function requestRender() {
    if (disposed || contextLost || document.hidden || !inView) return;
    if (canAnimate()) {
      if (!raf) raf = requestAnimationFrame(frame);
    } else settle();
  }
  const listen = (target, type, handler, options) => {
    target.addEventListener(type, handler, options);
    cleanups.push(() => target.removeEventListener(type, handler, options));
  };
  const pointerDown = (event) => {
    if (mode === "map") return;
    if (event.button !== undefined && event.button !== 0) return;
    dragCandidate = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      lastX: event.clientX,
      lastY: event.clientY,
      type: event.pointerType,
    };
    if (event.pointerType === "mouse") {
      isDragging = true;
      canvas.setPointerCapture?.(event.pointerId);
      canvas.style.cursor = "grabbing";
    }
  };
  const pointerMove = (event) => {
    if (!dragCandidate || event.pointerId !== dragCandidate.id) return;
    const totalX = event.clientX - dragCandidate.x;
    const totalY = event.clientY - dragCandidate.y;
    if (!isDragging && dragCandidate.type !== "mouse") {
      if (Math.abs(totalY) > Math.abs(totalX) && Math.abs(totalY) > 8) {
        dragCandidate = null;
        return;
      }
      if (Math.abs(totalX) > 8 && Math.abs(totalX) > Math.abs(totalY)) {
        isDragging = true;
        canvas.setPointerCapture?.(event.pointerId);
      }
    }
    if (!isDragging) return;
    userRotationY += (event.clientX - dragCandidate.lastX) * 0.007;
    userRotationX = clamp(
      userRotationX + (event.clientY - dragCandidate.lastY) * 0.004,
      -0.75,
      0.75,
    );
    dragCandidate.lastX = event.clientX;
    dragCandidate.lastY = event.clientY;
    if (mode === "map") return;
    requestRender();
  };
  const pointerUp = () => {
    const previous = dragCandidate;
    dragCandidate = null;
    isDragging = false;
    if (previous && canvas.hasPointerCapture?.(previous.id))
      canvas.releasePointerCapture(previous.id);
    canvas.style.cursor = mode === "map" ? "default" : "grab";
  };
  canvas.style.touchAction = "pan-y";
  canvas.style.cursor = "grab";
  listen(canvas, "pointerdown", pointerDown);
  listen(canvas, "pointermove", pointerMove);
  listen(canvas, "pointerup", pointerUp);
  listen(canvas, "pointercancel", pointerUp);
  listen(canvas, "lostpointercapture", pointerUp);
  listen(canvas, "webglcontextlost", (event) => {
    event.preventDefault();
    contextLost = true;
    stopLoop();
    onFallback({ reason: "context-lost" });
  });
  listen(canvas, "webglcontextrestored", () => {
    contextLost = false;
    fit();
    draw();
    if (disposed || contextLost) return;
    onReady({ renderer: "webgl2", restored: true });
    requestRender();
  });
  listen(document, "visibilitychange", () => {
    if (document.hidden) stopLoop();
    else requestRender();
  });
  const motionChange = (event) => {
    reducedMotion = event.matches;
    stopLoop();
    requestRender();
  };
  listen(reducedQuery, "change", motionChange);
  listen(narrowQuery, "change", fit);
  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver = new ResizeObserver(fit);
    resizeObserver.observe(container);
    cleanups.push(() => resizeObserver.disconnect());
  } else listen(window, "resize", fit, { passive: true });
  if (typeof IntersectionObserver !== "undefined") {
    const observer = new IntersectionObserver(
      (entries) => {
        inView = entries[0].isIntersecting;
        if (inView) requestRender();
        else stopLoop();
      },
      { rootMargin: "100px", threshold: 0 },
    );
    observer.observe(container);
    cleanups.push(() => observer.disconnect());
  }
  function dispose() {
    if (disposed) return;
    disposed = true;
    stopLoop();
    cleanups.forEach((fn) => fn());
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => m.dispose());
    textures.forEach((t) => t.dispose());
    renderer.dispose();
  }
  fit();
  draw();
  // A first-frame failure must release the instance before init reports fallback.
  // The global wrapper queues initialization callbacks until current is assigned.
  if (contextLost) {
    dispose();
    return null;
  }
  onReady({ renderer: "webgl2", conceptual: true });
  requestRender();

  return {
    setMode(value) {
      if (!["structure", "rna", "dna", "points", "map"].includes(value)) return;
      mode = value;
      targetCloud = value === "points" ? 1 : 0;
      targetMap = value === "map" ? 1 : 0;
      targetRna = value === "rna" ? 1 : 0;
      targetDna = value === "dna" ? 1 : 0;
      canvas.style.cursor = mode === "map" ? "default" : "grab";
      requestRender();
    },
    setScroll(value) {
      scrollTarget = clamp(Number(value) || 0, 0, 1);
      requestRender();
    },
    setExplode(value) {
      explosionTarget = clamp(Number(value) || 0, 0, 1);
      requestRender();
    },
    setPaused(value) {
      paused = Boolean(value);
      stopLoop();
      requestRender();
    },
    setQuality(value) {
      quality = ["high", "low"].includes(value) ? value : "auto";
      fit();
    },
    reset() {
      userRotationX = -0.07;
      userRotationY = -0.36;
      autoRotation = 0;
      explosionTarget = 0;
      mode = "structure";
      targetCloud = 0;
      targetMap = 0;
      targetRna = 0;
      targetDna = 0;
      scrollProgress = 0;
      scrollTarget = 0;
      canvas.style.cursor = "grab";
      requestRender();
    },
    destroy() {
      dispose();
    },
  };
}

/** Each controller owns its own scene, callbacks, observers, and renderer. */
function createController() {
  let current = null;
  return {
    create: createController,
    init(options = {}) {
      if (current) current.destroy();
      current = null;
      let initializing = true;
      const notifications = [];
      const notify = (name, detail) => {
        const callback = options[name];
        if (typeof callback !== "function") return;
        if (initializing) notifications.push(() => callback(detail));
        else callback(detail);
      };
      current = createScene({
        ...options,
        onReady: (detail) => notify("onReady", detail),
        onFallback: (detail) => notify("onFallback", detail),
      });
      const initialized = Boolean(current);
      initializing = false;
      // Callers may pause/reset/destroy this assigned instance synchronously.
      notifications.forEach((deliver) => deliver());
      return initialized;
    },
    setMode(value) {
      current?.setMode(value);
    },
    setScroll(value) {
      current?.setScroll(value);
    },
    setExplode(value) {
      current?.setExplode(value);
    },
    setPaused(value) {
      current?.setPaused(value);
    },
    setQuality(value) {
      current?.setQuality(value);
    },
    reset() {
      current?.reset();
    },
    destroy() {
      current?.destroy();
      current = null;
    },
  };
}
window.InflumaticsScene = createController();
