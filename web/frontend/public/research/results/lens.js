'use strict';

// This illustration combines an original optical study with three curved depth
// fields. The mesh projects the layers at different depths as the viewer moves.
(() => {
  const canvas = document.getElementById('lens');
  const surface = canvas.parentElement;
  const figure = surface.parentElement;
  const button = document.getElementById('motion');
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const pointer = { x: 0, y: 0 };
  const smoothed = { x: 0, y: 0 };
  let enabled = !preference.matches;
  let visible = true;
  let frame = 0;
  let previous = 0;
  let elapsed = 0;
  let renderScene = null;

  const vertexSource = `
    precision mediump float;
    attribute vec2 uv;
    uniform vec2 tilt;
    uniform float time;
    varying vec2 texCoord;
    varying float relief;
    float dome(vec2 point, vec2 centre, vec2 radius, float slope) {
      vec2 d = point - centre;
      d.y += slope * d.x;
      float k = length(d / radius);
      return 1.0 - smoothstep(0.65, 1.22, k);
    }
    void main() {
      float upper = dome(uv, vec2(0.696,0.182), vec2(0.245,0.067), 0.34);
      float middle = dome(uv, vec2(0.569,0.412), vec2(0.36,0.095), 0.34);
      float lower = dome(uv, vec2(0.519,0.700), vec2(0.47,0.140), 0.35);
      relief = upper * 0.95 + middle * 0.64 + lower * 0.32;
      vec2 xy = vec2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
      float drift = sin(time * 0.34);
      xy += tilt * relief * 0.034;
      xy.x += drift * (upper - lower * 0.21) * 0.014;
      xy.y += sin(time * 0.45) * (upper + middle * 0.5) * 0.018;
      float perspective = 1.0 - dot(xy, tilt) * 0.014 - relief * 0.012;
      gl_Position = vec4(xy, 0.0, perspective);
      texCoord = uv;
    }
  `;
  const fragmentSource = `
    precision mediump float;
    uniform sampler2D photograph;
    uniform float time;
    varying vec2 texCoord;
    varying float relief;
    void main() {
      vec3 color = texture2D(photograph, texCoord).rgb;
      float brightness = max(max(color.r, color.g), color.b);
      float sweep = pow(max(0.0, cos(8.0 * texCoord.x + 4.0 * texCoord.y - 0.47 * time)), 12.0);
      color += vec3(0.075,0.12,0.11) * relief * sweep * smoothstep(0.18,0.84,brightness);
      gl_FragColor = vec4(color, 1.0);
    }
  `;

  function createScene(image) {
    const gl = canvas.getContext('webgl', { alpha: false, antialias: false, depth: false, powerPreference: 'low-power' });
    if (!gl) return null;

    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return null;
      gl.attachShader(program, shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
    gl.useProgram(program);

    const cells = 56;
    const vertices = [];
    const triangles = [];
    for (let y = 0; y <= cells; y++) {
      for (let x = 0; x <= cells; x++) {
        vertices.push(x / cells, y / cells);
        if (x === cells || y === cells) continue;
        const first = y * (cells + 1) + x;
        const next = first + cells + 1;
        triangles.push(first, next, first + 1, first + 1, next, next + 1);
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    const uv = gl.getAttribLocation(program, 'uv');
    gl.enableVertexAttribArray(uv);
    gl.vertexAttribPointer(uv, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(triangles), gl.STATIC_DRAW);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
    const time = gl.getUniformLocation(program, 'time');
    const tilt = gl.getUniformLocation(program, 'tilt');
    return (seconds, x, y) => {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = Math.max(1, Math.round(surface.clientWidth * ratio));
      const height = Math.max(1, Math.round(surface.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }
      gl.uniform1f(time, seconds);
      gl.uniform2f(tilt, x, y);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawElements(gl.TRIANGLES, triangles.length, gl.UNSIGNED_SHORT, 0);
    };
  }

  // Browsers without WebGL retain depth and motion using a perspective transform.
  function renderPoster(seconds, x, y) {
    const drift = Math.sin(seconds * 0.36);
    const rise = Math.sin(seconds * 0.47);
    surface.style.transform = `perspective(820px) rotateX(${rise + y * 2.4}deg) rotateY(${drift * 1.8 + x * 3.4}deg) translateY(${rise * 3}px)`;
  }

  function paint(timestamp) {
    frame = 0;
    const delta = Math.min(timestamp - previous, 80);
    if (delta > 30) {
      previous = timestamp;
      elapsed += delta / 1000;
      smoothed.x += (pointer.x - smoothed.x) * 0.1;
      smoothed.y += (pointer.y - smoothed.y) * 0.1;
      renderScene(elapsed, smoothed.x, smoothed.y);
    }
    if (enabled && visible && !document.hidden) frame = requestAnimationFrame(paint);
  }

  function refresh() {
    cancelAnimationFrame(frame);
    if (!renderScene) return;
    previous = performance.now();
    button.textContent = enabled ? '움직임 끄기' : '움직임 켜기';
    button.setAttribute('aria-pressed', String(enabled));
    if (visible && !document.hidden) {
      renderScene(elapsed, enabled ? smoothed.x : 0, enabled ? smoothed.y : 0);
      if (enabled) frame = requestAnimationFrame(paint);
    }
  }

  button.addEventListener('click', () => { enabled = !enabled; refresh(); });
  preference.addEventListener('change', () => { enabled = !preference.matches; refresh(); });
  figure.addEventListener('pointermove', event => {
    if (event.pointerType === 'touch') return;
    const rect = figure.getBoundingClientRect();
    pointer.x = (event.clientX - rect.left) / rect.width * 2 - 1;
    pointer.y = 1 - (event.clientY - rect.top) / rect.height * 2;
  });
  figure.addEventListener('pointerleave', () => { pointer.x = 0; pointer.y = 0; });
  document.addEventListener('visibilitychange', refresh);
  window.addEventListener('resize', refresh, { passive: true });
  canvas.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    cancelAnimationFrame(frame);
    renderScene = renderPoster;
    canvas.hidden = true;
    refresh();
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => { visible = entries[0].isIntersecting; refresh(); }).observe(figure);
  }
  function startScene(renderer) {
    canvas.hidden = !renderer;
    renderScene = renderer || renderPoster;
    button.hidden = false;
    refresh();
  }

  const image = new Image();
  image.onload = () => {
    let renderer;
    try {
      renderer = createScene(image);
    } catch {
      // Some restricted browsers expose WebGL but refuse texture uploads.
    }
    startScene(renderer);
  };
  image.onerror = () => {
    surface.classList.add('without-artwork');
    for (let layer = 0; layer < 3; layer++) {
      surface.appendChild(document.createElement('i'));
    }
    startScene(null);
  };
  image.src = './optical-study.webp';
})();
