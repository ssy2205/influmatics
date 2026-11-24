/**
 * Influmatics — GPU-independent, interactive 3D projection renderer.
 * The geometry is decorative, deterministic, and not a measured protein model.
 * This Canvas2D path preserves the interactive experience when WebGL is blocked.
 */
(function () {
  "use strict";
  var TAU = Math.PI * 2;
  var clamp = function (n, a, b) {
    return Math.max(a, Math.min(b, n));
  };
  var mix = function (a, b, t) {
    return a + (b - a) * t;
  };
  var smooth = function (t) {
    return t * t * (3 - 2 * t);
  };

  function createScene(options) {
    options = options || {};
    var container = options.container;
    var onReady =
      typeof options.onReady === "function" ? options.onReady : function () {};
    var onFallback =
      typeof options.onFallback === "function"
        ? options.onFallback
        : function () {};
    if (!container) return null;
    var canvas = document.createElement("canvas");
    canvas.className = "fallback-canvas";
    canvas.setAttribute("aria-hidden", "true");
    canvas.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;z-index:1;cursor:grab;touch-action:pan-y;";
    var context;
    try {
      context = canvas.getContext("2d", { alpha: true });
    } catch (error) {
      context = null;
    }
    if (!context) {
      onFallback({ reason: "canvas2d-unavailable" });
      return null;
    }
    container.appendChild(canvas);

    var narrow = window.matchMedia("(max-width:760px)");
    var motionQuery = window.matchMedia("(prefers-reduced-motion:reduce)");
    var mobile = narrow.matches;
    var reduced = motionQuery.matches;
    var destroyed = false;
    var failed = false;
    var paused = false;
    var inView = true;
    var mode = "structure";
    var quality = "auto";
    var width = 1,
      height = 1,
      pixelRatio = 1,
      scale = 1;
    var cloudMix = 0,
      mapMix = 0,
      dnaMix = 0,
      rnaMix = 0,
      cloudTarget = 0,
      mapTarget = 0,
      dnaTarget = 0,
      rnaTarget = 0;
    var explosion = 0,
      explosionTarget = 0;
    var rotationX = -0.07,
      rotationY = -0.36,
      renderedX = -0.07,
      renderedY = -0.36,
      renderedZ = -0.16;
    var autoRotation = 0,
      scrollProgress = 0,
      scrollTarget = 0,
      age = 0,
      raf = 0,
      previousTime = 0;
    var drag = null,
      dragging = false;
    var cleanups = [];
    var perChain = mobile ? 120 : 174;
    var nodes = [],
      dust = [];
    var colors = ["#c4ff47", "#76dce8", "#b0c4ca"];
    var rgb = ["196,255,71", "118,220,232", "176,196,202"];
    var bright = ["#f0ffce", "#e7fbff", "#edf8ff"];
    var directions = [];
    var seed = 80231;
    function random() {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    }

    // Pre-render luminous spheres once; each frame only places these tiny images.
    var sprites = colors.map(function (color, branch) {
      var image = document.createElement("canvas");
      image.width = 64;
      image.height = 64;
      var c = image.getContext("2d");
      var halo = c.createRadialGradient(32, 32, 2, 32, 32, 32);
      halo.addColorStop(0, "rgba(" + rgb[branch] + ",0.65)");
      halo.addColorStop(0.28, "rgba(" + rgb[branch] + ",0.16)");
      halo.addColorStop(1, "rgba(" + rgb[branch] + ",0)");
      c.fillStyle = halo;
      c.fillRect(0, 0, 64, 64);
      var sphere = c.createRadialGradient(28, 28, 1, 32, 32, 11);
      sphere.addColorStop(0, bright[branch]);
      sphere.addColorStop(0.4, color);
      sphere.addColorStop(1, ["#4a6f21", "#354f44", "#294046"][branch]);
      c.fillStyle = sphere;
      c.beginPath();
      c.arc(32, 32, 11, 0, TAU);
      c.fill();
      return image;
    });

    for (var branch = 0; branch < 3; branch++) {
      var angle = (branch * TAU) / 3 + 0.4;
      var dx = Math.cos(angle),
        dz = Math.sin(angle);
      directions.push({ x: dx, z: dz });
      for (var j = 0; j < perChain; j++) {
        var t = j / (perChain - 1),
          r,
          y,
          side;
        if (t < 0.21) {
          var q = t / 0.21;
          r = 0.23 + 0.5 * smooth(q) + 0.065 * Math.sin(q * TAU * 3);
          side = 0.11 * Math.cos(q * TAU * 3 + branch);
          y = -2.25 + q * 1.64;
        } else {
          q = (t - 0.21) / 0.79;
          var vertical = q * 2 - 1;
          var span = Math.sqrt(Math.max(0, 1 - vertical * vertical));
          var winding = q * TAU * 9 + branch * 0.7;
          r =
            0.73 + Math.cos(winding) * span * (0.64 + 0.08 * Math.sin(q * 37));
          side = Math.sin(winding) * span * (0.65 + 0.08 * Math.cos(q * 31));
          y = 0.79 + 1.4 * vertical + 0.055 * Math.sin(winding * 2.8) * span;
        }
        var index = branch * perChain + j;
        var halfCount = (perChain * 3) / 2;
        var strand = index < halfCount ? 0 : 1;
        var helixT = (index % halfCount) / (halfCount - 1);
        var dnaAngle = helixT * TAU * 2.3 + strand * Math.PI;
        var rnaT = index / (perChain * 3 - 1);
        var rnaAngle = rnaT * TAU * 2.65;
        var sy = 1 - (index / (perChain * 3 - 1)) * 2;
        var sr = Math.sqrt(1 - sy * sy),
          theta = Math.PI * (3 - Math.sqrt(5)) * index;
        var radius = 1.8 + random() * 0.95;
        var clusterTheta = Math.PI * (3 - Math.sqrt(5)) * j;
        var spread = Math.sqrt((j + 0.5) / perChain);
        nodes.push({
          branch: branch,
          index: j,
          flatIndex: index,
          strand: strand,
          base: [dx * r - dz * side, y, dz * r + dx * side],
          cloud: [
            Math.cos(theta) * sr * radius,
            sy * radius,
            Math.sin(theta) * sr * radius,
          ],
          map: [
            [-1.47, 0.2, 1.67][branch] + Math.cos(clusterTheta) * spread * 0.76,
            [0.58, -0.85, 0.71][branch] + Math.sin(clusterTheta) * spread * 0.6,
            (random() - 0.5) * 0.2,
          ],
          // Explicitly schematic geometry, independent of any biological sequence.
          dna: [
            Math.cos(dnaAngle) * 1.05,
            (helixT - 0.5) * 5,
            Math.sin(dnaAngle) * 1.05,
          ],
          rna: [
            Math.cos(rnaAngle) * 0.94,
            (rnaT - 0.5) * 5,
            Math.sin(rnaAngle) * 0.94,
          ],
          radius: 0.042 + random() * 0.025 + (j % 17 === 0 ? 0.013 : 0),
          x: 0,
          y: 0,
          z: 0,
          sx: 0,
          sy: 0,
          depth: 0,
          factor: 1,
        });
      }
    }
    for (var d = 0; d < (mobile ? 75 : 125); d++) {
      var a = random() * TAU,
        vy = random() * 2 - 1,
        wr = Math.sqrt(1 - vy * vy);
      var dr = 2.9 + random() * 2.1;
      dust.push({
        x: Math.cos(a) * wr * dr,
        y: vy * dr,
        z: Math.sin(a) * wr * dr - 2,
        radius: 0.5 + random() * 0.7,
        alpha: 0.15 + random() * 0.3,
      });
    }
    var sorted = nodes.slice();
    function updateGeometry() {
      var structure = Math.max(0, 1 - cloudMix - mapMix - dnaMix - rnaMix);
      nodes.forEach(function (node) {
        var direction = directions[node.branch];
        var displacement = explosion * (structure + cloudMix) * 0.83;
        node.x =
          node.base[0] * structure +
          node.cloud[0] * cloudMix +
          node.map[0] * mapMix +
          node.dna[0] * dnaMix +
          node.rna[0] * rnaMix +
          direction.x * displacement +
          (node.strand ? 1 : -1) * explosion * dnaMix * 0.66 +
          node.rna[0] * explosion * rnaMix * 0.35;
        node.y =
          node.base[1] * structure +
          node.cloud[1] * cloudMix +
          node.map[1] * mapMix +
          node.dna[1] * dnaMix +
          node.rna[1] * rnaMix;
        node.z =
          node.base[2] * structure +
          node.cloud[2] * cloudMix +
          node.map[2] * mapMix +
          node.dna[2] * dnaMix +
          node.rna[2] * rnaMix +
          direction.z * displacement +
          node.rna[2] * explosion * rnaMix * 0.35;
      });
    }
    updateGeometry();

    function project(x, y, z, rx, ry, rz) {
      var cy = Math.cos(ry),
        sy = Math.sin(ry),
        cx = Math.cos(rx),
        sx = Math.sin(rx),
        cz = Math.cos(rz),
        sz = Math.sin(rz);
      var nx = x * cy + z * sy;
      z = -x * sy + z * cy;
      x = nx;
      var ny = y * cx - z * sx;
      z = y * sx + z * cx;
      y = ny;
      nx = x * cz - y * sz;
      y = x * sz + y * cz;
      x = nx;
      var factor = 10.8 / (10.8 - z);
      return {
        x: width / 2 + x * scale * factor,
        y: height / 2 - y * scale * factor,
        z: z,
        factor: factor,
      };
    }
    function orbit(radius, rx, ry, rz, opacity, dashed) {
      context.beginPath();
      for (var i = 0; i <= 100; i++) {
        var angle = (i / 100) * TAU;
        var p = project(
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          0,
          rx,
          ry,
          rz,
        );
        if (i === 0) context.moveTo(p.x, p.y);
        else context.lineTo(p.x, p.y);
      }
      context.strokeStyle = "rgba(165,209,114," + opacity + ")";
      context.lineWidth = 0.8;
      context.setLineDash(dashed ? [1.5, 5] : []);
      context.stroke();
      context.setLineDash([]);
    }

    function drawRail(start, count, colorIndex, opacity) {
      if (opacity < 0.01) return;
      // Segment-by-segment lighting makes each turn recede behind the next.
      for (var i = start; i < start + count - 1; i += 3) {
        var a = nodes[i];
        var b = nodes[Math.min(start + count - 1, i + 3)];
        var light = clamp(0.46 + (a.depth + b.depth) * 0.075, 0.2, 0.85);
        context.lineCap = "round";
        context.beginPath();
        context.moveTo(a.sx, a.sy);
        for (var k = i + 1; k <= Math.min(start + count - 1, i + 3); k++) {
          context.lineTo(nodes[k].sx, nodes[k].sy);
        }
        context.strokeStyle =
          "rgba(" + rgb[colorIndex] + "," + opacity * light * 0.22 + ")";
        context.lineWidth = Math.max(2.5, scale * 0.11 * a.factor);
        context.stroke();
        context.strokeStyle =
          "rgba(" + rgb[colorIndex] + "," + opacity * light + ")";
        context.lineWidth = Math.max(1.2, scale * 0.037 * a.factor);
        context.stroke();
        context.strokeStyle =
          "rgba(239,255,245," + opacity * light * 0.34 + ")";
        context.lineWidth = Math.max(0.6, scale * 0.009);
        context.stroke();
      }
    }

    function drawHelixConnections() {
      var dnaOpacity = Math.pow(dnaMix, 1.35);
      var rnaOpacity = Math.pow(rnaMix, 1.35);
      var half = nodes.length / 2;
      var a, b, j;
      if (dnaOpacity > 0.01) {
        // Thirty-two paired rungs identify the two rails as a double helix.
        for (j = 0; j < 32; j++) {
          var index = Math.round((j / 31) * (half - 1));
          a = nodes[index];
          b = nodes[half + index];
          var rungLight = clamp(0.55 + (a.depth + b.depth) * 0.04, 0.3, 0.8);
          var gradient = context.createLinearGradient(a.sx, a.sy, b.sx, b.sy);
          gradient.addColorStop(
            0,
            "rgba(" + rgb[0] + "," + dnaOpacity * rungLight + ")",
          );
          gradient.addColorStop(
            0.48,
            "rgba(222,249,226," + dnaOpacity * rungLight * 0.58 + ")",
          );
          gradient.addColorStop(
            0.52,
            "rgba(218,249,245," + dnaOpacity * rungLight * 0.58 + ")",
          );
          gradient.addColorStop(
            1,
            "rgba(" + rgb[1] + "," + dnaOpacity * rungLight + ")",
          );
          context.beginPath();
          context.moveTo(a.sx, a.sy);
          context.lineTo(b.sx, b.sy);
          context.lineCap = "round";
          context.lineWidth = Math.max(
            1.3,
            scale * 0.043 * ((a.factor + b.factor) / 2),
          );
          context.strokeStyle = gradient;
          context.stroke();
          // A glancing highlight at each pair's midpoint adds a glass-like edge.
          context.beginPath();
          context.arc(
            (a.sx + b.sx) / 2,
            (a.sy + b.sy) / 2,
            Math.max(0.7, scale * 0.015),
            0,
            TAU,
          );
          context.fillStyle = "rgba(235,255,250," + dnaOpacity * 0.42 + ")";
          context.fill();
        }
        drawRail(0, half, 0, dnaOpacity);
        drawRail(half, half, 1, dnaOpacity);
      }
      if (rnaOpacity > 0.01) {
        drawRail(0, nodes.length, 0, rnaOpacity);
        for (j = 0; j < 34; j++) {
          a = nodes[Math.round((j / 33) * (nodes.length - 1))];
          b = project(
            a.x * 0.38,
            a.y,
            a.z * 0.38,
            renderedX,
            renderedY,
            renderedZ,
          );
          context.beginPath();
          context.moveTo(a.sx, a.sy);
          context.lineTo(b.x, b.y);
          context.strokeStyle = "rgba(" + rgb[1] + "," + rnaOpacity * 0.5 + ")";
          context.lineWidth = Math.max(1.1, scale * 0.032);
          context.lineCap = "round";
          context.stroke();
          var diameter = scale * 0.22 * b.factor;
          context.globalAlpha = rnaOpacity * clamp(0.72 + b.z * 0.09, 0.4, 1);
          context.drawImage(
            sprites[1],
            b.x - diameter / 2,
            b.y - diameter / 2,
            diameter,
            diameter,
          );
          context.globalAlpha = 1;
        }
      }
    }

    function draw() {
      if (destroyed || failed || width < 2 || height < 2) return;
      try {
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        context.clearRect(0, 0, width, height);
        context.globalAlpha = 1;
        var haze = context.createRadialGradient(
          width * 0.5,
          height * 0.46,
          scale * 0.25,
          width * 0.5,
          height * 0.46,
          scale * 3.65,
        );
        haze.addColorStop(0, "rgba(131,180,49,.08)");
        haze.addColorStop(1, "rgba(131,180,49,0)");
        context.fillStyle = haze;
        context.fillRect(0, 0, width, height);
        dust.forEach(function (dot) {
          var p = project(dot.x, dot.y, dot.z, 0, age * 0.012, 0);
          context.fillStyle = "rgba(178,222,144," + dot.alpha + ")";
          context.beginPath();
          context.arc(p.x, p.y, dot.radius, 0, TAU);
          context.fill();
        });
        var orbitFade = 1 - mapMix * 0.7;
        orbit(
          3.05,
          0.99,
          0.15 + Math.sin(age * 0.09) * 0.13,
          -0.55,
          0.32 * orbitFade,
          false,
        );
        orbit(3.3, 0.99, 0.15, -0.55, 0.14 * orbitFade, true);
        orbit(2.85, 1.13, -0.7, 0.8, 0.17 * orbitFade, false);
        orbit(3.35, 0.2, 0.8, 0.42, 0.08 * orbitFade, true);
        var bob = mode === "map" ? 0 : Math.sin(age * 0.65) * 0.055;
        nodes.forEach(function (node) {
          var p = project(
            node.x,
            node.y + bob,
            node.z,
            renderedX,
            renderedY,
            renderedZ,
          );
          node.sx = p.x;
          node.sy = p.y;
          node.depth = p.z;
          node.factor = p.factor;
        });
        var structureOpacity = Math.pow(
          Math.max(0, 1 - cloudMix - mapMix - dnaMix - rnaMix),
          2,
        );
        if (structureOpacity > 0.01) {
          for (var branch = 0; branch < 3; branch++) {
            context.beginPath();
            for (var j = 0; j < perChain; j++) {
              var node = nodes[branch * perChain + j];
              if (j === 0) context.moveTo(node.sx, node.sy);
              else context.lineTo(node.sx, node.sy);
            }
            context.strokeStyle =
              "rgba(" + rgb[branch] + "," + 0.44 * structureOpacity + ")";
            context.lineWidth = Math.max(0.65, scale * 0.013);
            context.lineJoin = "round";
            context.stroke();
            context.beginPath();
            for (j = Math.floor(perChain * 0.27); j < perChain - 7; j += 3) {
              var n1 = nodes[branch * perChain + j],
                n2 = nodes[branch * perChain + j + 7];
              var distance = Math.hypot(n1.x - n2.x, n1.y - n2.y, n1.z - n2.z);
              if (distance > 1.1) continue;
              context.moveTo(n1.sx, n1.sy);
              context.lineTo(n2.sx, n2.sy);
            }
            context.strokeStyle =
              "rgba(" + rgb[branch] + "," + 0.13 * structureOpacity + ")";
            context.lineWidth = 0.6;
            context.stroke();
          }
        }
        drawHelixConnections();
        sorted.sort(function (a, b) {
          return a.depth - b.depth;
        });
        var sizeScale =
          1 - cloudMix * 0.2 - mapMix * 0.16 - dnaMix * 0.12 - rnaMix * 0.26;
        sorted.forEach(function (node) {
          var radius = Math.max(
            0.8,
            node.radius * scale * node.factor * sizeScale,
          );
          var diameter = radius * 5.82;
          context.globalAlpha = clamp(
            0.7 + (node.depth + 2.5) * 0.065,
            0.55,
            1,
          );
          var colorIndex =
            dnaMix > 0.45 ? node.strand : rnaMix > 0.45 ? 0 : node.branch;
          context.drawImage(
            sprites[colorIndex],
            node.sx - diameter / 2,
            node.sy - diameter / 2,
            diameter,
            diameter,
          );
        });
        context.globalAlpha = 1;
        var theta = age * 0.21;
        var satellite = project(
          Math.cos(theta) * 3.05,
          Math.sin(theta) * 3.05,
          0,
          0.99,
          0.15,
          -0.55,
        );
        context.fillStyle = "#d5ff83";
        context.beginPath();
        context.arc(satellite.x, satellite.y, 2.5, 0, TAU);
        context.fill();
      } catch (error) {
        failed = true;
        stop();
        canvas.style.visibility = "hidden";
        onFallback({ reason: "canvas2d-render-error" });
      }
    }
    function canAnimate() {
      return (
        !destroyed &&
        !failed &&
        !paused &&
        !reduced &&
        inView &&
        !document.hidden
      );
    }
    function stop() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      previousTime = 0;
    }
    function settle() {
      cloudMix = cloudTarget;
      mapMix = mapTarget;
      dnaMix = dnaTarget;
      rnaMix = rnaTarget;
      explosion = explosionTarget;
      updateGeometry();
      renderedX =
        mode === "map"
          ? 0
          : rotationX + Math.sin(scrollProgress * Math.PI) * 0.14;
      renderedY =
        mode === "map"
          ? 0
          : rotationY + autoRotation + scrollProgress * Math.PI * 1.6;
      renderedZ = mode === "map" ? 0 : -0.16 + scrollProgress * 0.18;
      draw();
    }
    function frame(time) {
      raf = 0;
      if (!canAnimate()) return;
      var interval =
        quality === "low" || (mobile && quality !== "high")
          ? 1000 / 30
          : 1000 / 60;
      if (previousTime && time - previousTime < interval - 1) {
        raf = requestAnimationFrame(frame);
        return;
      }
      var dt = previousTime
        ? Math.min((time - previousTime) / 1000, 0.05)
        : 1 / 60;
      previousTime = time;
      age += dt;
      var lerp = 1 - Math.exp(-dt * 5);
      if (
        Math.abs(cloudTarget - cloudMix) +
          Math.abs(mapTarget - mapMix) +
          Math.abs(dnaTarget - dnaMix) +
          Math.abs(rnaTarget - rnaMix) +
          Math.abs(explosionTarget - explosion) >
        0.0002
      ) {
        cloudMix = mix(cloudMix, cloudTarget, lerp);
        mapMix = mix(mapMix, mapTarget, lerp);
        dnaMix = mix(dnaMix, dnaTarget, lerp);
        rnaMix = mix(rnaMix, rnaTarget, lerp);
        explosion = mix(explosion, explosionTarget, lerp);
        updateGeometry();
      }
      if (!dragging && mode !== "map") autoRotation += dt * 0.085;
      var rotationLerp = 1 - Math.exp(-dt * 7);
      scrollProgress = mix(scrollProgress, scrollTarget, rotationLerp);
      renderedX = mix(
        renderedX,
        mode === "map"
          ? 0
          : rotationX + Math.sin(scrollProgress * Math.PI) * 0.14,
        rotationLerp,
      );
      renderedY = mix(
        renderedY,
        mode === "map"
          ? 0
          : rotationY + autoRotation + scrollProgress * Math.PI * 1.6,
        rotationLerp,
      );
      renderedZ = mix(
        renderedZ,
        mode === "map" ? 0 : -0.16 + scrollProgress * 0.18,
        rotationLerp,
      );
      draw();
      if (canAnimate()) raf = requestAnimationFrame(frame);
    }
    function requestRender() {
      if (destroyed || failed || !inView || document.hidden) return;
      if (canAnimate()) {
        if (!raf) raf = requestAnimationFrame(frame);
      } else settle();
    }
    function resize() {
      if (destroyed) return;
      var box = container.getBoundingClientRect();
      if (box.width < 1 || box.height < 1) return;
      mobile = narrow.matches;
      width = Math.round(box.width);
      height = Math.round(box.height);
      pixelRatio = Math.min(
        window.devicePixelRatio || 1,
        quality === "low" ? 1 : mobile ? 1.2 : 1.5,
      );
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      scale = Math.min(width, height) / 7.3;
      requestRender();
    }
    function listen(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      cleanups.push(function () {
        target.removeEventListener(type, handler, options);
      });
    }
    function down(event) {
      if (mode === "map" || (event.button !== undefined && event.button !== 0))
        return;
      drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        type: event.pointerType,
      };
      if (event.pointerType === "mouse") {
        dragging = true;
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
      }
    }
    function move(event) {
      if (!drag || event.pointerId !== drag.id) return;
      var x = event.clientX - drag.x,
        y = event.clientY - drag.y;
      if (!dragging && drag.type !== "mouse") {
        if (Math.abs(y) > Math.abs(x) && Math.abs(y) > 8) {
          drag = null;
          return;
        }
        if (Math.abs(x) > 8 && Math.abs(x) > Math.abs(y)) {
          dragging = true;
          canvas.setPointerCapture(event.pointerId);
        }
      }
      if (!dragging) return;
      rotationY += (event.clientX - drag.lastX) * 0.007;
      rotationX = clamp(
        rotationX + (event.clientY - drag.lastY) * 0.004,
        -0.75,
        0.75,
      );
      drag.lastX = event.clientX;
      drag.lastY = event.clientY;
      requestRender();
    }
    function up() {
      var previous = drag;
      drag = null;
      dragging = false;
      if (previous && canvas.hasPointerCapture(previous.id))
        canvas.releasePointerCapture(previous.id);
      canvas.style.cursor = mode === "map" ? "default" : "grab";
    }
    listen(canvas, "pointerdown", down);
    listen(canvas, "pointermove", move);
    listen(canvas, "pointerup", up);
    listen(canvas, "pointercancel", up);
    listen(canvas, "lostpointercapture", up);
    listen(document, "visibilitychange", function () {
      if (document.hidden) stop();
      else requestRender();
    });
    listen(motionQuery, "change", function (event) {
      reduced = event.matches;
      stop();
      requestRender();
    });
    listen(narrow, "change", resize);
    if (typeof ResizeObserver !== "undefined") {
      var ro = new ResizeObserver(resize);
      ro.observe(container);
      cleanups.push(function () {
        ro.disconnect();
      });
    } else listen(window, "resize", resize, { passive: true });
    if (typeof IntersectionObserver !== "undefined") {
      var observer = new IntersectionObserver(
        function (entries) {
          inView = entries[0].isIntersecting;
          if (inView) requestRender();
          else stop();
        },
        { rootMargin: "100px", threshold: 0 },
      );
      observer.observe(container);
      cleanups.push(function () {
        observer.disconnect();
      });
    }
    function dispose() {
      if (destroyed) return;
      destroyed = true;
      stop();
      cleanups.forEach(function (fn) {
        fn();
      });
      canvas.remove();
      nodes.length = 0;
      dust.length = 0;
      sorted.length = 0;
      sprites.length = 0;
    }
    resize();
    draw();
    // Release a failed first frame before the wrapper reports the fallback.
    if (failed) {
      dispose();
      return null;
    }
    onReady({ renderer: "canvas2d", conceptual: true });
    requestRender();
    return {
      setMode: function (value) {
        if (["structure", "rna", "dna", "points", "map"].indexOf(value) === -1)
          return;
        mode = value;
        cloudTarget = value === "points" ? 1 : 0;
        mapTarget = value === "map" ? 1 : 0;
        dnaTarget = value === "dna" ? 1 : 0;
        rnaTarget = value === "rna" ? 1 : 0;
        canvas.style.cursor = value === "map" ? "default" : "grab";
        requestRender();
      },
      setScroll: function (value) {
        scrollTarget = clamp(Number(value) || 0, 0, 1);
        if (!paused && !reduced) requestRender();
      },
      setExplode: function (value) {
        explosionTarget = clamp(Number(value) || 0, 0, 1);
        requestRender();
      },
      setPaused: function (value) {
        paused = Boolean(value);
        stop();
        requestRender();
      },
      setQuality: function (value) {
        quality = ["high", "low"].indexOf(value) !== -1 ? value : "auto";
        resize();
      },
      reset: function () {
        rotationX = -0.07;
        rotationY = -0.36;
        autoRotation = 0;
        scrollProgress = 0;
        scrollTarget = 0;
        explosionTarget = 0;
        mode = "structure";
        cloudTarget = 0;
        mapTarget = 0;
        dnaTarget = 0;
        rnaTarget = 0;
        canvas.style.cursor = "grab";
        requestRender();
      },
      destroy: function () {
        dispose();
      },
    };
  }
  function createController() {
    var current = null;
    return {
      init: function (options) {
        options = options || {};
        if (current) current.destroy();
        current = null;
        var initializing = true,
          notifications = [];
        function notify(name, detail) {
          var callback = options[name];
          if (typeof callback !== "function") return;
          if (initializing)
            notifications.push(function () {
              callback(detail);
            });
          else callback(detail);
        }
        current = createScene({
          container: options.container,
          canvas: options.canvas,
          onReady: function (detail) {
            notify("onReady", detail);
          },
          onFallback: function (detail) {
            notify("onFallback", detail);
          },
        });
        var initialized = Boolean(current);
        initializing = false;
        notifications.forEach(function (deliver) {
          deliver();
        });
        return initialized;
      },
      setMode: function (value) {
        if (current) current.setMode(value);
      },
      setExplode: function (value) {
        if (current) current.setExplode(value);
      },
      setScroll: function (value) {
        if (current) current.setScroll(value);
      },
      setPaused: function (value) {
        if (current) current.setPaused(value);
      },
      setQuality: function (value) {
        if (current) current.setQuality(value);
      },
      reset: function () {
        if (current) current.reset();
      },
      destroy: function () {
        if (current) current.destroy();
        current = null;
      },
    };
  }
  // The default controller preserves the original hero API. Additional ones own
  // their canvas, frame loop, geometry, listeners and lifecycle independently.
  var defaultController = createController();
  defaultController.create = createController;
  window.InflumaticsFallbackScene = defaultController;
})();
