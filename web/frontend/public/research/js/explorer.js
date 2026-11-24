/*
 * Influmatics · an intentionally illustrative data landscape.
 * All samples, coordinates and groups below are synthetic design fixtures.
 * They are NOT HA sequences, measured HI titres, antigenic distances or predictions.
 * Vanilla JavaScript; works from file:// without a network or a build step.
 */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = ["#c4ff47", "#76dce8", "#e9d6af"];
  const GROUPS = ["A", "B", "C"];
  const START_YEAR = 2003;
  const END_YEAR = 2022;
  let instances = 0;

  function svgElement(tag, attributes, text) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes || {}).forEach(([key, value]) =>
      node.setAttribute(key, String(value)),
    );
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function seedRandom(seed) {
    return function () {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    };
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function makeSamples() {
    const random = seedRandom(3292026);
    const centers = [
      [225, 176],
      [426, 310],
      [670, 220],
    ];
    const sequenceCenters = [
      [287, 302],
      [502, 179],
      [665, 332],
    ];
    const samples = [];

    for (let year = START_YEAR; year <= END_YEAR; year += 1) {
      for (let slot = 0; slot < 4; slot += 1) {
        const group = (year - START_YEAR + slot) % GROUPS.length;
        const angle = random() * Math.PI * 2;
        const radius = 14 + Math.sqrt(random()) * 77;
        const lean = ((year - START_YEAR) / (END_YEAR - START_YEAR) - 0.5) * 39;
        const x = centers[group][0] + Math.cos(angle) * radius * 1.19 + lean;
        const y =
          centers[group][1] + Math.sin(angle) * radius * 0.76 + lean * 0.35;
        const sequenceAngle = angle + 0.68 + group * 0.41;
        samples.push({
          index: samples.length,
          id: `DEMO-${year}-${String(slot + 1).padStart(2, "0")}`,
          year,
          group,
          size: 2.6 + random() * 1.6,
          antigenic: [x, y],
          sequence: [
            sequenceCenters[group][0] +
              Math.cos(sequenceAngle) * radius * 1.5 -
              lean,
            sequenceCenters[group][1] +
              Math.sin(sequenceAngle) * radius * 1.08 +
              lean * 0.5,
          ],
        });
      }
    }
    return samples;
  }

  function init(options) {
    const opts = options || {};
    const svg = opts.svg;
    if (!(svg instanceof SVGElement)) return null;

    const samples = makeSamples();
    const aborter = new AbortController();
    const signal = aborter.signal;
    const unique = `im-map-${++instances}`;
    const viewButtons = Array.from(opts.viewButtons || []);
    const tooltip = opts.tooltip || null;
    const initialTitleNodes = opts.detailTitle
      ? Array.from(opts.detailTitle.childNodes, (node) => node.cloneNode(true))
      : [];
    const initialBody = opts.detailBody ? opts.detailBody.textContent : "";
    let currentView = "antigenic";
    let currentYear = END_YEAR;
    let selected = null;
    let focusedIndex = 0;
    let hovered = null;
    let destroyed = false;
    let motionPaused = false;

    svg.replaceChildren();
    svg.setAttribute("viewBox", "0 0 900 520");
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-labelledby", `${unique}-title ${unique}-desc`);
    svg.append(
      svgElement(
        "title",
        { id: `${unique}-title` },
        "Influmatics 개념 탐색 지도",
      ),
      svgElement(
        "desc",
        { id: `${unique}-desc` },
        "80개의 가상 샘플로 항원 공간과 서열 공간의 시각화 개념을 설명합니다. 실제 측정 결과가 아닙니다. 탭으로 점에 진입한 후 방향키로 이동하고 Enter 또는 Space로 선택할 수 있습니다.",
      ),
    );

    const defs = svgElement("defs");
    const style = svgElement("style");
    style.textContent = `
      .im-point { cursor: pointer; outline: none; transition: transform .85s cubic-bezier(.2,.7,.2,1), opacity .28s ease; }
      .im-point-core { transition: r .2s ease, fill .2s ease; }
      .im-point-ring { opacity: 0; transition: opacity .2s ease; }
      .im-point-halo { opacity: .35; transition: opacity .2s ease; }
      .im-point:hover .im-point-core, .im-point:focus .im-point-core, .im-point.is-selected .im-point-core { r: 5.5px; }
      .im-point:hover .im-point-ring, .im-point:focus .im-point-ring, .im-point.is-selected .im-point-ring { opacity: 1; }
      .im-point:hover .im-point-halo, .im-point:focus .im-point-halo, .im-point.is-selected .im-point-halo { opacity: 1; }
      .im-map-caption { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: 1.3px; }
      .im-zone { transition: opacity .5s ease; }
      body.motion-paused .im-point, body.motion-paused .im-point *, body.motion-paused .im-zone,
      svg.is-motion-paused .im-point, svg.is-motion-paused .im-point *, svg.is-motion-paused .im-zone { transition: none !important; }
      @media (prefers-reduced-motion: reduce) { .im-point, .im-point-core, .im-point-ring, .im-point-halo, .im-zone { transition: none; } }
    `;
    defs.append(style);
    COLORS.forEach((color, index) => {
      const gradient = svgElement("radialGradient", {
        id: `${unique}-glow-${index}`,
      });
      gradient.append(
        svgElement("stop", {
          offset: "0%",
          "stop-color": color,
          "stop-opacity": ".4",
        }),
        svgElement("stop", {
          offset: "45%",
          "stop-color": color,
          "stop-opacity": ".1",
        }),
        svgElement("stop", {
          offset: "100%",
          "stop-color": color,
          "stop-opacity": "0",
        }),
      );
      defs.append(gradient);
    });
    svg.append(defs);

    const grid = svgElement("g", {
      fill: "none",
      stroke: "#52616e",
      "stroke-width": "1",
      "aria-hidden": "true",
    });
    for (let x = 84; x <= 840; x += 126) {
      grid.append(svgElement("path", { d: `M${x} 55V445`, opacity: ".16" }));
    }
    for (let y = 65; y <= 445; y += 76) {
      grid.append(svgElement("path", { d: `M70 ${y}H850`, opacity: ".16" }));
    }
    // Four small registration marks frame the plot without suggesting a numeric scale.
    [
      "M70 78V55H93",
      "M827 55H850V78",
      "M70 422V445H93",
      "M827 445H850V422",
    ].forEach((d) =>
      grid.append(svgElement("path", { d, stroke: "#81929e", opacity: ".42" })),
    );
    svg.append(grid);

    const zones = svgElement("g", {
      class: "im-zone",
      "aria-hidden": "true",
      fill: "none",
    });
    const zoneLocations = [
      [225, 176, 122, 90, -16],
      [426, 310, 117, 95, 22],
      [670, 220, 125, 91, -14],
    ];
    zoneLocations.forEach(([cx, cy, rx, ry, rotate], i) => {
      for (let ring = 0; ring < 2; ring += 1) {
        zones.append(
          svgElement("ellipse", {
            cx,
            cy,
            rx: rx + ring * 10,
            ry: ry + ring * 8,
            transform: `rotate(${rotate} ${cx} ${cy})`,
            stroke: COLORS[i],
            "stroke-width": ".7",
            "stroke-dasharray": ring ? "2 9" : "none",
            opacity: ring ? ".08" : ".12",
          }),
        );
      }
    });
    svg.append(zones);

    const links = svgElement("g", {
      "aria-hidden": "true",
      fill: "none",
      "stroke-width": "1",
      "stroke-dasharray": "3 6",
      "pointer-events": "none",
    });
    svg.append(links);
    const pointsLayer = svgElement("g");
    svg.append(pointsLayer);

    const labels = svgElement("g", {
      "aria-hidden": "true",
      "pointer-events": "none",
    });
    const axisX = svgElement(
      "text",
      {
        x: 460,
        y: 485,
        fill: "#7e919b",
        "font-size": "12",
        "text-anchor": "middle",
        class: "im-map-caption",
      },
      "MDS DIMENSION 1",
    );
    const axisY = svgElement(
      "text",
      {
        x: 24,
        y: 250,
        fill: "#7e919b",
        "font-size": "12",
        "text-anchor": "middle",
        class: "im-map-caption",
        transform: "rotate(-90 24 250)",
      },
      "MDS DIMENSION 2",
    );
    const sampleCount = svgElement("text", {
      x: 849,
      y: 28,
      fill: "#9eafb8",
      "font-size": "12",
      "text-anchor": "end",
      class: "im-map-caption",
    });
    labels.append(
      svgElement(
        "text",
        {
          x: 70,
          y: 28,
          fill: "#7e919b",
          "font-size": "12",
          class: "im-map-caption",
        },
        "ILLUSTRATIVE DATA",
      ),
      sampleCount,
      axisX,
      axisY,
    );
    svg.append(labels);

    const sampleLabels = new Map();
    const nodes = new Map();

    function fitHitTargets() {
      // Preserve a 24 CSS-pixel hit diameter when the viewBox shrinks on phones.
      const scale = (svg.getBoundingClientRect().width || 900) / 900;
      const radius = Math.max(12, 12 / scale);
      nodes.forEach((node) =>
        node.querySelector(".im-point-hit").setAttribute("r", String(radius)),
      );
    }

    function visibleSamples() {
      return samples.filter((sample) => sample.year <= currentYear);
    }

    function positionTooltip(sample) {
      if (!tooltip || !sample || destroyed) return;
      const parent = tooltip.offsetParent || svg.parentElement;
      if (!parent) return;
      const parentRect = parent.getBoundingClientRect();
      const svgRect = svg.getBoundingClientRect();
      const matrix = nodes.get(sample.index).getScreenCTM();
      let x;
      let y;
      if (matrix) {
        x = matrix.e - parentRect.left;
        y = matrix.f - parentRect.top;
      } else {
        const coordinate = sample[currentView];
        x =
          svgRect.left -
          parentRect.left +
          (coordinate[0] * svgRect.width) / 900;
        y =
          svgRect.top - parentRect.top + (coordinate[1] * svgRect.height) / 520;
      }
      const width = tooltip.offsetWidth || 200;
      const height = tooltip.offsetHeight || 76;
      const maxWidth = parent.clientWidth || parentRect.width;
      const maxHeight = parent.clientHeight || parentRect.height;
      tooltip.style.left = `${clamp(x + 15, 8, Math.max(8, maxWidth - width - 8))}px`;
      tooltip.style.top = `${clamp(y - height - 13, 8, Math.max(8, maxHeight - height - 8))}px`;
    }

    function showTooltip(sample) {
      if (!tooltip || !sample) return;
      const title = document.createElement("strong");
      title.textContent = sample.id;
      title.style.color = COLORS[sample.group];
      const subtitle = document.createElement("span");
      subtitle.textContent = `${sample.year} · GROUP ${GROUPS[sample.group]} · 설명용 예시`;
      subtitle.style.display = "block";
      subtitle.style.marginTop = "5px";
      tooltip.replaceChildren(title, subtitle);
      tooltip.hidden = false;
      tooltip.setAttribute("aria-hidden", "true");
      tooltip.style.pointerEvents = "none";
      positionTooltip(sample);
    }

    function hideTooltip() {
      if (tooltip) tooltip.hidden = true;
    }

    function updateRovingTabindex() {
      const candidates = visibleSamples();
      if (!candidates.some((sample) => sample.index === focusedIndex))
        focusedIndex = candidates[0].index;
      samples.forEach((sample) => {
        nodes
          .get(sample.index)
          .setAttribute(
            "tabindex",
            sample.index === focusedIndex && sample.year <= currentYear
              ? "0"
              : "-1",
          );
      });
    }

    function renderConnections() {
      links.replaceChildren();
      if (!selected || selected.year > currentYear) return;
      const [x, y] = selected[currentView];
      const neighbors = visibleSamples()
        .filter((sample) => sample !== selected)
        .map((sample) => ({
          sample,
          distance: Math.hypot(
            sample[currentView][0] - x,
            sample[currentView][1] - y,
          ),
        }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, 3);
      neighbors.forEach(({ sample }) => {
        links.append(
          svgElement("path", {
            d: `M${x} ${y}L${sample[currentView][0]} ${sample[currentView][1]}`,
            stroke: COLORS[selected.group],
            opacity: ".38",
          }),
        );
      });
    }

    function clearSelection() {
      selected = null;
      samples.forEach((sample) => {
        const node = nodes.get(sample.index);
        node.classList.remove("is-selected");
        node.setAttribute("aria-pressed", "false");
      });
      if (opts.detailTitle)
        opts.detailTitle.replaceChildren(
          ...initialTitleNodes.map((node) => node.cloneNode(true)),
        );
      if (opts.detailBody) opts.detailBody.textContent = initialBody;
      links.replaceChildren();
      hideTooltip();
    }

    function selectSample(sample) {
      selected = sample;
      focusedIndex = sample.index;
      samples.forEach((item) => {
        const node = nodes.get(item.index);
        node.classList.toggle("is-selected", item === sample);
        node.setAttribute("aria-pressed", String(item === sample));
      });
      if (opts.detailTitle) opts.detailTitle.textContent = sample.id;
      if (opts.detailBody) {
        opts.detailBody.textContent = `${sample.year}년으로 표시한 가상 샘플 · 그룹 ${GROUPS[sample.group]}. 선으로 연결된 점은 현재 화면에서 가까운 예시들입니다. 좌표와 그룹은 개념 설명용이며, 실제 항원 거리나 모델의 분석 결과가 아닙니다.`;
      }
      updateRovingTabindex();
      renderConnections();
      showTooltip(sample);
      if (typeof opts.onSelect === "function") {
        opts.onSelect({
          id: sample.id,
          year: sample.year,
          group: GROUPS[sample.group],
          illustrative: true,
        });
      }
    }

    function moveFocus(sample, key) {
      const visible = visibleSamples();
      let target;
      if (key === "Home") target = visible[0];
      else if (key === "End") target = visible[visible.length - 1];
      else {
        const [x, y] = sample[currentView];
        const axis = key === "ArrowLeft" || key === "ArrowRight" ? 0 : 1;
        const direction = key === "ArrowRight" || key === "ArrowDown" ? 1 : -1;
        const coordinate = axis === 0 ? x : y;
        target = visible
          .filter(
            (candidate) =>
              candidate !== sample &&
              (candidate[currentView][axis] - coordinate) * direction > 0,
          )
          .map((candidate) => ({
            candidate,
            cost:
              Math.hypot(
                candidate[currentView][0] - x,
                candidate[currentView][1] - y,
              ) +
              Math.abs(
                candidate[currentView][1 - axis] -
                  sample[currentView][1 - axis],
              ) *
                1.2,
          }))
          .sort((a, b) => a.cost - b.cost)[0]?.candidate;
        if (!target) {
          target = visible
            .slice()
            .sort(
              (a, b) =>
                (a[currentView][axis] - b[currentView][axis]) * direction,
            )[0];
        }
      }
      focusedIndex = target.index;
      updateRovingTabindex();
      nodes.get(target.index).focus({ preventScroll: true });
    }

    samples.forEach((sample) => {
      const point = svgElement("g", {
        class: "im-point",
        role: "button",
        tabindex: "-1",
        "aria-pressed": "false",
        "aria-label": `${sample.id}, ${sample.year}년, 예시 그룹 ${GROUPS[sample.group]}. 개념 설명용 가상 샘플`,
        "data-sample": sample.id,
      });
      point.style.transform = `translate(${sample.antigenic[0]}px, ${sample.antigenic[1]}px)`;
      point.append(
        svgElement("circle", {
          r: 21,
          fill: `url(#${unique}-glow-${sample.group})`,
          class: "im-point-halo",
          "pointer-events": "none",
        }),
        svgElement("circle", {
          r: 12,
          fill: "transparent",
          "pointer-events": "all",
          class: "im-point-hit",
        }),
        svgElement("circle", {
          r: 10,
          fill: "none",
          stroke: COLORS[sample.group],
          "stroke-width": "1",
          class: "im-point-ring",
          "pointer-events": "none",
        }),
        svgElement("circle", {
          r: sample.size,
          fill: COLORS[sample.group],
          class: "im-point-core",
          "pointer-events": "none",
        }),
      );
      point.addEventListener(
        "pointerenter",
        () => {
          hovered = sample;
          showTooltip(sample);
        },
        { signal },
      );
      point.addEventListener(
        "pointerleave",
        () => {
          hovered = null;
          if (selected) showTooltip(selected);
          else hideTooltip();
        },
        { signal },
      );
      point.addEventListener(
        "focus",
        () => {
          focusedIndex = sample.index;
          updateRovingTabindex();
          showTooltip(sample);
        },
        { signal },
      );
      point.addEventListener(
        "blur",
        () => {
          if (selected) showTooltip(selected);
          else if (!hovered) hideTooltip();
        },
        { signal },
      );
      point.addEventListener("click", () => selectSample(sample), { signal });
      point.addEventListener(
        "keydown",
        (event) => {
          if (
            [
              "ArrowLeft",
              "ArrowRight",
              "ArrowUp",
              "ArrowDown",
              "Home",
              "End",
            ].includes(event.key)
          ) {
            event.preventDefault();
            moveFocus(sample, event.key);
          } else if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectSample(sample);
          } else if (event.key === "Escape") {
            event.preventDefault();
            clearSelection();
          }
        },
        { signal },
      );
      nodes.set(sample.index, point);
      pointsLayer.append(point);
      if ([12, 29, 50, 75].includes(sample.index)) {
        const label = svgElement(
          "text",
          {
            fill: COLORS[sample.group],
            "font-size": "12",
            "font-family": "ui-monospace, SFMono-Regular, Consolas, monospace",
            opacity: ".76",
          },
          sample.id,
        );
        sampleLabels.set(sample.index, label);
        labels.append(label);
      }
    });

    function render() {
      if (destroyed) return;
      samples.forEach((sample) => {
        const point = nodes.get(sample.index);
        const visible = sample.year <= currentYear;
        const position = sample[currentView];
        point.style.transform = `translate(${position[0]}px, ${position[1]}px)`;
        point.style.opacity = visible ? "1" : "0";
        point.style.pointerEvents = visible ? "auto" : "none";
        point.style.visibility = visible ? "visible" : "hidden";
        point.setAttribute("aria-hidden", String(!visible));
        const label = sampleLabels.get(sample.index);
        if (label) {
          label.setAttribute("x", position[0] + 13);
          label.setAttribute("y", position[1] - 11);
          label.setAttribute("opacity", visible ? ".76" : "0");
        }
      });
      if (selected && selected.year > currentYear) clearSelection();
      updateRovingTabindex();
      sampleCount.textContent = `${visibleSamples().length} / ${samples.length} DEMO SAMPLES`;
      axisX.textContent =
        currentView === "antigenic" ? "MDS DIMENSION 1" : "EMBEDDING AXIS 1";
      axisY.textContent =
        currentView === "antigenic" ? "MDS DIMENSION 2" : "EMBEDDING AXIS 2";
      zones.style.opacity = currentView === "antigenic" ? "1" : "0";
      if (opts.yearInput) {
        opts.yearInput.value = String(currentYear);
        opts.yearInput.setAttribute(
          "aria-valuetext",
          `${currentYear}년까지의 설명용 샘플 ${visibleSamples().length}개`,
        );
        opts.yearInput.style.setProperty(
          "--range-progress",
          `${((currentYear - START_YEAR) / (END_YEAR - START_YEAR)) * 100}%`,
        );
      }
      if (opts.yearOutput) opts.yearOutput.textContent = String(currentYear);
      viewButtons.forEach((button) => {
        const active =
          (button.dataset.mapView || button.dataset.view || button.value) ===
          currentView;
        button.setAttribute("aria-pressed", String(active));
        button.classList.toggle("is-active", active);
      });
      renderConnections();
      fitHitTargets();
      // Hide during a morph: a tooltip must not appear attached to the old position.
      hideTooltip();
    }

    function setYear(year) {
      const parsed = Number(year);
      currentYear = Number.isFinite(parsed)
        ? clamp(Math.round(parsed), START_YEAR, END_YEAR)
        : END_YEAR;
      render();
    }

    function setView(view) {
      if (view !== "antigenic" && view !== "sequence") return;
      currentView = view;
      render();
    }

    function reset() {
      currentYear = END_YEAR;
      currentView = "antigenic";
      clearSelection();
      render();
    }

    function setPaused(paused) {
      motionPaused = Boolean(paused);
      svg.classList.toggle("is-motion-paused", motionPaused);
      // The explorer never advances itself. Disabling transitions also makes
      // user-requested coordinate changes instantaneous while motion is off.
    }

    if (tooltip) {
      tooltip.hidden = true;
      tooltip.style.position = "absolute";
    }
    if (opts.yearInput) {
      opts.yearInput.min = String(START_YEAR);
      opts.yearInput.max = String(END_YEAR);
      opts.yearInput.step = "1";
      opts.yearInput.addEventListener(
        "input",
        (event) => setYear(event.target.value),
        { signal },
      );
    }
    viewButtons.forEach((button) =>
      button.addEventListener(
        "click",
        () =>
          setView(
            button.dataset.mapView || button.dataset.view || button.value,
          ),
        { signal },
      ),
    );
    if (opts.resetButton)
      opts.resetButton.addEventListener("click", reset, { signal });
    window.addEventListener(
      "resize",
      () => {
        fitHitTargets();
        if (selected && tooltip && !tooltip.hidden) positionTooltip(selected);
      },
      { signal, passive: true },
    );
    svg.addEventListener(
      "click",
      (event) => {
        if (!event.target.closest(".im-point")) clearSelection();
      },
      { signal },
    );
    render();

    return {
      setYear,
      setView,
      reset,
      setPaused,
      destroy() {
        destroyed = true;
        aborter.abort();
        hideTooltip();
      },
      getState() {
        return {
          year: currentYear,
          view: currentView,
          visibleSamples: visibleSamples().length,
          selected: selected ? selected.id : null,
          paused: motionPaused,
          illustrative: true,
        };
      },
    };
  }

  window.InflumaticsExplorer = Object.freeze({ init });
})();
