/* A scroll-led protein → RNA → DNA tour. All geometry is conceptual. */
(function () {
  "use strict";
  const section = document.querySelector("#molecules");
  if (!section) return;
  const stage = document.querySelector("#story-stage");
  const canvas = document.querySelector("#story-canvas");
  const poster = document.querySelector("#story-poster");
  const title = document.querySelector("#story-caption-title");
  const body = document.querySelector("#story-caption-body");
  const note = document.querySelector("#story-caption-note");
  const label = document.querySelector("#story-type");
  const sideLabel = document.querySelector("#story-side-type");
  const progressFill = document.querySelector("#story-progress-fill");
  const buttons = [...document.querySelectorAll("[data-story-mode]")];
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const config = window.INFLUMATICS_CONFIG || {};
  const phases = [
    {
      mode: "structure",
      label: "HA PROTEIN",
      short: "PROTEIN",
      title: "우리가 살펴보는 단백질, HA.",
      body: "HA1은 HA 단백질의 일부입니다. Influmatics에서는 이 부분의 아미노산 서열을 비교합니다.",
      note: "HA에서 영감을 받은 개념 모형입니다. 실제 구조 좌표는 사용하지 않았습니다.",
    },
    {
      mode: "rna",
      label: "SINGLE-STRANDED RNA",
      short: "RNA",
      title: "인플루엔자의 유전물질은 RNA입니다.",
      body: "인플루엔자는 단일가닥 RNA를 유전물질로 가집니다. 여기서는 RNA 한 가닥의 모습을 단순화해 보여줍니다.",
      note: "RNA 한 가닥을 설명하는 모형입니다. 실제 바이러스 유전체의 접힌 모양을 재현하지는 않았습니다.",
    },
    {
      mode: "dna",
      label: "DNA · STRUCTURE COMPARISON",
      short: "DNA",
      title: "익숙한 이중나선과 비교해 보세요.",
      body: "두 가닥이 짝을 이루는 DNA의 기본 형태입니다. RNA와는 어떻게 다른지, 직접 돌려보며 비교할 수 있습니다.",
      note: "구조 비교용 DNA 모형입니다. 인플루엔자의 유전체는 RNA입니다.",
    },
  ];
  let activeIndex = 0;
  let currentProgress = 0;
  let ready = false;
  let fallback = false;
  let paused = document.body.classList.contains("motion-paused");
  let pending = false;
  let manualSelection = false;
  let previousScroll = window.scrollY;
  let renderer = window.InflumaticsScene?.create?.() || null;

  function markReady(info) {
    ready = true;
    stage.classList.add("is-ready");
    stage.dataset.renderer = info.renderer;
    poster.setAttribute("aria-hidden", "true");
    canvas.setAttribute("aria-hidden", String(info.renderer !== "webgl2"));
  }
  function staticFallback() {
    ready = false;
    stage.classList.remove("is-ready");
    stage.classList.add("is-static");
    stage.dataset.renderer = "svg";
    poster.removeAttribute("aria-hidden");
    note.textContent =
      "이 브라우저에서는 정적 모형으로 표시됩니다. 설명은 아래 버튼으로 읽을 수 있습니다.";
  }
  function applyRendererState() {
    if (!renderer || !ready) return;
    renderer.setMode(phases[activeIndex].mode);
    renderer.setScroll?.(currentProgress);
    renderer.setPaused(paused);
    renderer.setQuality(
      ["auto", "high", "low"].includes(config.quality)
        ? config.quality
        : "auto",
    );
  }
  function startFallback() {
    if (fallback) return;
    fallback = true;
    renderer?.destroy();
    renderer = window.InflumaticsFallbackScene?.create?.() || null;
    if (!renderer) {
      staticFallback();
      return;
    }
    try {
      renderer.init({
        container: stage,
        canvas,
        onReady: markReady,
        onFallback: staticFallback,
      });
      applyRendererState();
    } catch (_) {
      staticFallback();
    }
  }
  try {
    if (renderer) {
      const ok = renderer.init({
        container: stage,
        canvas,
        onReady: markReady,
        onFallback: startFallback,
      });
      if (!ok && !ready) startFallback();
    } else startFallback();
    applyRendererState();
  } catch (_) {
    startFallback();
  }

  function selectPhase(index) {
    const phase = phases[index];
    activeIndex = index;
    title.textContent = phase.title;
    body.textContent = phase.body;
    note.textContent = phase.note;
    label.textContent = phase.label;
    sideLabel.textContent = phase.short;
    stage.dataset.model = phase.mode;
    buttons.forEach((button, i) =>
      button.setAttribute("aria-pressed", String(i === index)),
    );
    if (ready) renderer.setMode(phase.mode);
  }
  function update() {
    pending = false;
    const rect = section.getBoundingClientRect();
    const travel = Math.max(1, section.offsetHeight - window.innerHeight);
    const progress = Math.max(0, Math.min(1, -rect.top / travel));
    const scrollChanged = Math.abs(window.scrollY - previousScroll) > 2;
    previousScroll = window.scrollY;
    if (scrollChanged && !reduced.matches) manualSelection = false;
    currentProgress = progress;
    progressFill.style.transform = `scaleX(${progress})`;
    if (!paused && !reduced.matches && !manualSelection) {
      const index = progress < 0.32 ? 0 : progress < 0.66 ? 1 : 2;
      if (index !== activeIndex) selectPhase(index);
    }
    if (
      ready &&
      !paused &&
      !reduced.matches &&
      rect.top < window.innerHeight &&
      rect.bottom > 0
    )
      renderer.setScroll?.(progress);
  }
  function queueUpdate() {
    if (!pending) {
      pending = true;
      requestAnimationFrame(update);
    }
  }
  window.addEventListener("scroll", queueUpdate, { passive: true });
  window.addEventListener("resize", queueUpdate, { passive: true });
  window.addEventListener("influmatics-motion", (event) => {
    paused = Boolean(event.detail?.paused);
    if (ready) renderer.setPaused(paused);
  });
  buttons.forEach((button, index) =>
    button.addEventListener("click", () => {
      manualSelection = true;
      selectPhase(index);
      if (ready) renderer.setScroll?.([0.12, 0.46, 0.82][index]);
      progressFill.style.transform = `scaleX(${[0.12, 0.46, 0.82][index]})`;
    }),
  );
  reduced.addEventListener("change", queueUpdate);
  selectPhase(0);
  update();

  // A restrained specular highlight follows the pointer across the glass panels.
  if (window.matchMedia("(pointer:fine)").matches) {
    document.querySelectorAll(".glass-panel").forEach((panel) => {
      panel.addEventListener(
        "pointermove",
        (event) => {
          if (paused) return;
          const rect = panel.getBoundingClientRect();
          panel.style.setProperty(
            "--glass-x",
            `${((event.clientX - rect.left) / rect.width) * 100}%`,
          );
          panel.style.setProperty(
            "--glass-y",
            `${((event.clientY - rect.top) / rect.height) * 100}%`,
          );
        },
        { passive: true },
      );
    });
  }
})();
