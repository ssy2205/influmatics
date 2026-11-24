/* Influmatics landing page — no runtime dependencies except the local Three.js bundle. */
(function () {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const config = window.INFLUMATICS_CONFIG || {};
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  let paused = reduced.matches;
  let sceneReady = false;
  let activeScene = window.InflumaticsScene || null;
  let fallbackStarted = false;
  let manualScene = false;
  let activeMode = "structure";
  let scrollPending = false;
  let scrambleTimer = null;

  // All content remains visible if JavaScript is unavailable or interrupted.
  if ("IntersectionObserver" in window) {
    document.body.classList.add("js-ready");
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries)
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
      },
      { threshold: 0.08, rootMargin: "0px 0px 25px 0px" },
    );
    $$(".reveal").forEach((element) => observer.observe(element));
  }

  function updateRange(input) {
    const min = Number(input.min) || 0;
    const max = Number(input.max) || 100;
    const progress = ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--range-progress", `${progress}%`);
  }
  $$('input[type="range"]').forEach((input) => {
    updateRange(input);
    input.addEventListener("input", () => updateRange(input));
  });

  // The decorative scene is isolated: if GPU rendering fails, the SVG stays visible.
  function showFallback() {
    sceneReady = false;
    $("#molecule-stage").classList.remove("is-ready");
    $("#molecule-stage").dataset.renderer = "svg";
    $("#molecule-fallback").removeAttribute("aria-hidden");
    $("#scene-hint").textContent = "CONCEPTUAL MOLECULAR FORM";
    $("#molecule-canvas").setAttribute("aria-hidden", "true");
    $$("#scene-controls button, #scene-controls input").forEach((control) => {
      control.disabled = true;
    });
    $("#scene-controls").setAttribute(
      "aria-label",
      "정적 시각화 모드: 현재 브라우저에서 3D 렌더링을 사용할 수 없습니다.",
    );
  }
  function markSceneReady(info) {
    sceneReady = true;
    $("#molecule-stage").classList.add("is-ready");
    $("#molecule-stage").dataset.renderer = info.renderer;
    $("#scene-controls").removeAttribute("aria-label");
    $("#molecule-fallback").setAttribute("aria-hidden", "true");
    $("#molecule-canvas").setAttribute(
      "aria-hidden",
      String(info.renderer !== "webgl2"),
    );
    $("#scene-hint").textContent = "↔ DRAG TO ROTATE";
    $$("#scene-controls button, #scene-controls input").forEach((control) => {
      control.disabled = false;
    });
  }
  function startFallback() {
    if (fallbackStarted) return;
    fallbackStarted = true;
    if (window.InflumaticsScene) window.InflumaticsScene.destroy();
    activeScene = window.InflumaticsFallbackScene || null;
    if (!activeScene) {
      showFallback();
      return;
    }
    try {
      activeScene.init({
        container: $("#molecule-stage"),
        canvas: $("#molecule-canvas"),
        onReady: markSceneReady,
        onFallback: showFallback,
      });
      activeScene.setMode(activeMode);
      activeScene.setExplode(Number($("#explode").value) / 100);
      activeScene.setPaused(paused);
      activeScene.setQuality(
        ["auto", "high", "low"].includes(config.quality)
          ? config.quality
          : "auto",
      );
    } catch (_) {
      showFallback();
    }
  }
  if (activeScene) {
    try {
      const ready = activeScene.init({
        container: $("#molecule-stage"),
        canvas: $("#molecule-canvas"),
        onReady: markSceneReady,
        onFallback: startFallback,
      });
      if (!ready && !sceneReady) startFallback();
    } catch (_) {
      startFallback();
    }
  } else {
    startFallback();
  }
  if (activeScene)
    activeScene.setQuality(
      ["auto", "high", "low"].includes(config.quality)
        ? config.quality
        : "auto",
    );

  function setMode(mode, manual) {
    if (!["structure", "points", "map"].includes(mode)) return;
    activeMode = mode;
    if (manual) manualScene = true;
    $$("[data-scene-mode]").forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.sceneMode === mode),
      ),
    );
    if (sceneReady) activeScene.setMode(mode);
  }
  $$("[data-scene-mode]").forEach((button) =>
    button.addEventListener("click", () =>
      setMode(button.dataset.sceneMode, true),
    ),
  );
  $("#explode").addEventListener("input", (event) => {
    manualScene = true;
    if (sceneReady) activeScene.setExplode(Number(event.target.value) / 100);
  });
  $("#molecule-stage").addEventListener(
    "pointerdown",
    () => {
      manualScene = true;
    },
    { passive: true },
  );
  $("#reset-scene").addEventListener("click", () => {
    manualScene = false;
    if (sceneReady) activeScene.reset();
    $("#explode").value = "0";
    updateRange($("#explode"));
    setMode("structure", false);
  });
  setMode(
    config.defaultScene || "structure",
    ["points", "map"].includes(config.defaultScene),
  );

  function updateMotion(next) {
    paused = Boolean(next);
    document.body.classList.toggle("motion-paused", paused);
    document.documentElement.classList.toggle("motion-paused", paused);
    window.dispatchEvent(
      new CustomEvent("influmatics-motion", { detail: { paused } }),
    );
    $("#motion-toggle").setAttribute("aria-pressed", String(paused));
    $("#motion-toggle").setAttribute(
      "aria-label",
      paused ? "모션 재생" : "모션 일시정지",
    );
    $("#motion-label").textContent = paused ? "MOTION OFF" : "MOTION ON";
    if (sceneReady) activeScene.setPaused(paused);
    if (paused && scrambleTimer) {
      clearInterval(scrambleTimer);
      scrambleTimer = null;
      $("[data-scramble]").textContent = $("[data-scramble]").dataset.scramble;
    }
  }
  $("#motion-toggle").addEventListener("click", () => updateMotion(!paused));
  reduced.addEventListener("change", (event) => updateMotion(event.matches));
  updateMotion(paused);

  // One short typographic reveal; never an infinite text scramble.
  if (!paused) {
    const heading = $("[data-scramble]");
    const target = heading.dataset.scramble;
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let frame = 0;
    scrambleTimer = setInterval(() => {
      frame += 1;
      heading.textContent = target
        .split("")
        .map((letter, index) =>
          index < frame / 2 || letter === "."
            ? letter
            : letters[(index * 11 + frame * 7) % letters.length],
        )
        .join("");
      if (frame >= target.length * 2) {
        clearInterval(scrambleTimer);
        scrambleTimer = null;
        heading.textContent = target;
      }
    }, 45);
  }

  function updateScroll() {
    scrollPending = false;
    const max = document.documentElement.scrollHeight - window.innerHeight;
    $(".scroll-progress span").style.transform =
      `scaleX(${max > 0 ? window.scrollY / max : 0})`;
    // Desktop scroll tells the same structure → particles → relationships story.
    // Any direct scene interaction takes priority until the user resets the scene.
    if (sceneReady && !manualScene && !paused && window.innerWidth > 960) {
      const progress = window.scrollY / $("#top").offsetHeight;
      const next =
        progress < 0.23 ? "structure" : progress < 0.5 ? "points" : "map";
      if (next !== activeMode) setMode(next, false);
    }
  }
  window.addEventListener(
    "scroll",
    () => {
      if (!scrollPending) {
        scrollPending = true;
        requestAnimationFrame(updateScroll);
      }
    },
    { passive: true },
  );
  updateScroll();

  // The research illustration changes with the selected explanation.
  const svgNS = "http://www.w3.org/2000/svg";
  const pipelinePoints = [];
  for (let i = 0; i < 68; i++) {
    const point = document.createElementNS(svgNS, "circle");
    point.setAttribute("r", String(i % 11 === 0 ? 4 : 1.6 + (i % 3) * 0.5));
    point.setAttribute(
      "fill",
      i % 3 === 0 ? "#c4ff47" : i % 3 === 1 ? "#76dce8" : "#bed0af",
    );
    point.setAttribute("opacity", String(0.3 + (i % 7) * 0.1));
    point.style.transition = "cx .7s ease, cy .7s ease";
    $("#pipeline-points").append(point);
    pipelinePoints.push(point);
  }
  const phaseNames = ["CURATE", "REPRESENT", "UNDERSTAND"];
  const phaseLabels = [
    "비교하기 전에, 데이터부터 확인합니다.",
    "같은 서열을 다른 방법으로 읽어봅니다.",
    "그림에서 발견한 차이를 다시 확인합니다.",
  ];
  const phaseDetails = [
    "저장된 개발 자료에는 고유 바이러스 928개, 평가 자료에는 545개가 있었습니다.",
    "4개년 평균 절대오차는 기준 모델이 가장 낮았습니다. 연도별로 보면 가장 낮은 오차를 보인 모델이 달라집니다.",
    "무작위 분할의 평가 자료에 포함된 바이러스는 모두 학습 자료에 다른 짝으로 등장했습니다. 이 결과는 이미 관찰된 바이러스 간의 거리 복원에 해당합니다.",
  ];
  const steps = $$("[data-step]");
  function setStep(index, moveFocus) {
    steps.forEach((step, i) => {
      step.classList.toggle("is-active", i === index);
      step.setAttribute("aria-selected", String(i === index));
      step.tabIndex = i === index ? 0 : -1;
    });
    $("#step-panel").setAttribute("aria-labelledby", steps[index].id);
    $("#step-panel").textContent = phaseDetails[index];
    $("#pipeline-phase").textContent = phaseNames[index];
    $("#pipeline-visual-label").textContent = phaseLabels[index];
    $$(".pipeline-dots i").forEach((dot, i) =>
      dot.classList.toggle("is-active", i === index),
    );
    pipelinePoints.forEach((point, i) => {
      let x, y;
      if (index === 0) {
        x = 92 + (i % 12) * 37 + (Math.floor(i / 12) % 2) * 5;
        y = 115 + Math.floor(i / 12) * 42;
      } else if (index === 1) {
        const angle = i * 2.399;
        const radius = 76 + (i % 17) * 8;
        x = 300 + Math.cos(angle) * radius;
        y = 230 + Math.sin(angle) * radius * 0.69;
      } else {
        const group = i % 3;
        const angle = i * 2.399;
        const radius = 12 + ((i * 7) % 60);
        x = [170, 395, 350][group] + Math.cos(angle) * radius;
        y = [140, 160, 345][group] + Math.sin(angle) * radius * 0.6;
      }
      point.style.transition = paused ? "none" : "cx .7s ease, cy .7s ease";
      point.setAttribute("cx", x.toFixed(2));
      point.setAttribute("cy", y.toFixed(2));
    });
    if (moveFocus) steps[index].focus();
  }
  steps.forEach((step, index) => {
    step.addEventListener("click", () => setStep(index, false));
    step.addEventListener("keydown", (event) => {
      const keys = [
        "ArrowDown",
        "ArrowRight",
        "ArrowUp",
        "ArrowLeft",
        "Home",
        "End",
      ];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      const next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? steps.length - 1
            : (index +
                (event.key === "ArrowDown" || event.key === "ArrowRight"
                  ? 1
                  : -1) +
                steps.length) %
              steps.length;
      setStep(next, true);
    });
  });
  setStep(0, false);

  // Mobile navigation stays keyboard accessible and closes when a destination is chosen.
  const menuButton = $("#menu-toggle");
  const mobileNav = $("#mobile-nav");
  function closeMenu(returnFocus) {
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.setAttribute("aria-label", "메뉴 열기");
    mobileNav.hidden = true;
    if (returnFocus) menuButton.focus();
  }
  menuButton.addEventListener("click", () => {
    const open = menuButton.getAttribute("aria-expanded") !== "true";
    mobileNav.hidden = !open;
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "메뉴 닫기" : "메뉴 열기");
  });
  mobileNav
    .querySelectorAll("a,button")
    .forEach((item) => item.addEventListener("click", () => closeMenu(false)));
  document.addEventListener("click", (event) => {
    if (
      !mobileNav.hidden &&
      !mobileNav.contains(event.target) &&
      !menuButton.contains(event.target)
    )
      closeMenu(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !mobileNav.hidden) closeMenu(true);
  });

  const dialog = $("#brief-dialog");
  let dialogOpener = null;
  function openBrief(event) {
    dialogOpener = event.currentTarget;
    $("#copy-status").textContent = "";
    closeMenu(false);
    dialog.showModal();
    document.body.classList.add("modal-open");
    $("#close-brief").focus();
  }
  $$(".brief-trigger").forEach((button) =>
    button.addEventListener("click", openBrief),
  );
  $("#credits-trigger").addEventListener("click", openBrief);
  $("#close-brief").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const rect = dialog.getBoundingClientRect();
      if (
        event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom
      )
        dialog.close();
    }
  });
  dialog.addEventListener("close", () => {
    document.body.classList.remove("modal-open");
    if (dialogOpener && dialogOpener.offsetParent !== null)
      dialogOpener.focus();
    else menuButton.focus();
  });
  $("#copy-link").addEventListener("click", async () => {
    const status = $("#copy-status");
    if (!/^https?:$/.test(location.protocol)) {
      status.textContent =
        "웹에 배포하면 공유 주소를 복사할 수 있습니다. 브라우저 주소창에서 경로를 확인하세요.";
      return;
    }
    const url = `${location.origin}${location.pathname}`;
    try {
      if (navigator.clipboard && window.isSecureContext)
        await navigator.clipboard.writeText(url);
      else {
        const textarea = document.createElement("textarea");
        textarea.value = url;
        textarea.setAttribute("aria-label", "공유할 페이지 주소");
        dialog.append(textarea);
        textarea.focus();
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        $("#copy-link").focus();
        if (!copied) throw new Error("Clipboard unavailable");
      }
      status.textContent = "페이지 주소를 복사했습니다.";
    } catch (_) {
      status.textContent = `브라우저 주소창에서 복사해 주세요: ${url}`;
    }
  });
  $("#copyright-year").textContent = String(new Date().getFullYear());

  // One development-friendly handle, containing only public UI state.
  window.InflumaticsApp = Object.freeze({
    getState: () => ({ sceneReady, activeMode, motionPaused: paused }),
    setMotionPaused: updateMotion,
  });
})();
