(() => {
  const explorer = window.InflumaticsExplorer?.init({
    svg: document.getElementById("antigen-map"),
    tooltip: document.getElementById("map-tooltip"),
    detailTitle: document.getElementById("map-detail-title"),
    detailBody: document.getElementById("map-detail-body"),
    yearInput: document.getElementById("map-year"),
    yearOutput: document.getElementById("map-year-output"),
    resetButton: document.getElementById("reset-map"),
    viewButtons: document.querySelectorAll("[data-map-view]"),
  });
  if (!explorer) return;

  for (const control of document.querySelectorAll(".map-views, .timeline-control, .inspector-note, #reset-map")) {
    control.hidden = false;
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let hostPaused = false;
  const updateMotion = () => explorer.setPaused(hostPaused || reducedMotion.matches);
  reducedMotion.addEventListener("change", updateMotion);
  window.addEventListener("influmatics-map-motion", (event) => {
    hostPaused = event.detail?.paused === true;
    updateMotion();
  });
  updateMotion();
})();
