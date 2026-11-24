(() => {
  const plot = document.getElementById("antigen-map");
  const series = document.getElementById("map-points");
  const points = Array.from(series.querySelectorAll("circle[data-year]"));
  if (!points.length) return;

  const filters = document.querySelector(".year-filters");
  const buttons = Array.from(filters.querySelectorAll("[data-filter]"));
  const title = document.getElementById("map-detail-title");
  const count = document.getElementById("map-count");
  const detail = document.getElementById("map-detail-body");
  const allDescription = detail.textContent;
  const yearCounts = new Map();
  let visible = points;
  let focusPoint = points[0];
  let filter = "all";

  function setFocusPoint(point) {
    focusPoint.setAttribute("tabindex", "-1");
    focusPoint = point;
    point.setAttribute("tabindex", "0");
  }

  function updateSelection(point) {
    for (const item of points) {
      item.setAttribute("aria-pressed", String(item === point));
      item.classList.toggle("muted-point", Boolean(point && item.dataset.year !== point.dataset.year));
    }
    const year = point?.dataset.year || filter;
    if (year === "all") {
      title.textContent = "전체 표본";
      count.textContent = String(points.length);
      detail.textContent = allDescription;
      return;
    }
    const amount = yearCounts.get(year);
    title.textContent = year + "년 표본";
    count.textContent = String(amount);
    detail.textContent = point && filter === "all"
      ? year + "년 항원주 " + amount + "개를 전체 지도 위에 강조했습니다."
      : year + "년 항원주 " + amount + "개를 전체 지도와 같은 위치에 표시했습니다.";
  }

  function applyFilter(year) {
    filter = year;
    visible = points.filter(point => year === "all" || point.dataset.year === year);
    for (const point of points) {
      point.toggleAttribute("hidden", year !== "all" && point.dataset.year !== year);
    }
    for (const button of buttons) {
      button.setAttribute("aria-pressed", String(button.dataset.filter === year));
    }
    setFocusPoint(visible[0]);
    updateSelection(null);
  }

  function choose(point) {
    setFocusPoint(point);
    point.focus({ preventScroll: true });
    updateSelection(point);
  }

  for (const point of points) {
    const year = point.dataset.year;
    yearCounts.set(year, (yearCounts.get(year) || 0) + 1);
    point.setAttribute("role", "button");
    point.setAttribute("aria-label", year + "년 항원주, 같은 연도 강조");
    point.setAttribute("aria-pressed", "false");
    point.setAttribute("tabindex", "-1");
  }
  setFocusPoint(focusPoint);
  plot.setAttribute("role", "group");
  filters.hidden = false;
  document.querySelector(".point-help").hidden = false;

  for (const button of buttons) {
    button.addEventListener("click", () => applyFilter(button.dataset.filter));
  }

  // Small marks share a forgiving tap area. Resolve a tap in screen pixels so
  // dense regions and narrow phone screens keep the original plotted geometry.
  plot.addEventListener("click", event => {
    if (event.target.matches("circle[data-year]")) {
      choose(event.target);
      return;
    }
    const matrix = series.getScreenCTM();
    if (!matrix) return;
    let hit;
    let hitRadius = 13;
    for (const point of visible) {
      const center = new DOMPoint(point.cx.baseVal.value, point.cy.baseVal.value).matrixTransform(matrix);
      const distance = Math.hypot(center.x - event.clientX, center.y - event.clientY);
      if (distance < hitRadius) {
        hit = point;
        hitRadius = distance;
      }
    }
    if (hit) choose(hit);
  });

  series.addEventListener("keydown", event => {
    const index = visible.indexOf(event.target);
    if (index === -1) return;
    let next = index;
    switch (event.key) {
      case "ArrowRight": case "ArrowDown": next = (index + 1) % visible.length; break;
      case "ArrowLeft": case "ArrowUp": next = (index - 1 + visible.length) % visible.length; break;
      case "Home": next = 0; break;
      case "End": next = visible.length - 1; break;
      case "Enter": case " ":
        event.preventDefault();
        choose(event.target);
        return;
      case "Escape":
        event.preventDefault();
        applyFilter("all");
        focusPoint.focus({ preventScroll: true });
        return;
      default: return;
    }
    event.preventDefault();
    setFocusPoint(visible[next]);
    focusPoint.focus({ preventScroll: true });
  });

  window.addEventListener("influmatics-map-motion", event => {
    document.documentElement.classList.toggle("motion-paused", event.detail?.paused === true);
  });
})();
