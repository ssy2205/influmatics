// The same map is embedded on the homepage and the research report.
(() => {
  const frame = document.querySelector('.map-frame');
  if (!frame) return;

  let observer;
  let paused = window.InflumaticsApp?.getState().motionPaused;

  function syncMotion() {
    if (typeof paused === 'boolean' && frame.contentDocument?.body) {
      frame.contentWindow.dispatchEvent(
        new CustomEvent('influmatics-map-motion', { detail: { paused } })
      );
    }
  }

  function fitFrame() {
    const body = frame.contentDocument?.body;
    if (!body?.childElementCount) return;
    const height = Math.ceil(body.getBoundingClientRect().height);
    if (height > 0) frame.style.height = height + 'px';
  }

  function connect() {
    observer?.disconnect();
    const body = frame.contentDocument?.body;
    if (!body?.childElementCount) return;
    if (window.ResizeObserver) {
      observer = new ResizeObserver(fitFrame);
      observer.observe(body);
    }
    fitFrame();
    syncMotion();
  }

  frame.addEventListener('load', connect);
  window.addEventListener('resize', fitFrame, { passive: true });
  window.addEventListener('influmatics-motion', event => {
    paused = event.detail.paused;
    syncMotion();
  });
  connect();
})();
