import { defineConfig } from "vite";

// Vite's SPA fallback otherwise serves the dashboard for public directory URLs.
function researchRoutes(server) {
  server.middlewares.use((request, response, next) => {
    const url = new URL(request.url, "http://localhost");
    if (["/", "/research", "/research/results"].includes(url.pathname)) {
      const destination = url.pathname === "/" ? "/research/" : url.pathname + "/";
      response.writeHead(302, { Location: destination + url.search });
      response.end();
      return;
    }
    if (["/research/", "/research/results/"].includes(url.pathname)) {
      request.url = url.pathname + "index.html" + url.search;
    }
    next();
  });
}

export default defineConfig({
  plugins: [{
    name: "research-pages",
    configureServer: researchRoutes,
    configurePreviewServer: researchRoutes,
  }],
});
