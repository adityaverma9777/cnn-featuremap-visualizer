import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "katex/dist/katex.min.css";
import "./styles.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element #root not found.");
}

const root = createRoot(rootElement);

async function bootstrap() {
  try {
    const [{ default: App }, { default: ErrorBoundary }] = await Promise.all([
      import("./App"),
      import("./ErrorBoundary"),
    ]);

    root.render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("Frontend bootstrap failed:", error);
    rootElement.innerHTML = `
      <main style="padding:16px;font-family:Segoe UI,sans-serif;color:#222;">
        <h1>Frontend Startup Error</h1>
        <p>The app bundle failed before render.</p>
        <p><strong>Details:</strong> ${message}</p>
      </main>
    `;
  }
}

bootstrap();
