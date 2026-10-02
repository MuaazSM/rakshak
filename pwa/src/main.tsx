import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { registerSW } from "virtual:pwa-register";
import { App } from "./App";
import { initInstallCapture } from "./lib/install";
import { captureParentFromUrl } from "./lib/storage";
import "./styles/tokens.css";
import "./styles/app.css";

captureParentFromUrl(); // FR-7: remember ?parent=mom
initInstallCapture();
registerSW({ immediate: true });

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
