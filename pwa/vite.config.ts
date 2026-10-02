import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

// Same-origin API paths (PRD §5.3). The hub serves pwa/dist in production;
// in dev, proxy everything the hub owns to it.
const HUB = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg", "favicon-32.png", "favicon-16.png", "apple-touch-icon-180.png"],
      manifest: {
        name: "Rakshak",
        short_name: "Rakshak",
        description: "Send us a message. We'll tell you if it's real or a scam.",
        theme_color: "#FBF7F0",
        background_color: "#FBF7F0",
        display: "standalone",
        orientation: "portrait",
        start_url: "/",
        scope: "/",
        icons: [
          { src: "/pwa-icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "/pwa-icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "/pwa-icon-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
        // FR-1 / FR-7: Web Share Target. The hub handles POST /share and 303s to /v/{id}.
        share_target: {
          action: "/share?parent=mom",
          method: "POST",
          enctype: "multipart/form-data",
          params: {
            title: "title",
            text: "text",
            url: "url",
            files: [{ name: "image", accept: ["image/*"] }],
          },
        },
      } as never, // share_target is not in the plugin's manifest typing
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png,ico,woff2}"],
        navigateFallback: "/index.html",
        // Never serve the app shell for hub-owned paths.
        navigateFallbackDenylist: [/^\/api\//, /^\/voice/, /^\/share/, /^\/health/],
      },
    }),
  ],
  server: {
    proxy: {
      "/api": HUB,
      "/voice": HUB,
      "/share": HUB,
      "/health": HUB,
    },
  },
});
