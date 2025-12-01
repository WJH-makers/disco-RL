import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Basic Vite config. For proxying backend API, set VITE_API_PROXY in .env.local
export default defineConfig(({ mode }) => {
  const proxy = process.env.VITE_API_PROXY;
  return {
    plugins: [react()],
    server: proxy
      ? {
          proxy: {
            "/infer": {
              target: proxy,
              changeOrigin: true
            }
          }
        }
      : undefined
  };
});
