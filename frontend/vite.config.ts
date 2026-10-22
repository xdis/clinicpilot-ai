import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The UI talks only to the NestJS backend; /api is proxied in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: process.env.BACKEND_ORIGIN ?? 'http://localhost:3000', changeOrigin: true },
    },
  },
})
