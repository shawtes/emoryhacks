import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer'

// https://vitejs.dev/config/
export default defineConfig(() => {
  const plugins = [react()]

  if (process.env.ANALYZE) {
    plugins.push(
      visualizer({
        filename: 'dist/bundle-report.html',
        gzipSize: true,
        brotliSize: true,
        open: false
      })
    )
  }

  return {
    plugins,
    server: {
      port: 3000,
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '')
        }
      }
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
      target: 'es2018',
      cssCodeSplit: true,
      reportCompressedSize: true,
      modulePreload: {
        polyfill: false
      },
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes('node_modules')) {
              if (id.includes('react-router-dom')) {
                return 'router'
              }
              if (id.includes('react')) {
                return 'react-core'
              }
              if (id.includes('web-vitals')) {
                return 'web-vitals'
              }
            }
          }
        }
      }
    }
  }
})
