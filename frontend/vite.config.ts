import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(() => {
  const env = loadEnv('', process.cwd());
  const devTarget = env.VITE_API_BASE_URL || 'http://localhost:10000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: '0.0.0.0',
      proxy: {
        '/api': {
          target: devTarget,
          changeOrigin: true,
        },
        '/health': {
          target: devTarget,
          changeOrigin: true,
        },
      }
    }
  };
});
