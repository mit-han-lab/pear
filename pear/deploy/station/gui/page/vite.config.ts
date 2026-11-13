import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

const DEFAULT_PORT = 18800; // of the GUI service that `npm run dev` proxies to
const service = `http://127.0.0.1:${process.env.PEAR_STATION_PORT ?? DEFAULT_PORT}`;

export default defineConfig({
  plugins: [react()],
  build: {outDir: '../static', emptyOutDir: true},
  server: {proxy: {'/api': service, '/health': service}},
});
