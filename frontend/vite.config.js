import { defineConfig } from 'vite';

export default defineConfig({
  base: './', // Ensures correct relative paths after build
  build: {
    outDir: 'dist', // Ensures build files go to the dist folder
  },
});