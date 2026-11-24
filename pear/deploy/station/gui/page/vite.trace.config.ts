import type {OutputBundle} from 'rollup';
import {defineConfig, type Plugin} from 'vite';
import react from '@vitejs/plugin-react';

/** Puts the built script and stylesheet into the trace template, which the service then sends as one file. */
class InlineAssets {
  static plugin(): Plugin {
    return {name: 'pear-inline-assets', enforce: 'post', generateBundle: (_, bundle) => InlineAssets.inline(bundle)};
  }

  static inline(bundle: OutputBundle) {
    const page = Object.values(bundle).find((file) => file.fileName.endsWith('.html'));
    if (!page || page.type !== 'asset') return;
    let html = String(page.source);
    for (const [name, file] of Object.entries(bundle)) {
      const at = InlineAssets.pattern(file.fileName);
      if (file.type === 'chunk') {
        // no text of the script may close its element early
        const code = file.code.replaceAll('</script', '<\\/script');
        html = html.replace(
          new RegExp(`<script[^>]*src="[^"]*${at}"[^>]*></script>`),
          () => `<script type="module">${code}</script>`,
        );
      } else if (file.fileName.endsWith('.css')) {
        html = html.replace(
          new RegExp(`<link[^>]*href="[^"]*${at}"[^>]*>`),
          () => `<style>${String(file.source)}</style>`,
        );
      } else continue;
      delete bundle[name];
    }
    page.source = html;
  }

  static pattern(fileName: string) {
    return fileName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}

export default defineConfig({
  plugins: [react(), InlineAssets.plugin()],
  build: {
    outDir: '../static',
    emptyOutDir: false, // the console's page is built into the same folder first
    cssCodeSplit: false,
    assetsInlineLimit: Number.MAX_SAFE_INTEGER, // every asset the trace imports goes into the one file
    rollupOptions: {input: 'trace.html', output: {inlineDynamicImports: true}},
  },
});
