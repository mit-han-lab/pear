import {mkdtempSync, readdirSync, readFileSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'vite';

/** Rebuilds the page into a temporary directory and compares it byte for byte with the committed bundle. */
class BundleCheck {
  constructor(root, committed) {
    this.root = root;
    this.committed = committed;
  }

  async run() {
    const rebuilt = mkdtempSync(path.join(tmpdir(), 'pear-station-bundle-'));
    try {
      await build({root: this.root, logLevel: 'warn', build: {outDir: rebuilt, emptyOutDir: true}});
      const differences = this.differences(rebuilt);
      for (const difference of differences) console.error(difference);
      if (differences.length)
        console.error(
          'pear/deploy/station/gui/static is stale: run `npm run build` in pear/deploy/station/gui/page and commit it.',
        );
      return differences.length ? 1 : 0;
    } finally {
      rmSync(rebuilt, {recursive: true, force: true});
    }
  }

  differences(rebuilt) {
    const expected = this.files(this.committed);
    const actual = this.files(rebuilt);
    const names = [...new Set([...expected, ...actual])].sort();
    return names.flatMap((name) => {
      if (!actual.includes(name)) return [`only committed: ${name}`];
      if (!expected.includes(name)) return [`only rebuilt: ${name}`];
      const same = readFileSync(path.join(this.committed, name)).equals(readFileSync(path.join(rebuilt, name)));
      return same ? [] : [`differs: ${name}`];
    });
  }

  files(directory) {
    return readdirSync(directory, {recursive: true, withFileTypes: true})
      .filter((entry) => entry.isFile())
      .map((entry) => path.relative(directory, path.join(entry.parentPath, entry.name)));
  }
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
process.exitCode = await new BundleCheck(root, path.resolve(root, '../static')).run();
