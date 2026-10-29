import { resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(fileURLToPath(new URL('.', import.meta.url)));
const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 8089,
  fetch(request) {
    const url = new URL(request.url);
    const pathParts = url.pathname.split('/').filter(Boolean);
    const mode = pathParts[0] === 'isolated' || pathParts[0] === 'plain' ? pathParts.shift() : 'isolated';
    let relative = decodeURIComponent(pathParts.join('/')) || 'index.html';
    if (relative.includes('\0')) return new Response('Bad path', { status: 400 });
    const filePath = resolve(root, relative);
    if (filePath !== root && !filePath.startsWith(root + sep)) return new Response('Forbidden', { status: 403 });
    const file = Bun.file(filePath);
    if (!file.size) return new Response('Not found', { status: 404 });

    const ext = relative.split('.').pop()?.toLowerCase();
    const contentType = ({
      html: 'text/html; charset=utf-8',
      js: 'text/javascript; charset=utf-8',
      mjs: 'text/javascript; charset=utf-8',
      wasm: 'application/wasm',
      onnx: 'application/octet-stream',
      npy: 'application/octet-stream',
      map: 'application/json; charset=utf-8'
    })[ext] || 'application/octet-stream';
    const headers = new Headers({
      'Content-Type': contentType,
      'Cross-Origin-Resource-Policy': 'same-origin',
      'Cache-Control': 'no-store'
    });
    if (mode === 'isolated') {
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    console.log(`${mode} ${request.method} ${url.pathname} ${file.size} bytes`);
    return new Response(file, { headers });
  }
});

console.log(`Static spike server listening on http://${server.hostname}:${server.port}`);
