import { resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = resolve(fileURLToPath(new URL('../../', import.meta.url)));
const artifactRoot = resolve(projectRoot, 'artifacts/ghost-unet3d-step-22000');
const artifactParent = resolve(projectRoot, 'artifacts');
const webRoot = resolve(projectRoot, 'web');
const isolated = !Bun.argv.slice(2).includes('--no-isolation');

const routes = [
  { prefix: '/package/', root: artifactRoot, index: 'manifest.json' },
  { prefix: '/artifacts/', root: artifactParent },
  { prefix: '/dist/', root: resolve(webRoot, 'dist') },
  { prefix: '/src/', root: resolve(webRoot, 'src') },
  { prefix: '/pkg/', root: resolve(webRoot, 'pkg') },
  { prefix: '/ort/', root: resolve(webRoot, 'node_modules/onnxruntime-web/dist') }
];

const contentTypes: Record<string, string> = {
  css: 'text/css; charset=utf-8',
  html: 'text/html; charset=utf-8',
  js: 'text/javascript; charset=utf-8',
  json: 'application/json; charset=utf-8',
  map: 'application/json; charset=utf-8',
  mjs: 'text/javascript; charset=utf-8',
  npy: 'application/octet-stream',
  onnx: 'application/octet-stream',
  png: 'image/png',
  svg: 'image/svg+xml',
  wasm: 'application/wasm'
};

function responseHeaders(contentType = 'text/plain; charset=utf-8') {
  const headers = new Headers({
    'Cache-Control': 'no-store',
    'Content-Type': contentType,
    'Cross-Origin-Resource-Policy': 'same-origin'
  });
  if (isolated) {
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  }
  return headers;
}

function errorResponse(message: string, status: number) {
  return new Response(message, { status, headers: responseHeaders() });
}

function contentTypeFor(path: string) {
  const extension = path.split('.').at(-1)?.toLowerCase() ?? '';
  return contentTypes[extension] ?? 'application/octet-stream';
}

const server = Bun.serve({
  hostname: '127.0.0.1',
  port: 8089,
  async fetch(request) {
    const url = new URL(request.url);
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return errorResponse('Method not allowed', 405);
    }

    if (url.pathname === '/package' || url.pathname === '/artifacts') {
      return Response.redirect(new URL(`${url.pathname}/`, url), 308);
    }

    let filePath: string;
    let relativePath: string;
    if (url.pathname === '/') {
      filePath = resolve(webRoot, 'smoke/index.html');
      relativePath = 'index.html';
    } else {
      const route = routes.find(({ prefix }) => url.pathname.startsWith(prefix));
      if (!route) return errorResponse('Not found', 404);

      try {
        relativePath = decodeURIComponent(url.pathname.slice(route.prefix.length));
      } catch {
        return errorResponse('Bad path', 400);
      }
      if (relativePath.includes('\0')) return errorResponse('Bad path', 400);
      if (!relativePath || relativePath.endsWith('/')) {
        relativePath = route.index ?? `${relativePath}index.html`;
      }

      const routeRoot = resolve(route.root);
      filePath = resolve(routeRoot, relativePath);
      if (filePath !== routeRoot && !filePath.startsWith(`${routeRoot}${sep}`)) {
        return errorResponse('Forbidden', 403);
      }
    }

    const file = Bun.file(filePath);
    if (!(await file.exists())) return errorResponse('Not found', 404);

    const headers = responseHeaders(contentTypeFor(relativePath));
    console.log(`${isolated ? 'isolated' : 'plain'} ${request.method} ${url.pathname} ${file.size} bytes`);
    return new Response(request.method === 'HEAD' ? null : file, { headers });
  }
});

console.log(`Preludio smoke server listening on http://${server.hostname}:${server.port} (${isolated ? 'COOP/COEP enabled' : 'isolation disabled'})`);
