from __future__ import annotations

import json
from pathlib import Path

from .options import RuntimeBrowserOptions, RuntimeBrowserRenderOptions


def _runtime_browser_render_html(asset_path: Path, options: RuntimeBrowserRenderOptions) -> str:
    asset_url = json.dumps(asset_path.as_uri())
    background = json.dumps([round(channel / 255.0, 6) for channel in options.background_color])
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>fascat browser render preview</title></head>
<body style="margin:0;overflow:hidden;background:transparent">
<canvas id="canvas" width="{options.width}" height="{options.height}"></canvas>
<pre id="result" style="display:none">{{"status":"running"}}</pre>
<script>
const ASSET_URL = {asset_url};
const BACKGROUND = {background};
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

function finish(payload) {{
  result.textContent = JSON.stringify(payload);
  document.title = "fascat-browser-render-done";
}}

function bytesToString(bytes) {{
  return new TextDecoder("utf-8").decode(bytes);
}}

function readUint32(bytes, offset) {{
  return new DataView(bytes.buffer, bytes.byteOffset + offset, 4).getUint32(0, true);
}}

function dataUriToBuffer(uri) {{
  const comma = uri.indexOf(",");
  if (comma < 0) throw new Error("invalid data URI buffer");
  const header = uri.slice(0, comma);
  const payload = uri.slice(comma + 1);
  if (header.includes(";base64")) {{
    const binary = atob(payload);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }}
  return new TextEncoder().encode(decodeURIComponent(payload)).buffer;
}}

function loadBufferUri(uri) {{
  const xhr = new XMLHttpRequest();
  xhr.open("GET", uri, false);
  xhr.overrideMimeType("text/plain; charset=x-user-defined");
  xhr.send(null);
  if (xhr.status !== 0 && (xhr.status < 200 || xhr.status >= 300)) throw new Error("failed to load " + uri);
  if (xhr.responseText === undefined) throw new Error("empty response for " + uri);
  const text = xhr.responseText;
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i) & 0xff;
  return bytes.buffer;
}}

function loadAsset() {{
  const buffer = loadBufferUri(ASSET_URL);
  const bytes = new Uint8Array(buffer);
  if (bytes.length >= 20 && bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46) {{
    const jsonLength = readUint32(bytes, 12);
    const jsonType = readUint32(bytes, 16);
    if (jsonType !== 0x4e4f534a) throw new Error("first GLB chunk is not JSON");
    const jsonStart = 20;
    const jsonEnd = jsonStart + jsonLength;
    const document = JSON.parse(bytesToString(bytes.slice(jsonStart, jsonEnd)));
    const buffers = [];
    let chunkOffset = jsonEnd + (jsonLength % 4 === 0 ? 0 : 4 - (jsonLength % 4));
    if (bytes.length >= chunkOffset + 8) {{
      const binaryLength = readUint32(bytes, chunkOffset);
      const binaryType = readUint32(bytes, chunkOffset + 4);
      if (binaryType === 0x004e4942) buffers[0] = bytes.slice(chunkOffset + 8, chunkOffset + 8 + binaryLength).buffer;
    }}
    return {{ document, buffers }};
  }}
  const document = JSON.parse(bytesToString(bytes));
  const buffers = [];
  const bufferDefs = Array.isArray(document.buffers) ? document.buffers : [];
  for (let i = 0; i < bufferDefs.length; i++) {{
    const uri = bufferDefs[i].uri;
    if (!uri) throw new Error("external glTF buffer is missing a URI");
    if (uri.startsWith("data:")) {{
      buffers[i] = dataUriToBuffer(uri);
    }} else {{
      buffers[i] = loadBufferUri(new URL(uri, ASSET_URL).href);
    }}
  }}
  return {{ document, buffers }};
}}

const COMPONENTS = {{ SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }};
const ARRAY_TYPES = {{ 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }};
const COMPONENT_BYTES = {{ 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }};
const VERTEX_ATTRIBUTE_COMPONENT_TYPES = new Set([5120, 5121, 5122, 5123, 5126]);

function readAccessorComponent(view, byteOffset, componentType) {{
  if (componentType === 5120) return view.getInt8(byteOffset);
  if (componentType === 5121) return view.getUint8(byteOffset);
  if (componentType === 5122) return view.getInt16(byteOffset, true);
  if (componentType === 5123) return view.getUint16(byteOffset, true);
  if (componentType === 5125) return view.getUint32(byteOffset, true);
  if (componentType === 5126) return view.getFloat32(byteOffset, true);
  throw new Error("unsupported accessor component type " + componentType);
}}

function normalizedComponentValue(value, componentType) {{
  if (componentType === 5120) return Math.max(value / 127.0, -1.0);
  if (componentType === 5121) return value / 255.0;
  if (componentType === 5122) return Math.max(value / 32767.0, -1.0);
  if (componentType === 5123) return value / 65535.0;
  return value;
}}

function accessorComponentValue(accessor, row, column) {{
  const value = accessor.array[row * accessor.itemSize + column];
  return accessor.normalized ? normalizedComponentValue(value, accessor.componentType) : value;
}}

function readAccessor(document, buffers, accessorIndex) {{
  const accessor = document.accessors && document.accessors[accessorIndex];
  if (!accessor) throw new Error("missing accessor " + accessorIndex);
  if (accessor.sparse) throw new Error("sparse accessors are not supported by browser render preview");
  const view = document.bufferViews && document.bufferViews[accessor.bufferView];
  if (!view) throw new Error("missing bufferView for accessor " + accessorIndex);
  const buffer = buffers[view.buffer || 0];
  if (!buffer) throw new Error("missing buffer data for accessor " + accessorIndex);
  const itemSize = COMPONENTS[accessor.type] || 1;
  const ArrayType = ARRAY_TYPES[accessor.componentType];
  const componentBytes = COMPONENT_BYTES[accessor.componentType];
  if (!ArrayType || !componentBytes) throw new Error("unsupported accessor component type " + accessor.componentType);
  const byteOffset = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || (componentBytes * itemSize);
  const length = accessor.count * itemSize;
  if (stride === componentBytes * itemSize) {{
    return {{ array: new ArrayType(buffer, byteOffset, length), itemSize, count: accessor.count, componentType: accessor.componentType, normalized: !!accessor.normalized }};
  }}
  const sourceLength = accessor.count === 0 ? 0 : (accessor.count - 1) * stride + componentBytes * itemSize;
  const source = new DataView(buffer, byteOffset, sourceLength);
  const values = new ArrayType(length);
  for (let row = 0; row < accessor.count; row++) {{
    for (let column = 0; column < itemSize; column++) {{
      values[row * itemSize + column] = readAccessorComponent(source, row * stride + column * componentBytes, accessor.componentType);
    }}
  }}
  return {{ array: values, itemSize, count: accessor.count, componentType: accessor.componentType, normalized: !!accessor.normalized }};
}}

function identity() {{
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}}

function multiply(a, b) {{
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column++) {{
    for (let row = 0; row < 4; row++) {{
      out[column * 4 + row] =
        a[0 * 4 + row] * b[column * 4 + 0] +
        a[1 * 4 + row] * b[column * 4 + 1] +
        a[2 * 4 + row] * b[column * 4 + 2] +
        a[3 * 4 + row] * b[column * 4 + 3];
    }}
  }}
  return out;
}}

function nodeMatrix(node) {{
  if (Array.isArray(node.matrix) && node.matrix.length === 16) return node.matrix.slice();
  const t = node.translation || [0, 0, 0];
  const r = node.rotation || [0, 0, 0, 1];
  const s = node.scale || [1, 1, 1];
  const x = r[0], y = r[1], z = r[2], w = r[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1
  ];
}}

function transformPoint(m, p) {{
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]
  ];
}}

function subtract(a, b) {{ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }}
function cross(a, b) {{ return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }}
function dot(a, b) {{ return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }}
function normalize(v) {{
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}}

function lookAt(eye, center, up) {{
  const z = normalize(subtract(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ];
}}

function perspective(fovy, aspect, near, far) {{
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}}

function shader(gl, type, source) {{
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}}

function program(gl) {{
  const vertexSource = [
    "attribute vec3 p;",
    "attribute vec2 uv;",
    "uniform mat4 mvp;",
    "varying vec2 vUv;",
    "void main() {{ vUv = uv; gl_Position = mvp * vec4(p, 1.0); }}"
  ].join("\\n");
  const fragmentSource = [
    "precision mediump float;",
    "uniform vec4 color;",
    "uniform sampler2D baseColorTexture;",
    "uniform bool useTexture;",
    "varying vec2 vUv;",
    "void main() {{ vec4 texel = useTexture ? texture2D(baseColorTexture, vUv) : vec4(1.0); gl_FragColor = color * texel; }}"
  ].join("\\n");
  const item = gl.createProgram();
  gl.attachShader(item, shader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(item, shader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(item);
  if (!gl.getProgramParameter(item, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(item));
  return item;
}}

function materialColor(document, materialIndex) {{
  const material = document.materials && document.materials[materialIndex];
  const factor = material && material.pbrMetallicRoughness && material.pbrMetallicRoughness.baseColorFactor;
  return Array.isArray(factor) ? factor : [0.45, 0.58, 0.72, 1.0];
}}

function materialBaseColorImageIndex(document, materialIndex) {{
  const material = document.materials && document.materials[materialIndex];
  const textureInfo = material && material.pbrMetallicRoughness && material.pbrMetallicRoughness.baseColorTexture;
  if (!textureInfo || textureInfo.index === undefined) return null;
  if (textureInfo.texCoord !== undefined && textureInfo.texCoord !== 0) return null;
  const texture = document.textures && document.textures[textureInfo.index];
  if (!texture || texture.source === undefined) return null;
  return texture.source;
}}

function imageUri(document, imageIndex) {{
  const image = document.images && document.images[imageIndex];
  if (!image || !image.uri) return null;
  if (image.uri.startsWith("data:")) return image.uri;
  return new URL(image.uri, ASSET_URL).href;
}}

async function loadImage(uri) {{
  if (typeof fetch === "function" && typeof createImageBitmap === "function") {{
    try {{
      const response = await fetch(uri);
      if (response.ok || uri.startsWith("data:") || uri.startsWith("file:")) {{
        return await createImageBitmap(await response.blob());
      }}
    }} catch (_error) {{
      // Fall back to Image for file URLs or older browser builds.
    }}
  }}
  return await new Promise((resolve, reject) => {{
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("failed to load texture image " + uri));
    image.src = uri;
  }});
}}

function createGlTexture(gl, image) {{
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  const error = gl.getError();
  if (error !== gl.NO_ERROR) throw new Error("failed to upload texture image: WebGL error " + error);
  return texture;
}}

async function textureForMaterial(gl, document, materialIndex, cache) {{
  const imageIndex = materialBaseColorImageIndex(document, materialIndex);
  if (imageIndex === null) return null;
  if (cache.has(imageIndex)) return cache.get(imageIndex);
  const uri = imageUri(document, imageIndex);
  if (!uri) return null;
  const image = await loadImage(uri);
  const texture = createGlTexture(gl, image);
  cache.set(imageIndex, texture);
  return texture;
}}

function collectDraws(document, buffers) {{
  const draws = [];
  const nodes = Array.isArray(document.nodes) ? document.nodes : [];
  const scenes = Array.isArray(document.scenes) ? document.scenes : [];
  const scene = scenes[document.scene || 0] || scenes[0] || {{ nodes: nodes.map((_, index) => index) }};

  function addMesh(meshIndex, world) {{
    const mesh = document.meshes && document.meshes[meshIndex];
    if (!mesh || !Array.isArray(mesh.primitives)) return;
    for (const primitive of mesh.primitives) {{
      if ((primitive.mode === undefined ? 4 : primitive.mode) !== 4) continue;
      if (!primitive.attributes || primitive.attributes.POSITION === undefined) continue;
      const position = readAccessor(document, buffers, primitive.attributes.POSITION);
      if (!VERTEX_ATTRIBUTE_COMPONENT_TYPES.has(position.componentType) || position.itemSize !== 3) {{
        throw new Error("browser render preview currently supports FLOAT or quantized VEC3 positions");
      }}
      const texcoord = primitive.attributes.TEXCOORD_0 === undefined
        ? null
        : readAccessor(document, buffers, primitive.attributes.TEXCOORD_0);
      if (texcoord && (!VERTEX_ATTRIBUTE_COMPONENT_TYPES.has(texcoord.componentType) || texcoord.itemSize !== 2)) {{
        throw new Error("browser render preview currently supports FLOAT or quantized VEC2 TEXCOORD_0");
      }}
      const indices = primitive.indices === undefined ? null : readAccessor(document, buffers, primitive.indices);
      const triangles = indices ? Math.floor(indices.count / 3) : Math.floor(position.count / 3);
      draws.push({{
        position,
        indices,
        texcoord,
        material: primitive.material,
        texture: null,
        matrix: world,
        color: materialColor(document, primitive.material),
        quantized: position.componentType !== 5126 || position.normalized || (texcoord && (texcoord.componentType !== 5126 || texcoord.normalized)),
        triangles
      }});
    }}
  }}

  function walk(nodeIndex, parent) {{
    const node = nodes[nodeIndex];
    if (!node) return;
    const world = multiply(parent, nodeMatrix(node));
    if (node.mesh !== undefined) addMesh(node.mesh, world);
    if (Array.isArray(node.children)) for (const child of node.children) walk(child, world);
  }}

  const roots = Array.isArray(scene.nodes) ? scene.nodes : [];
  for (const root of roots) walk(root, identity());
  if (!roots.length && Array.isArray(document.meshes)) {{
    for (let index = 0; index < document.meshes.length; index++) addMesh(index, identity());
  }}
  return draws;
}}

function bounds(draws) {{
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const draw of draws) {{
    const values = draw.position.array;
    for (let i = 0; i < draw.position.count; i++) {{
      const p = transformPoint(draw.matrix, [
        accessorComponentValue(draw.position, i, 0),
        accessorComponentValue(draw.position, i, 1),
        accessorComponentValue(draw.position, i, 2)
      ]);
      for (let axis = 0; axis < 3; axis++) {{
        min[axis] = Math.min(min[axis], p[axis]);
        max[axis] = Math.max(max[axis], p[axis]);
      }}
    }}
  }}
  return {{ min, max }};
}}

(async () => {{
  try {{
    const loaded = loadAsset();
    const document = loaded.document;
    const draws = collectDraws(document, loaded.buffers);
    if (!draws.length) throw new Error("asset contains no renderable mesh primitives");
    const contextOptions = {{ preserveDrawingBuffer: true }};
    const gl = canvas.getContext("webgl2", contextOptions) || canvas.getContext("webgl", contextOptions);
    if (!gl) throw new Error("WebGL context unavailable");
    if (draws.some(draw => draw.indices && draw.indices.componentType === 5125) && !gl.getExtension("OES_element_index_uint")) {{
      throw new Error("browser does not support unsigned-int index buffers");
    }}

    const box = bounds(draws);
    const center = [
      (box.min[0] + box.max[0]) * 0.5,
      (box.min[1] + box.max[1]) * 0.5,
      (box.min[2] + box.max[2]) * 0.5
    ];
    const extent = Math.max(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2], 1e-6);
    const eye = [center[0] + extent * 1.3, center[1] + extent * 0.9, center[2] + extent * 1.3];
    const view = lookAt(eye, center, [0, 1, 0]);
    const projection = perspective(Math.PI / 4, canvas.width / canvas.height, extent * 0.01, extent * 10.0);
    const drawProgram = program(gl);
    const positionLocation = gl.getAttribLocation(drawProgram, "p");
    const texcoordLocation = gl.getAttribLocation(drawProgram, "uv");
    const mvpLocation = gl.getUniformLocation(drawProgram, "mvp");
    const colorLocation = gl.getUniformLocation(drawProgram, "color");
    const textureLocation = gl.getUniformLocation(drawProgram, "baseColorTexture");
    const useTextureLocation = gl.getUniformLocation(drawProgram, "useTexture");
    const textureCache = new Map();
    let texturedPrimitives = 0;
    let quantizedPrimitives = 0;
    for (const draw of draws) {{
      draw.texture = draw.texcoord ? await textureForMaterial(gl, document, draw.material, textureCache) : null;
      if (draw.texture) texturedPrimitives += 1;
      if (draw.quantized) quantizedPrimitives += 1;
    }}

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], BACKGROUND[3]);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(drawProgram);

    let triangles = 0;
    for (const draw of draws) {{
      const positionBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, draw.position.array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 3, draw.position.componentType, draw.position.normalized, 0, 0);
      if (texcoordLocation >= 0) {{
        if (draw.texcoord && draw.texture) {{
          const texcoordBuffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
          gl.bufferData(gl.ARRAY_BUFFER, draw.texcoord.array, gl.STATIC_DRAW);
          gl.enableVertexAttribArray(texcoordLocation);
          gl.vertexAttribPointer(texcoordLocation, 2, draw.texcoord.componentType, draw.texcoord.normalized, 0, 0);
        }} else {{
          gl.disableVertexAttribArray(texcoordLocation);
          gl.vertexAttrib2f(texcoordLocation, 0.0, 0.0);
        }}
      }}
      gl.uniformMatrix4fv(mvpLocation, false, new Float32Array(multiply(projection, multiply(view, draw.matrix))));
      gl.uniform4fv(colorLocation, new Float32Array(draw.color));
      gl.uniform1i(useTextureLocation, draw.texture ? 1 : 0);
      if (draw.texture) {{
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, draw.texture);
        gl.uniform1i(textureLocation, 0);
      }}
      if (draw.indices) {{
        const indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, draw.indices.array, gl.STATIC_DRAW);
        gl.drawElements(gl.TRIANGLES, draw.indices.count, draw.indices.componentType, 0);
      }} else {{
        gl.drawArrays(gl.TRIANGLES, 0, draw.position.count);
      }}
      const drawError = gl.getError();
      if (drawError !== gl.NO_ERROR) throw new Error("failed to draw primitive: WebGL error " + drawError);
      triangles += draw.triangles;
    }}
    gl.finish();
    finish({{ status: "rendered", meshes: Array.isArray(document.meshes) ? document.meshes.length : 0, triangles, textured_primitives: texturedPrimitives, sampled_textures: textureCache.size, quantized_primitives: quantizedPrimitives, screenshot_data: canvas.toDataURL("image/png") }});
  }} catch (error) {{
    finish({{ status: "failed", error: String(error), meshes: 0, triangles: 0, textured_primitives: 0, sampled_textures: 0, quantized_primitives: 0 }});
  }}
}})();
</script>
</body>
</html>
"""


def _runtime_harness_html(asset_path: Path, options: RuntimeBrowserOptions) -> str:
    asset_url = json.dumps(asset_path.as_uri())
    warmup_ms = int(options.warmup_seconds * 1000.0)
    duration_ms = int(options.duration_seconds * 1000.0)
    max_workload_triangles = int(options.max_workload_triangles)
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>fascat runtime browser harness</title></head>
<body>
<canvas id="canvas" width="{options.width}" height="{options.height}"></canvas>
<pre id="result">{{"status":"running"}}</pre>
<script>
const ASSET_URL = {asset_url};
const WARMUP_MS = {warmup_ms};
const DURATION_MS = {duration_ms};
const MAX_WORKLOAD_TRIANGLES = {max_workload_triangles};
const result = document.getElementById("result");
const canvas = document.getElementById("canvas");

function finish(payload) {{
  result.textContent = JSON.stringify(payload);
  document.title = "fascat-runtime-done";
}}

function countTriangles(document) {{
  let triangles = 0;
  const meshes = Array.isArray(document.meshes) ? document.meshes : [];
  const accessors = Array.isArray(document.accessors) ? document.accessors : [];
  for (const mesh of meshes) {{
    const primitives = Array.isArray(mesh.primitives) ? mesh.primitives : [];
    for (const primitive of primitives) {{
      const mode = primitive.mode === undefined ? 4 : primitive.mode;
      if (mode !== 4) continue;
      if (primitive.indices !== undefined && accessors[primitive.indices]) {{
        triangles += Math.floor((accessors[primitive.indices].count || 0) / 3);
      }} else if (primitive.attributes && primitive.attributes.POSITION !== undefined) {{
        const accessor = accessors[primitive.attributes.POSITION];
        if (accessor) triangles += Math.floor((accessor.count || 0) / 3);
      }}
    }}
  }}
  return {{ meshes: meshes.length, triangles }};
}}

function parseGltf(buffer, url) {{
  const bytes = new Uint8Array(buffer);
  if (bytes.length >= 20 && bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46) {{
    const view = new DataView(buffer);
    const jsonLength = view.getUint32(12, true);
    const jsonType = view.getUint32(16, true);
    if (jsonType !== 0x4e4f534a) throw new Error("first GLB chunk is not JSON");
    const jsonBytes = bytes.slice(20, 20 + jsonLength);
    return JSON.parse(new TextDecoder("utf-8").decode(jsonBytes));
  }}
  return JSON.parse(new TextDecoder("utf-8").decode(bytes));
}}

function shader(gl, type, source) {{
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}}

function program(gl) {{
  const item = gl.createProgram();
  gl.attachShader(item, shader(gl, gl.VERTEX_SHADER, "attribute vec2 p; void main() {{ gl_Position = vec4(p, 0.0, 1.0); }}"));
  gl.attachShader(item, shader(gl, gl.FRAGMENT_SHADER, "precision mediump float; void main() {{ gl_FragColor = vec4(0.25, 0.55, 0.9, 1.0); }}"));
  gl.linkProgram(item);
  if (!gl.getProgramParameter(item, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(item));
  return item;
}}

function workloadVertices(triangles) {{
  const vertexCount = Math.max(3, triangles * 3);
  const data = new Float32Array(vertexCount * 2);
  for (let i = 0; i < triangles; i++) {{
    const x = ((i % 256) / 128.0) - 1.0;
    const y = ((Math.floor(i / 256) % 256) / 128.0) - 1.0;
    const base = i * 6;
    data[base] = x; data[base + 1] = y;
    data[base + 2] = x + 0.006; data[base + 3] = y;
    data[base + 4] = x; data[base + 5] = y + 0.006;
  }}
  return data;
}}

(async () => {{
  try {{
    const loadStart = performance.now();
    const response = await fetch(ASSET_URL);
    const buffer = await response.arrayBuffer();
    const loadTimeMs = performance.now() - loadStart;
    const gltf = parseGltf(buffer, ASSET_URL);
    const counts = countTriangles(gltf);
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL context unavailable");
    const drawTriangles = Math.max(1, Math.min(counts.triangles || 1, MAX_WORKLOAD_TRIANGLES));
    const vertices = workloadVertices(drawTriangles);
    const bufferObject = gl.createBuffer();
    const drawProgram = program(gl);
    const location = gl.getAttribLocation(drawProgram, "p");
    gl.bindBuffer(gl.ARRAY_BUFFER, bufferObject);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.useProgram(drawProgram);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    let frames = 0;
    let measureStart = null;
    function frame(now) {{
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, drawTriangles * 3);
      if (measureStart === null && now >= WARMUP_MS) {{
        measureStart = now;
        frames = 0;
      }}
      if (measureStart !== null) {{
        frames += 1;
        const elapsed = now - measureStart;
        if (elapsed >= DURATION_MS) {{
          finish({{
            status: "measured",
            load_time_ms: Math.round(loadTimeMs),
            measured_fps: frames * 1000.0 / Math.max(1.0, elapsed),
            frame_count: frames,
            measurement_duration_ms: Math.round(elapsed),
            memory_bytes: performance.memory ? performance.memory.usedJSHeapSize : buffer.byteLength + vertices.byteLength,
            meshes: counts.meshes,
            triangles: counts.triangles,
            workload_triangles: drawTriangles
          }});
          return;
        }}
      }}
      requestAnimationFrame(frame);
    }}
    requestAnimationFrame(frame);
  }} catch (error) {{
    finish({{ status: "failed", error: String(error), meshes: 0, triangles: 0, workload_triangles: 0 }});
  }}
}})();
</script>
</body>
</html>
"""
