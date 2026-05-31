using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using GLTFast;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

public static class FascatRuntimeHarness
{
    const int PreviewWidth = 800;
    const int PreviewHeight = 600;
    const int PreviewBenchmarkFrames = 30;

    public static void Run()
    {
        string assetPath = Arg("-fascatAsset");
        string reportPath = Arg("-fascatReport");
        string previewPath = Arg("-fascatPreview");
        var stopwatch = Stopwatch.StartNew();

        string status = "measured";
        string error = "";
        RenderResult renderResult = RenderResult.NotRequested();
        long memoryBytes = 0;
        int meshes = 0;
        int triangles = 0;
        int frameCount = 0;

        if (string.IsNullOrEmpty(assetPath) || !File.Exists(assetPath))
        {
            status = "failed";
            error = "missing -fascatAsset input";
            renderResult = string.IsNullOrEmpty(previewPath) ? RenderResult.NotRequested() : RenderResult.Failed(error);
        }
        else
        {
            try
            {
                AssetCounts counts = LoadAssetCounts(assetPath);
                meshes = counts.Meshes;
                triangles = counts.Triangles;
                memoryBytes = counts.MemoryBytes;
                renderResult = RenderPreview(assetPath, previewPath);
                frameCount = renderResult.RenderedFrames;
                memoryBytes += GC.GetTotalMemory(false);
            }
            catch (Exception exception)
            {
                status = "failed";
                error = exception.Message;
                renderResult = string.IsNullOrEmpty(previewPath) ? RenderResult.NotRequested() : RenderResult.Failed(error);
            }
        }

        stopwatch.Stop();
        long measurementDurationMs = renderResult.BenchmarkTimeMs >= 0
            ? renderResult.BenchmarkTimeMs
            : stopwatch.ElapsedMilliseconds;
        JObject payload = new JObject
        {
            ["status"] = status,
            ["engine_version"] = Application.unityVersion,
            ["load_time_ms"] = stopwatch.ElapsedMilliseconds,
            ["measured_fps"] = renderResult.MeasuredFps >= 0.0 ? new JValue(renderResult.MeasuredFps) : JValue.CreateNull(),
            ["frame_count"] = frameCount,
            ["measurement_duration_ms"] = measurementDurationMs,
            ["memory_bytes"] = memoryBytes,
            ["meshes"] = meshes,
            ["triangles"] = triangles,
            ["preview_path"] = string.IsNullOrEmpty(previewPath) ? JValue.CreateNull() : new JValue(previewPath),
            ["render_status"] = renderResult.Status,
            ["render_time_ms"] = renderResult.RenderTimeMs >= 0 ? new JValue(renderResult.RenderTimeMs) : JValue.CreateNull(),
            ["rendered_frames"] = renderResult.RenderedFrames,
            ["render_error"] = renderResult.Error,
            ["error"] = error
        };
        string json = payload.ToString(Formatting.None);

        if (!string.IsNullOrEmpty(reportPath))
        {
            string reportDirectory = Path.GetDirectoryName(reportPath);
            if (!string.IsNullOrEmpty(reportDirectory))
            {
                Directory.CreateDirectory(reportDirectory);
            }
            File.WriteAllText(reportPath, json);
        }
        else
        {
            Console.WriteLine(json);
        }
    }

    static string Arg(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == name)
            {
                return args[i + 1];
            }
        }
        return "";
    }

    static RenderResult RenderPreview(string assetPath, string previewPath)
    {
        if (string.IsNullOrEmpty(previewPath))
        {
            return RenderResult.NotRequested();
        }

        var stopwatch = Stopwatch.StartNew();
        GameObject root = null;
        Camera camera = null;
        Light light = null;
        RenderTexture renderTexture = null;
        Texture2D image = null;
        GltfImport gltf = null;
        RenderTexture previousTarget = RenderTexture.active;

        try
        {
            string previewDirectory = Path.GetDirectoryName(previewPath);
            if (!string.IsNullOrEmpty(previewDirectory))
            {
                Directory.CreateDirectory(previewDirectory);
            }

            root = new GameObject("FascatRuntimeAsset");
            gltf = new GltfImport();
            string assetUri = new Uri(Path.GetFullPath(assetPath)).AbsoluteUri;
            bool loaded = gltf.Load(assetUri).GetAwaiter().GetResult();
            if (!loaded)
            {
                stopwatch.Stop();
                return RenderResult.Failed("glTFast failed to load the asset", stopwatch.ElapsedMilliseconds);
            }

            bool instantiated = gltf.InstantiateMainSceneAsync(root.transform).GetAwaiter().GetResult();
            if (!instantiated)
            {
                stopwatch.Stop();
                return RenderResult.Failed(
                    "glTFast failed to instantiate the main scene",
                    stopwatch.ElapsedMilliseconds
                );
            }

            Bounds bounds = SceneBounds(root);
            camera = CreateCamera(bounds);
            light = CreateLight(bounds);
            RenderSettings.ambientLight = new Color(0.35f, 0.35f, 0.35f, 1.0f);

            renderTexture = new RenderTexture(PreviewWidth, PreviewHeight, 24, RenderTextureFormat.ARGB32);
            renderTexture.Create();
            camera.targetTexture = renderTexture;
            camera.Render();

            var renderStopwatch = Stopwatch.StartNew();
            for (int frame = 0; frame < PreviewBenchmarkFrames; frame++)
            {
                camera.Render();
            }
            renderStopwatch.Stop();

            RenderTexture.active = renderTexture;
            image = new Texture2D(PreviewWidth, PreviewHeight, TextureFormat.RGBA32, false);
            image.ReadPixels(new Rect(0, 0, PreviewWidth, PreviewHeight), 0, 0);
            image.Apply();
            File.WriteAllBytes(previewPath, image.EncodeToPNG());
            stopwatch.Stop();
            long benchmarkTimeMs = Math.Max(renderStopwatch.ElapsedMilliseconds, 1);

            return new RenderResult
            {
                Status = "rendered",
                Error = "",
                RenderTimeMs = stopwatch.ElapsedMilliseconds,
                RenderedFrames = PreviewBenchmarkFrames,
                BenchmarkTimeMs = benchmarkTimeMs,
                MeasuredFps = PreviewBenchmarkFrames * 1000.0 / benchmarkTimeMs
            };
        }
        catch (Exception exception)
        {
            stopwatch.Stop();
            return new RenderResult
            {
                Status = "failed",
                Error = exception.Message,
                RenderTimeMs = stopwatch.ElapsedMilliseconds,
                RenderedFrames = 0,
                BenchmarkTimeMs = -1,
                MeasuredFps = -1.0
            };
        }
        finally
        {
            RenderTexture.active = previousTarget;
            if (gltf != null)
            {
                gltf.Dispose();
            }
            if (camera != null)
            {
                camera.targetTexture = null;
            }
            if (renderTexture != null)
            {
                renderTexture.Release();
                UnityEngine.Object.DestroyImmediate(renderTexture);
            }
            if (image != null)
            {
                UnityEngine.Object.DestroyImmediate(image);
            }
            if (light != null)
            {
                UnityEngine.Object.DestroyImmediate(light.gameObject);
            }
            if (camera != null)
            {
                UnityEngine.Object.DestroyImmediate(camera.gameObject);
            }
            if (root != null)
            {
                UnityEngine.Object.DestroyImmediate(root);
            }
        }
    }

    static Bounds SceneBounds(GameObject root)
    {
        Renderer[] renderers = root.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            return new Bounds(Vector3.zero, Vector3.one);
        }

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }
        if (bounds.size.sqrMagnitude < 1e-6f)
        {
            bounds.Expand(1.0f);
        }
        return bounds;
    }

    static Camera CreateCamera(Bounds bounds)
    {
        GameObject cameraObject = new GameObject("FascatRuntimeCamera");
        Camera camera = cameraObject.AddComponent<Camera>();
        Vector3 center = bounds.center;
        float radius = Mathf.Max(bounds.extents.magnitude, 0.5f);
        camera.transform.position = center + new Vector3(radius * 1.4f, radius * 0.9f, radius * 1.4f);
        camera.transform.LookAt(center);
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.972f, 0.976f, 0.980f, 1.0f);
        camera.nearClipPlane = Mathf.Max(radius * 0.01f, 0.001f);
        camera.farClipPlane = Mathf.Max(radius * 8.0f, 10.0f);
        camera.fieldOfView = 35.0f;
        camera.allowHDR = false;
        camera.allowMSAA = false;
        return camera;
    }

    static Light CreateLight(Bounds bounds)
    {
        GameObject lightObject = new GameObject("FascatRuntimeKeyLight");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.0f;
        light.transform.rotation = Quaternion.LookRotation(bounds.center - (bounds.center + new Vector3(1.0f, 1.5f, 1.0f)));
        return light;
    }

    static AssetCounts LoadAssetCounts(string assetPath)
    {
        byte[] bytes = File.ReadAllBytes(assetPath);
        string json = IsGlb(bytes) ? ExtractGlbJson(bytes) : Encoding.UTF8.GetString(bytes);
        JObject document = JObject.Parse(json);
        JArray meshes = document["meshes"] as JArray;
        JArray accessors = document["accessors"] as JArray;
        int triangleCount = 0;

        if (meshes != null)
        {
            foreach (JToken meshToken in meshes)
            {
                JObject mesh = meshToken as JObject;
                JArray primitives = mesh == null ? null : mesh["primitives"] as JArray;
                if (primitives == null)
                {
                    continue;
                }
                foreach (JToken primitiveToken in primitives)
                {
                    JObject primitive = primitiveToken as JObject;
                    if (primitive == null)
                    {
                        continue;
                    }
                    int mode = (int?)primitive["mode"] ?? 4;
                    if (mode != 4)
                    {
                        continue;
                    }
                    int accessorIndex = -1;
                    if ((int?)primitive["indices"] != null)
                    {
                        accessorIndex = (int)primitive["indices"];
                    }
                    else
                    {
                        JObject attributes = primitive["attributes"] as JObject;
                        if (attributes != null && (int?)attributes["POSITION"] != null)
                        {
                            accessorIndex = (int)attributes["POSITION"];
                        }
                    }
                    triangleCount += AccessorTriangleCount(accessors, accessorIndex);
                }
            }
        }

        return new AssetCounts
        {
            Meshes = meshes == null ? 0 : meshes.Count,
            Triangles = triangleCount,
            MemoryBytes = bytes.LongLength + Encoding.UTF8.GetByteCount(json)
        };
    }

    static int AccessorTriangleCount(JArray accessors, int accessorIndex)
    {
        if (accessors == null || accessorIndex < 0 || accessorIndex >= accessors.Count)
        {
            return 0;
        }
        JObject accessor = accessors[accessorIndex] as JObject;
        if (accessor == null || (int?)accessor["count"] == null)
        {
            return 0;
        }
        return (int)accessor["count"] / 3;
    }

    static bool IsGlb(byte[] bytes)
    {
        return bytes.Length >= 20 && bytes[0] == 0x67 && bytes[1] == 0x6c && bytes[2] == 0x54 && bytes[3] == 0x46;
    }

    static string ExtractGlbJson(byte[] bytes)
    {
        uint jsonLength = ReadUInt32LittleEndian(bytes, 12);
        uint jsonType = ReadUInt32LittleEndian(bytes, 16);
        if (jsonType != 0x4e4f534a)
        {
            throw new InvalidDataException("first GLB chunk is not JSON");
        }
        if (jsonLength > int.MaxValue || 20 + jsonLength > bytes.LongLength)
        {
            throw new InvalidDataException("GLB JSON chunk length is invalid");
        }
        return Encoding.UTF8.GetString(bytes, 20, (int)jsonLength);
    }

    static uint ReadUInt32LittleEndian(byte[] bytes, int offset)
    {
        return (uint)(bytes[offset]
            | bytes[offset + 1] << 8
            | bytes[offset + 2] << 16
            | bytes[offset + 3] << 24);
    }

    struct AssetCounts
    {
        public int Meshes;
        public int Triangles;
        public long MemoryBytes;
    }

    struct RenderResult
    {
        public string Status;
        public string Error;
        public long RenderTimeMs;
        public int RenderedFrames;
        public long BenchmarkTimeMs;
        public double MeasuredFps;

        public static RenderResult NotRequested()
        {
            return new RenderResult
            {
                Status = "not_requested",
                Error = "",
                RenderTimeMs = -1,
                RenderedFrames = 0,
                BenchmarkTimeMs = -1,
                MeasuredFps = -1.0
            };
        }

        public static RenderResult Failed(string error, long renderTimeMs = -1)
        {
            return new RenderResult
            {
                Status = "failed",
                Error = error,
                RenderTimeMs = renderTimeMs,
                RenderedFrames = 0,
                BenchmarkTimeMs = -1,
                MeasuredFps = -1.0
            };
        }
    }
}
