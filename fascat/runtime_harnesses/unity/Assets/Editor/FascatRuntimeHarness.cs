using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

public static class FascatRuntimeHarness
{
    public static void Run()
    {
        string assetPath = Arg("-fascatAsset");
        string reportPath = Arg("-fascatReport");
        var stopwatch = Stopwatch.StartNew();

        string status = "measured";
        string error = "";
        long memoryBytes = 0;
        int meshes = 0;
        int triangles = 0;
        int frameCount = 0;

        if (string.IsNullOrEmpty(assetPath) || !File.Exists(assetPath))
        {
            status = "failed";
            error = "missing -fascatAsset input";
        }
        else
        {
            try
            {
                AssetCounts counts = LoadAssetCounts(assetPath);
                meshes = counts.Meshes;
                triangles = counts.Triangles;
                memoryBytes = counts.MemoryBytes + GC.GetTotalMemory(false);
            }
            catch (Exception exception)
            {
                status = "failed";
                error = exception.Message;
            }
        }

        stopwatch.Stop();
        JObject payload = new JObject
        {
            ["status"] = status,
            ["engine_version"] = Application.unityVersion,
            ["load_time_ms"] = stopwatch.ElapsedMilliseconds,
            ["measured_fps"] = JValue.CreateNull(),
            ["frame_count"] = frameCount,
            ["measurement_duration_ms"] = stopwatch.ElapsedMilliseconds,
            ["memory_bytes"] = memoryBytes,
            ["meshes"] = meshes,
            ["triangles"] = triangles,
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
}
