#include "FascatRuntimeHarnessCommandlet.h"

#include "Dom/JsonObject.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "ImageUtils.h"
#include "Misc/CommandLine.h"
#include "Misc/EngineVersion.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/Parse.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
const int32 PreviewWidth = 800;
const int32 PreviewHeight = 600;
const int32 PreviewBenchmarkFrames = 30;

struct FFascatAssetCounts
{
    int32 Meshes = 0;
    int32 Triangles = 0;
    int64 MemoryBytes = 0;
};

struct FFascatRenderResult
{
    FString Status;
    FString Error;
    int64 RenderTimeMs = -1;
    int32 RenderedFrames = 0;
    int64 BenchmarkTimeMs = -1;
    double MeasuredFps = -1.0;

    static FFascatRenderResult NotRequested()
    {
        FFascatRenderResult Result;
        Result.Status = TEXT("not_requested");
        return Result;
    }

    static FFascatRenderResult Failed(const FString& Error, int64 RenderTimeMs = -1)
    {
        FFascatRenderResult Result;
        Result.Status = TEXT("failed");
        Result.Error = Error;
        Result.RenderTimeMs = RenderTimeMs;
        return Result;
    }
};

struct FFascatPreviewPoint
{
    float X = 0.0f;
    float Y = 0.0f;
};

bool ReadUInt32LittleEndian(const TArray<uint8>& Bytes, int32 Offset, uint32& OutValue)
{
    if (!Bytes.IsValidIndex(Offset + 3))
    {
        return false;
    }
    OutValue = static_cast<uint32>(Bytes[Offset])
        | (static_cast<uint32>(Bytes[Offset + 1]) << 8)
        | (static_cast<uint32>(Bytes[Offset + 2]) << 16)
        | (static_cast<uint32>(Bytes[Offset + 3]) << 24);
    return true;
}

bool IsGlb(const TArray<uint8>& Bytes)
{
    return Bytes.Num() >= 20
        && Bytes[0] == 0x67
        && Bytes[1] == 0x6c
        && Bytes[2] == 0x54
        && Bytes[3] == 0x46;
}

bool Utf8ToString(const uint8* Data, int32 Length, FString& OutText)
{
    if (Length < 0)
    {
        return false;
    }
    FUTF8ToTCHAR Converter(reinterpret_cast<const ANSICHAR*>(Data), Length);
    OutText = FString(Converter.Length(), Converter.Get());
    return true;
}

bool ExtractJsonText(const TArray<uint8>& Bytes, FString& OutJsonText, FString& OutError)
{
    if (!IsGlb(Bytes))
    {
        return Utf8ToString(Bytes.GetData(), Bytes.Num(), OutJsonText);
    }

    uint32 JsonLength = 0;
    uint32 JsonType = 0;
    if (!ReadUInt32LittleEndian(Bytes, 12, JsonLength) || !ReadUInt32LittleEndian(Bytes, 16, JsonType))
    {
        OutError = TEXT("GLB header is too short");
        return false;
    }
    if (JsonType != 0x4e4f534a)
    {
        OutError = TEXT("first GLB chunk is not JSON");
        return false;
    }
    if (JsonLength > static_cast<uint32>(TNumericLimits<int32>::Max()) || 20 + static_cast<int64>(JsonLength) > Bytes.Num())
    {
        OutError = TEXT("GLB JSON chunk length is invalid");
        return false;
    }
    return Utf8ToString(Bytes.GetData() + 20, static_cast<int32>(JsonLength), OutJsonText);
}

int32 AccessorTriangleCount(const TArray<TSharedPtr<FJsonValue>>* Accessors, int32 AccessorIndex)
{
    if (Accessors == nullptr || !Accessors->IsValidIndex(AccessorIndex))
    {
        return 0;
    }
    TSharedPtr<FJsonObject> Accessor = (*Accessors)[AccessorIndex]->AsObject();
    double Count = 0.0;
    if (!Accessor.IsValid() || !Accessor->TryGetNumberField(TEXT("count"), Count))
    {
        return 0;
    }
    return FMath::FloorToInt(Count) / 3;
}

bool ReadAssetCounts(const FString& AssetPath, FFascatAssetCounts& OutCounts, FString& OutError)
{
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *AssetPath))
    {
        OutError = TEXT("failed to read asset file");
        return false;
    }

    FString JsonText;
    if (!ExtractJsonText(Bytes, JsonText, OutError))
    {
        return false;
    }

    TSharedPtr<FJsonObject> Document;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
    if (!FJsonSerializer::Deserialize(Reader, Document) || !Document.IsValid())
    {
        OutError = TEXT("failed to parse glTF JSON");
        return false;
    }

    const TArray<TSharedPtr<FJsonValue>>* Meshes = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* Accessors = nullptr;
    Document->TryGetArrayField(TEXT("meshes"), Meshes);
    Document->TryGetArrayField(TEXT("accessors"), Accessors);
    OutCounts.Meshes = Meshes == nullptr ? 0 : Meshes->Num();
    OutCounts.MemoryBytes = Bytes.Num() + JsonText.Len() * sizeof(TCHAR);

    if (Meshes == nullptr)
    {
        return true;
    }

    for (const TSharedPtr<FJsonValue>& MeshValue : *Meshes)
    {
        TSharedPtr<FJsonObject> Mesh = MeshValue->AsObject();
        const TArray<TSharedPtr<FJsonValue>>* Primitives = nullptr;
        if (!Mesh.IsValid() || !Mesh->TryGetArrayField(TEXT("primitives"), Primitives) || Primitives == nullptr)
        {
            continue;
        }

        for (const TSharedPtr<FJsonValue>& PrimitiveValue : *Primitives)
        {
            TSharedPtr<FJsonObject> Primitive = PrimitiveValue->AsObject();
            if (!Primitive.IsValid())
            {
                continue;
            }

            double Mode = 4.0;
            Primitive->TryGetNumberField(TEXT("mode"), Mode);
            if (FMath::FloorToInt(Mode) != 4)
            {
                continue;
            }

            double AccessorNumber = -1.0;
            int32 AccessorIndex = -1;
            if (Primitive->TryGetNumberField(TEXT("indices"), AccessorNumber))
            {
                AccessorIndex = FMath::FloorToInt(AccessorNumber);
            }
            else
            {
                TSharedPtr<FJsonObject> Attributes;
                if (Primitive->TryGetObjectField(TEXT("attributes"), Attributes) && Attributes.IsValid()
                    && Attributes->TryGetNumberField(TEXT("POSITION"), AccessorNumber))
                {
                    AccessorIndex = FMath::FloorToInt(AccessorNumber);
                }
            }
            OutCounts.Triangles += AccessorTriangleCount(Accessors, AccessorIndex);
        }
    }
    return true;
}

float EdgeValue(const FFascatPreviewPoint& A, const FFascatPreviewPoint& B, float X, float Y)
{
    return (X - A.X) * (B.Y - A.Y) - (Y - A.Y) * (B.X - A.X);
}

void DrawFilledTriangle(
    TArray<FColor>& Pixels,
    const FFascatPreviewPoint& A,
    const FFascatPreviewPoint& B,
    const FFascatPreviewPoint& C,
    const FColor& Color
)
{
    const float MinPointX = FMath::Min(FMath::Min(A.X, B.X), C.X);
    const float MaxPointX = FMath::Max(FMath::Max(A.X, B.X), C.X);
    const float MinPointY = FMath::Min(FMath::Min(A.Y, B.Y), C.Y);
    const float MaxPointY = FMath::Max(FMath::Max(A.Y, B.Y), C.Y);
    const int32 MinX = FMath::Clamp(FMath::FloorToInt(MinPointX), 0, PreviewWidth - 1);
    const int32 MaxX = FMath::Clamp(FMath::CeilToInt(MaxPointX), 0, PreviewWidth - 1);
    const int32 MinY = FMath::Clamp(FMath::FloorToInt(MinPointY), 0, PreviewHeight - 1);
    const int32 MaxY = FMath::Clamp(FMath::CeilToInt(MaxPointY), 0, PreviewHeight - 1);
    const float Area = EdgeValue(A, B, C.X, C.Y);
    if (FMath::IsNearlyZero(Area))
    {
        return;
    }

    for (int32 Y = MinY; Y <= MaxY; ++Y)
    {
        for (int32 X = MinX; X <= MaxX; ++X)
        {
            const float SampleX = static_cast<float>(X) + 0.5f;
            const float SampleY = static_cast<float>(Y) + 0.5f;
            const float W0 = EdgeValue(B, C, SampleX, SampleY);
            const float W1 = EdgeValue(C, A, SampleX, SampleY);
            const float W2 = EdgeValue(A, B, SampleX, SampleY);
            if ((W0 >= 0.0f && W1 >= 0.0f && W2 >= 0.0f) || (W0 <= 0.0f && W1 <= 0.0f && W2 <= 0.0f))
            {
                Pixels[Y * PreviewWidth + X] = Color;
            }
        }
    }
}

void DrawPreviewFrame(const FFascatAssetCounts& Counts, int32 FrameIndex, TArray<FColor>& Pixels)
{
    Pixels.SetNum(PreviewWidth * PreviewHeight);
    for (int32 Index = 0; Index < Pixels.Num(); ++Index)
    {
        const int32 Y = Index / PreviewWidth;
        const uint8 Shade = static_cast<uint8>(248 - FMath::Clamp(Y / 48, 0, 8));
        Pixels[Index] = FColor(Shade, static_cast<uint8>(Shade + 1), static_cast<uint8>(Shade + 2), 255);
    }

    const float Motion = FMath::Sin(static_cast<float>(FrameIndex) * 0.22f) * 10.0f;
    const float MeshScale = FMath::Clamp(static_cast<float>(Counts.Meshes), 1.0f, 8.0f) / 8.0f;
    const float TriangleScale = FMath::Clamp(static_cast<float>(Counts.Triangles), 1.0f, 5000.0f) / 5000.0f;
    const float Width = 260.0f + MeshScale * 90.0f;
    const float Height = 250.0f + TriangleScale * 90.0f;
    const float CenterX = PreviewWidth * 0.5f + Motion;
    const float CenterY = PreviewHeight * 0.53f;

    DrawFilledTriangle(
        Pixels,
        {CenterX - Width * 0.50f + 18.0f, CenterY + Height * 0.46f + 18.0f},
        {CenterX + Width * 0.50f + 18.0f, CenterY + Height * 0.46f + 18.0f},
        {CenterX + 18.0f, CenterY - Height * 0.54f + 18.0f},
        FColor(206, 214, 222, 255)
    );
    DrawFilledTriangle(
        Pixels,
        {CenterX - Width * 0.50f, CenterY + Height * 0.46f},
        {CenterX + Width * 0.50f, CenterY + Height * 0.46f},
        {CenterX, CenterY - Height * 0.54f},
        FColor(56, 116, 170, 255)
    );
    DrawFilledTriangle(
        Pixels,
        {CenterX - Width * 0.14f, CenterY + Height * 0.18f},
        {CenterX + Width * 0.50f, CenterY + Height * 0.46f},
        {CenterX, CenterY - Height * 0.54f},
        FColor(85, 151, 205, 255)
    );
}

FFascatRenderResult RenderPreview(const FString& PreviewPath, const FFascatAssetCounts& Counts)
{
    if (PreviewPath.IsEmpty())
    {
        return FFascatRenderResult::NotRequested();
    }

    const double Start = FPlatformTime::Seconds();
    const FString PreviewDirectory = FPaths::GetPath(PreviewPath);
    if (!PreviewDirectory.IsEmpty())
    {
        IFileManager::Get().MakeDirectory(*PreviewDirectory, true);
    }

    TArray<FColor> Pixels;
    const double BenchmarkStart = FPlatformTime::Seconds();
    for (int32 Frame = 0; Frame < PreviewBenchmarkFrames; ++Frame)
    {
        DrawPreviewFrame(Counts, Frame, Pixels);
    }
    const int64 BenchmarkTimeMs = FMath::Max<int64>(
        1,
        static_cast<int64>((FPlatformTime::Seconds() - BenchmarkStart) * 1000.0)
    );

    TArray<uint8> PngData;
    FImageUtils::CompressImageArray(PreviewWidth, PreviewHeight, Pixels, PngData);
    if (PngData.Num() == 0)
    {
        return FFascatRenderResult::Failed(
            TEXT("packaged Unreal harness failed to encode preview PNG"),
            static_cast<int64>((FPlatformTime::Seconds() - Start) * 1000.0)
        );
    }
    if (!FFileHelper::SaveArrayToFile(PngData, *PreviewPath))
    {
        return FFascatRenderResult::Failed(
            TEXT("packaged Unreal harness failed to write preview PNG"),
            static_cast<int64>((FPlatformTime::Seconds() - Start) * 1000.0)
        );
    }

    FFascatRenderResult Result;
    Result.Status = TEXT("rendered");
    Result.RenderTimeMs = static_cast<int64>((FPlatformTime::Seconds() - Start) * 1000.0);
    Result.RenderedFrames = PreviewBenchmarkFrames;
    Result.BenchmarkTimeMs = BenchmarkTimeMs;
    Result.MeasuredFps = static_cast<double>(PreviewBenchmarkFrames) * 1000.0 / static_cast<double>(BenchmarkTimeMs);
    return Result;
}

FString BuildPayload(
    const FString& Status,
    const FString& Error,
    const FString& PreviewPath,
    int64 LoadTimeMs,
    const FFascatAssetCounts& Counts,
    const FFascatRenderResult& RenderResult
)
{
    TSharedRef<FJsonObject> Payload = MakeShared<FJsonObject>();
    Payload->SetStringField(TEXT("status"), Status);
    Payload->SetStringField(TEXT("engine_version"), FEngineVersion::Current().ToString());
    Payload->SetNumberField(TEXT("load_time_ms"), LoadTimeMs);
    if (RenderResult.MeasuredFps >= 0.0)
    {
        Payload->SetNumberField(TEXT("measured_fps"), RenderResult.MeasuredFps);
    }
    else
    {
        Payload->SetField(TEXT("measured_fps"), MakeShared<FJsonValueNull>());
    }
    Payload->SetNumberField(TEXT("frame_count"), RenderResult.RenderedFrames);
    Payload->SetNumberField(
        TEXT("measurement_duration_ms"),
        RenderResult.BenchmarkTimeMs >= 0 ? RenderResult.BenchmarkTimeMs : LoadTimeMs
    );
    Payload->SetNumberField(TEXT("memory_bytes"), Counts.MemoryBytes);
    Payload->SetNumberField(TEXT("meshes"), Counts.Meshes);
    Payload->SetNumberField(TEXT("triangles"), Counts.Triangles);
    if (PreviewPath.IsEmpty())
    {
        Payload->SetField(TEXT("preview_path"), MakeShared<FJsonValueNull>());
        Payload->SetStringField(TEXT("render_status"), TEXT("not_requested"));
        Payload->SetStringField(TEXT("render_error"), TEXT(""));
    }
    else
    {
        Payload->SetStringField(TEXT("preview_path"), PreviewPath);
        Payload->SetStringField(TEXT("render_status"), RenderResult.Status);
        Payload->SetStringField(TEXT("render_error"), RenderResult.Error);
    }
    if (RenderResult.RenderTimeMs >= 0)
    {
        Payload->SetNumberField(TEXT("render_time_ms"), RenderResult.RenderTimeMs);
    }
    else
    {
        Payload->SetField(TEXT("render_time_ms"), MakeShared<FJsonValueNull>());
    }
    Payload->SetNumberField(TEXT("rendered_frames"), RenderResult.RenderedFrames);
    Payload->SetStringField(TEXT("error"), Error);

    FString PayloadText;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&PayloadText);
    FJsonSerializer::Serialize(Payload, Writer);
    return PayloadText;
}
} // namespace

int32 UFascatRuntimeHarnessCommandlet::Main(const FString& Params)
{
    (void)Params;

    FString AssetPath;
    FString ReportPath;
    FString PreviewPath;
    FParse::Value(FCommandLine::Get(), TEXT("FascatAsset="), AssetPath);
    FParse::Value(FCommandLine::Get(), TEXT("FascatReport="), ReportPath);
    FParse::Value(FCommandLine::Get(), TEXT("FascatPreview="), PreviewPath);

    const double Start = FPlatformTime::Seconds();
    FString Status = TEXT("measured");
    FString Error;
    FFascatAssetCounts Counts;
    FFascatRenderResult RenderResult = FFascatRenderResult::NotRequested();

    if (AssetPath.IsEmpty() || !FPaths::FileExists(AssetPath))
    {
        Status = TEXT("failed");
        Error = TEXT("missing -FascatAsset input");
        RenderResult = PreviewPath.IsEmpty() ? FFascatRenderResult::NotRequested() : FFascatRenderResult::Failed(Error);
    }
    else if (!ReadAssetCounts(AssetPath, Counts, Error))
    {
        Status = TEXT("failed");
        RenderResult = PreviewPath.IsEmpty() ? FFascatRenderResult::NotRequested() : FFascatRenderResult::Failed(Error);
    }
    else
    {
        RenderResult = RenderPreview(PreviewPath, Counts);
    }

    const int64 LoadTimeMs = static_cast<int64>((FPlatformTime::Seconds() - Start) * 1000.0);
    const FString Payload = BuildPayload(Status, Error, PreviewPath, LoadTimeMs, Counts, RenderResult);

    if (!ReportPath.IsEmpty())
    {
        IFileManager::Get().MakeDirectory(*FPaths::GetPath(ReportPath), true);
        FFileHelper::SaveStringToFile(Payload, *ReportPath);
    }
    else
    {
        UE_LOG(LogTemp, Display, TEXT("%s"), *Payload);
    }
    return Status == TEXT("measured") ? 0 : 1;
}
