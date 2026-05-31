#include "FascatRuntimeHarnessCommandlet.h"

#include "Dom/JsonObject.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "Misc/CommandLine.h"
#include "Misc/EngineVersion.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/Parse.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
struct FFascatAssetCounts
{
    int32 Meshes = 0;
    int32 Triangles = 0;
    int64 MemoryBytes = 0;
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

FString BuildPayload(
    const FString& Status,
    const FString& Error,
    int64 LoadTimeMs,
    const FFascatAssetCounts& Counts
)
{
    TSharedRef<FJsonObject> Payload = MakeShared<FJsonObject>();
    Payload->SetStringField(TEXT("status"), Status);
    Payload->SetStringField(TEXT("engine_version"), FEngineVersion::Current().ToString());
    Payload->SetNumberField(TEXT("load_time_ms"), LoadTimeMs);
    Payload->SetField(TEXT("measured_fps"), MakeShared<FJsonValueNull>());
    Payload->SetNumberField(TEXT("frame_count"), 0);
    Payload->SetNumberField(TEXT("measurement_duration_ms"), LoadTimeMs);
    Payload->SetNumberField(TEXT("memory_bytes"), Counts.MemoryBytes);
    Payload->SetNumberField(TEXT("meshes"), Counts.Meshes);
    Payload->SetNumberField(TEXT("triangles"), Counts.Triangles);
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
    FParse::Value(FCommandLine::Get(), TEXT("FascatAsset="), AssetPath);
    FParse::Value(FCommandLine::Get(), TEXT("FascatReport="), ReportPath);

    const double Start = FPlatformTime::Seconds();
    FString Status = TEXT("measured");
    FString Error;
    FFascatAssetCounts Counts;

    if (AssetPath.IsEmpty() || !FPaths::FileExists(AssetPath))
    {
        Status = TEXT("failed");
        Error = TEXT("missing -FascatAsset input");
    }
    else if (!ReadAssetCounts(AssetPath, Counts, Error))
    {
        Status = TEXT("failed");
    }

    const int64 LoadTimeMs = static_cast<int64>((FPlatformTime::Seconds() - Start) * 1000.0);
    const FString Payload = BuildPayload(Status, Error, LoadTimeMs, Counts);

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
