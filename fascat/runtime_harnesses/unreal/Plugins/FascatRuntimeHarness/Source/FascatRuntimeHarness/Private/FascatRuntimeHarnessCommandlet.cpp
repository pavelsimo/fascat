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
    FString Backend;
    TArray<FString> Limitations;
    int64 RenderTimeMs = -1;
    int32 RenderedFrames = 0;
    int64 BenchmarkTimeMs = -1;
    double MeasuredFps = -1.0;

    static FFascatRenderResult NotRequested()
    {
        FFascatRenderResult Result;
        Result.Status = TEXT("not_requested");
        Result.Backend = TEXT("none");
        return Result;
    }

    static FFascatRenderResult Failed(const FString& Error, int64 RenderTimeMs = -1)
    {
        FFascatRenderResult Result;
        Result.Status = TEXT("failed");
        Result.Error = Error;
        Result.Backend = TEXT("none");
        Result.RenderTimeMs = RenderTimeMs;
        return Result;
    }
};

struct FFascatPreviewPoint
{
    float X = 0.0f;
    float Y = 0.0f;
    float Depth = 0.0f;
};

struct FFascatPreviewTriangle
{
    FVector A;
    FVector B;
    FVector C;
    FColor Color;
};

struct FFascatPreviewGeometry
{
    TArray<FFascatPreviewTriangle> Triangles;
    FVector Min = FVector::ZeroVector;
    FVector Max = FVector::ZeroVector;
    bool HasBounds = false;
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

bool ExtractJsonAndBinary(
    const TArray<uint8>& Bytes,
    FString& OutJsonText,
    TArray<uint8>& OutBinaryData,
    FString& OutError
)
{
    OutBinaryData.Reset();
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
    if (!Utf8ToString(Bytes.GetData() + 20, static_cast<int32>(JsonLength), OutJsonText))
    {
        return false;
    }

    const int64 BinaryHeaderOffset = 20 + static_cast<int64>(JsonLength);
    if (BinaryHeaderOffset + 8 <= Bytes.Num())
    {
        uint32 BinaryLength = 0;
        uint32 BinaryType = 0;
        if (ReadUInt32LittleEndian(Bytes, static_cast<int32>(BinaryHeaderOffset), BinaryLength)
            && ReadUInt32LittleEndian(Bytes, static_cast<int32>(BinaryHeaderOffset + 4), BinaryType)
            && BinaryType == 0x004e4942
            && BinaryLength <= static_cast<uint32>(TNumericLimits<int32>::Max())
            && BinaryHeaderOffset + 8 + static_cast<int64>(BinaryLength) <= Bytes.Num())
        {
            OutBinaryData.Append(Bytes.GetData() + BinaryHeaderOffset + 8, static_cast<int32>(BinaryLength));
        }
    }
    return true;
}

bool ExtractJsonText(const TArray<uint8>& Bytes, FString& OutJsonText, FString& OutError)
{
    TArray<uint8> BinaryData;
    return ExtractJsonAndBinary(Bytes, OutJsonText, BinaryData, OutError);
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

TSharedPtr<FJsonObject> JsonObjectAt(const TArray<TSharedPtr<FJsonValue>>* Values, int32 Index)
{
    if (Values == nullptr || !Values->IsValidIndex(Index))
    {
        return nullptr;
    }
    return (*Values)[Index]->AsObject();
}

bool JsonIntField(const TSharedPtr<FJsonObject>& Object, const TCHAR* FieldName, int32& OutValue)
{
    if (!Object.IsValid())
    {
        return false;
    }
    double Number = 0.0;
    if (!Object->TryGetNumberField(FieldName, Number))
    {
        return false;
    }
    OutValue = FMath::FloorToInt(Number);
    return true;
}

int32 JsonIntFieldOrDefault(const TSharedPtr<FJsonObject>& Object, const TCHAR* FieldName, int32 DefaultValue)
{
    int32 Value = DefaultValue;
    return JsonIntField(Object, FieldName, Value) ? Value : DefaultValue;
}

bool ReadFloatLittleEndian(const TArray<uint8>& Bytes, int64 Offset, float& OutValue)
{
    if (Offset < 0 || Offset + 3 >= Bytes.Num())
    {
        return false;
    }
    const int32 ByteOffset = static_cast<int32>(Offset);
    uint32 Bits = static_cast<uint32>(Bytes[ByteOffset])
        | (static_cast<uint32>(Bytes[ByteOffset + 1]) << 8)
        | (static_cast<uint32>(Bytes[ByteOffset + 2]) << 16)
        | (static_cast<uint32>(Bytes[ByteOffset + 3]) << 24);
    FMemory::Memcpy(&OutValue, &Bits, sizeof(float));
    return true;
}

FVector VectorSubtract(const FVector& A, const FVector& B)
{
    return FVector(A.X - B.X, A.Y - B.Y, A.Z - B.Z);
}

FVector VectorAdd(const FVector& A, const FVector& B)
{
    return FVector(A.X + B.X, A.Y + B.Y, A.Z + B.Z);
}

FVector VectorScale(const FVector& Value, float Scale)
{
    return FVector(Value.X * Scale, Value.Y * Scale, Value.Z * Scale);
}

float VectorDot(const FVector& A, const FVector& B)
{
    return A.X * B.X + A.Y * B.Y + A.Z * B.Z;
}

FVector VectorCross(const FVector& A, const FVector& B)
{
    return FVector(
        A.Y * B.Z - A.Z * B.Y,
        A.Z * B.X - A.X * B.Z,
        A.X * B.Y - A.Y * B.X
    );
}

float VectorLength(const FVector& Value)
{
    return FMath::Sqrt(VectorDot(Value, Value));
}

FVector VectorNormalize(const FVector& Value, const FVector& Fallback)
{
    const float Length = VectorLength(Value);
    if (Length <= 1.0e-6f)
    {
        return Fallback;
    }
    return VectorScale(Value, 1.0f / Length);
}

void ExtendBounds(FFascatPreviewGeometry& Geometry, const FVector& Point)
{
    if (!Geometry.HasBounds)
    {
        Geometry.Min = Point;
        Geometry.Max = Point;
        Geometry.HasBounds = true;
        return;
    }
    Geometry.Min.X = FMath::Min(Geometry.Min.X, Point.X);
    Geometry.Min.Y = FMath::Min(Geometry.Min.Y, Point.Y);
    Geometry.Min.Z = FMath::Min(Geometry.Min.Z, Point.Z);
    Geometry.Max.X = FMath::Max(Geometry.Max.X, Point.X);
    Geometry.Max.Y = FMath::Max(Geometry.Max.Y, Point.Y);
    Geometry.Max.Z = FMath::Max(Geometry.Max.Z, Point.Z);
}

FColor ColorFromFactor(const TArray<TSharedPtr<FJsonValue>>* Values)
{
    if (Values == nullptr || Values->Num() < 3)
    {
        return FColor(56, 116, 170, 255);
    }
    const double R = (*Values)[0]->AsNumber();
    const double G = (*Values)[1]->AsNumber();
    const double B = (*Values)[2]->AsNumber();
    const double A = Values->Num() >= 4 ? (*Values)[3]->AsNumber() : 1.0;
    return FColor(
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(R * 255.0), 0, 255)),
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(G * 255.0), 0, 255)),
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(B * 255.0), 0, 255)),
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(A * 255.0), 0, 255))
    );
}

FColor PrimitiveBaseColor(const TArray<TSharedPtr<FJsonValue>>* Materials, int32 MaterialIndex)
{
    TSharedPtr<FJsonObject> Material = JsonObjectAt(Materials, MaterialIndex);
    if (!Material.IsValid())
    {
        return FColor(56, 116, 170, 255);
    }
    TSharedPtr<FJsonObject> Pbr;
    if (!Material->TryGetObjectField(TEXT("pbrMetallicRoughness"), Pbr) || !Pbr.IsValid())
    {
        return FColor(56, 116, 170, 255);
    }
    const TArray<TSharedPtr<FJsonValue>>* Factors = nullptr;
    if (!Pbr->TryGetArrayField(TEXT("baseColorFactor"), Factors))
    {
        return FColor(56, 116, 170, 255);
    }
    return ColorFromFactor(Factors);
}

bool ReadAccessorPositions(
    const TArray<TSharedPtr<FJsonValue>>* Accessors,
    const TArray<TSharedPtr<FJsonValue>>* BufferViews,
    const TArray<uint8>& BinaryData,
    int32 AccessorIndex,
    TArray<FVector>& OutPositions,
    FString& OutError
)
{
    TSharedPtr<FJsonObject> Accessor = JsonObjectAt(Accessors, AccessorIndex);
    if (!Accessor.IsValid())
    {
        OutError = TEXT("missing POSITION accessor");
        return false;
    }
    int32 BufferViewIndex = -1;
    int32 ComponentType = 0;
    int32 Count = 0;
    if (!JsonIntField(Accessor, TEXT("bufferView"), BufferViewIndex)
        || !JsonIntField(Accessor, TEXT("componentType"), ComponentType)
        || !JsonIntField(Accessor, TEXT("count"), Count))
    {
        OutError = TEXT("POSITION accessor is missing bufferView/componentType/count");
        return false;
    }
    FString Type;
    Accessor->TryGetStringField(TEXT("type"), Type);
    if (ComponentType != 5126 || Type != TEXT("VEC3"))
    {
        OutError = TEXT("packaged Unreal geometry preview currently supports FLOAT VEC3 positions");
        return false;
    }
    TSharedPtr<FJsonObject> BufferView = JsonObjectAt(BufferViews, BufferViewIndex);
    if (!BufferView.IsValid())
    {
        OutError = TEXT("POSITION accessor references a missing bufferView");
        return false;
    }
    const int32 ViewOffset = JsonIntFieldOrDefault(BufferView, TEXT("byteOffset"), 0);
    const int32 AccessorOffset = JsonIntFieldOrDefault(Accessor, TEXT("byteOffset"), 0);
    const int32 Stride = JsonIntFieldOrDefault(BufferView, TEXT("byteStride"), 12);
    if (Count < 0 || Stride < 12)
    {
        OutError = TEXT("POSITION accessor stride/count is invalid");
        return false;
    }
    const int64 Start = static_cast<int64>(ViewOffset) + static_cast<int64>(AccessorOffset);
    if (Start < 0 || Start + static_cast<int64>(Count - 1) * Stride + 12 > BinaryData.Num())
    {
        OutError = TEXT("POSITION accessor extends beyond the GLB binary buffer");
        return false;
    }
    OutPositions.Reset();
    OutPositions.Reserve(Count);
    for (int32 Index = 0; Index < Count; ++Index)
    {
        const int64 Offset = Start + static_cast<int64>(Index) * Stride;
        float X = 0.0f;
        float Y = 0.0f;
        float Z = 0.0f;
        if (!ReadFloatLittleEndian(BinaryData, Offset, X)
            || !ReadFloatLittleEndian(BinaryData, Offset + 4, Y)
            || !ReadFloatLittleEndian(BinaryData, Offset + 8, Z))
        {
            OutError = TEXT("failed to read POSITION accessor data");
            return false;
        }
        OutPositions.Add(FVector(X, Y, Z));
    }
    return true;
}

bool ReadAccessorIndices(
    const TArray<TSharedPtr<FJsonValue>>* Accessors,
    const TArray<TSharedPtr<FJsonValue>>* BufferViews,
    const TArray<uint8>& BinaryData,
    int32 AccessorIndex,
    int32 PositionCount,
    TArray<int32>& OutIndices,
    FString& OutError
)
{
    OutIndices.Reset();
    if (AccessorIndex < 0)
    {
        OutIndices.Reserve(PositionCount);
        for (int32 Index = 0; Index < PositionCount; ++Index)
        {
            OutIndices.Add(Index);
        }
        return true;
    }
    TSharedPtr<FJsonObject> Accessor = JsonObjectAt(Accessors, AccessorIndex);
    if (!Accessor.IsValid())
    {
        OutError = TEXT("missing index accessor");
        return false;
    }
    int32 BufferViewIndex = -1;
    int32 ComponentType = 0;
    int32 Count = 0;
    if (!JsonIntField(Accessor, TEXT("bufferView"), BufferViewIndex)
        || !JsonIntField(Accessor, TEXT("componentType"), ComponentType)
        || !JsonIntField(Accessor, TEXT("count"), Count))
    {
        OutError = TEXT("index accessor is missing bufferView/componentType/count");
        return false;
    }
    FString Type;
    Accessor->TryGetStringField(TEXT("type"), Type);
    if (Type != TEXT("SCALAR") || (ComponentType != 5121 && ComponentType != 5123 && ComponentType != 5125))
    {
        OutError = TEXT("packaged Unreal geometry preview currently supports unsigned byte/short/int indices");
        return false;
    }
    TSharedPtr<FJsonObject> BufferView = JsonObjectAt(BufferViews, BufferViewIndex);
    if (!BufferView.IsValid())
    {
        OutError = TEXT("index accessor references a missing bufferView");
        return false;
    }
    const int32 ComponentSize = ComponentType == 5121 ? 1 : (ComponentType == 5123 ? 2 : 4);
    const int32 ViewOffset = JsonIntFieldOrDefault(BufferView, TEXT("byteOffset"), 0);
    const int32 AccessorOffset = JsonIntFieldOrDefault(Accessor, TEXT("byteOffset"), 0);
    const int32 Stride = JsonIntFieldOrDefault(BufferView, TEXT("byteStride"), ComponentSize);
    if (Count < 0 || Stride < ComponentSize)
    {
        OutError = TEXT("index accessor stride/count is invalid");
        return false;
    }
    const int64 Start = static_cast<int64>(ViewOffset) + static_cast<int64>(AccessorOffset);
    if (Start < 0 || Start + static_cast<int64>(Count - 1) * Stride + ComponentSize > BinaryData.Num())
    {
        OutError = TEXT("index accessor extends beyond the GLB binary buffer");
        return false;
    }
    OutIndices.Reserve(Count);
    for (int32 Index = 0; Index < Count; ++Index)
    {
        const int64 Offset = Start + static_cast<int64>(Index) * Stride;
        uint32 Value = 0;
        const int32 ByteOffset = static_cast<int32>(Offset);
        if (ComponentType == 5121)
        {
            Value = BinaryData[ByteOffset];
        }
        else if (ComponentType == 5123)
        {
            Value = static_cast<uint32>(BinaryData[ByteOffset])
                | (static_cast<uint32>(BinaryData[ByteOffset + 1]) << 8);
        }
        else if (!ReadUInt32LittleEndian(BinaryData, ByteOffset, Value))
        {
            OutError = TEXT("failed to read index accessor data");
            return false;
        }
        if (Value >= static_cast<uint32>(PositionCount) || Value > static_cast<uint32>(TNumericLimits<int32>::Max()))
        {
            OutError = TEXT("index accessor references a missing POSITION vertex");
            return false;
        }
        OutIndices.Add(static_cast<int32>(Value));
    }
    return true;
}

bool ReadPreviewGeometry(const FString& AssetPath, FFascatPreviewGeometry& OutGeometry, FString& OutError)
{
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *AssetPath))
    {
        OutError = TEXT("failed to read asset file");
        return false;
    }

    FString JsonText;
    TArray<uint8> BinaryData;
    if (!ExtractJsonAndBinary(Bytes, JsonText, BinaryData, OutError))
    {
        return false;
    }
    if (BinaryData.Num() == 0)
    {
        OutError = TEXT("asset-driven packaged Unreal preview currently requires GLB binary buffer data");
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
    const TArray<TSharedPtr<FJsonValue>>* BufferViews = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* Materials = nullptr;
    Document->TryGetArrayField(TEXT("meshes"), Meshes);
    Document->TryGetArrayField(TEXT("accessors"), Accessors);
    Document->TryGetArrayField(TEXT("bufferViews"), BufferViews);
    Document->TryGetArrayField(TEXT("materials"), Materials);
    if (Meshes == nullptr || Accessors == nullptr || BufferViews == nullptr)
    {
        OutError = TEXT("glTF document is missing meshes/accessors/bufferViews required for preview");
        return false;
    }

    OutGeometry = FFascatPreviewGeometry();
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
            const int32 Mode = JsonIntFieldOrDefault(Primitive, TEXT("mode"), 4);
            if (Mode != 4)
            {
                continue;
            }
            TSharedPtr<FJsonObject> Attributes;
            double PositionNumber = -1.0;
            if (!Primitive->TryGetObjectField(TEXT("attributes"), Attributes)
                || !Attributes.IsValid()
                || !Attributes->TryGetNumberField(TEXT("POSITION"), PositionNumber))
            {
                continue;
            }

            TArray<FVector> Positions;
            if (!ReadAccessorPositions(
                    Accessors,
                    BufferViews,
                    BinaryData,
                    FMath::FloorToInt(PositionNumber),
                    Positions,
                    OutError
                ))
            {
                return false;
            }

            double IndexNumber = -1.0;
            int32 IndexAccessor = Primitive->TryGetNumberField(TEXT("indices"), IndexNumber)
                ? FMath::FloorToInt(IndexNumber)
                : -1;
            TArray<int32> Indices;
            if (!ReadAccessorIndices(Accessors, BufferViews, BinaryData, IndexAccessor, Positions.Num(), Indices, OutError))
            {
                return false;
            }

            const int32 MaterialIndex = JsonIntFieldOrDefault(Primitive, TEXT("material"), -1);
            const FColor Color = PrimitiveBaseColor(Materials, MaterialIndex);
            const int32 TriangleIndexCount = (Indices.Num() / 3) * 3;
            for (int32 Index = 0; Index < TriangleIndexCount; Index += 3)
            {
                const FVector& A = Positions[Indices[Index]];
                const FVector& B = Positions[Indices[Index + 1]];
                const FVector& C = Positions[Indices[Index + 2]];
                OutGeometry.Triangles.Add({A, B, C, Color});
                ExtendBounds(OutGeometry, A);
                ExtendBounds(OutGeometry, B);
                ExtendBounds(OutGeometry, C);
            }
        }
    }
    if (OutGeometry.Triangles.Num() == 0)
    {
        OutError = TEXT("glTF document does not contain supported triangle geometry for preview");
        return false;
    }
    return true;
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

FColor ShadedColor(const FColor& Color, const FVector& A, const FVector& B, const FVector& C)
{
    const FVector Normal = VectorNormalize(
        VectorCross(VectorSubtract(B, A), VectorSubtract(C, A)),
        FVector(0.0f, 0.0f, 1.0f)
    );
    const FVector Light = VectorNormalize(FVector(0.45f, 0.65f, 0.55f), FVector(0.0f, 1.0f, 0.0f));
    const float Intensity = 0.36f + 0.64f * FMath::Abs(VectorDot(Normal, Light));
    return FColor(
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(static_cast<float>(Color.R) * Intensity), 0, 255)),
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(static_cast<float>(Color.G) * Intensity), 0, 255)),
        static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(static_cast<float>(Color.B) * Intensity), 0, 255)),
        Color.A
    );
}

void DrawRasterTriangle(
    TArray<FColor>& Pixels,
    TArray<float>& DepthBuffer,
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
            const float W0 = EdgeValue(B, C, SampleX, SampleY) / Area;
            const float W1 = EdgeValue(C, A, SampleX, SampleY) / Area;
            const float W2 = EdgeValue(A, B, SampleX, SampleY) / Area;
            if (W0 < 0.0f || W1 < 0.0f || W2 < 0.0f)
            {
                continue;
            }
            const int32 PixelIndex = Y * PreviewWidth + X;
            const float Depth = W0 * A.Depth + W1 * B.Depth + W2 * C.Depth;
            if (Depth < DepthBuffer[PixelIndex])
            {
                continue;
            }
            DepthBuffer[PixelIndex] = Depth;
            Pixels[PixelIndex] = Color;
        }
    }
}

void DrawGeometryPreviewFrame(const FFascatPreviewGeometry& Geometry, int32 FrameIndex, TArray<FColor>& Pixels)
{
    Pixels.SetNum(PreviewWidth * PreviewHeight);
    for (int32 Index = 0; Index < Pixels.Num(); ++Index)
    {
        const int32 Y = Index / PreviewWidth;
        const uint8 Shade = static_cast<uint8>(248 - FMath::Clamp(Y / 48, 0, 8));
        Pixels[Index] = FColor(Shade, static_cast<uint8>(Shade + 1), static_cast<uint8>(Shade + 2), 255);
    }
    if (!Geometry.HasBounds || Geometry.Triangles.Num() == 0)
    {
        return;
    }

    TArray<float> DepthBuffer;
    DepthBuffer.Init(-TNumericLimits<float>::Max(), PreviewWidth * PreviewHeight);
    const FVector Center = VectorScale(VectorAdd(Geometry.Min, Geometry.Max), 0.5f);
    const FVector Extents = VectorScale(VectorSubtract(Geometry.Max, Geometry.Min), 0.5f);
    const float Radius = FMath::Max(VectorLength(Extents), 0.5f);
    const float Orbit = static_cast<float>(FrameIndex) * 0.015f;
    const FVector ViewDirection = VectorNormalize(
        FVector(1.35f + FMath::Sin(Orbit) * 0.08f, 0.85f, 1.35f + FMath::Cos(Orbit) * 0.08f),
        FVector(1.0f, 0.6f, 1.0f)
    );
    const FVector WorldUp(0.0f, 1.0f, 0.0f);
    FVector Right = VectorNormalize(VectorCross(WorldUp, ViewDirection), FVector(1.0f, 0.0f, 0.0f));
    FVector Up = VectorNormalize(VectorCross(ViewDirection, Right), FVector(0.0f, 1.0f, 0.0f));
    const float Scale = static_cast<float>(FMath::Min(PreviewWidth, PreviewHeight)) * 0.39f / Radius;

    for (const FFascatPreviewTriangle& Triangle : Geometry.Triangles)
    {
        const FVector DA = VectorSubtract(Triangle.A, Center);
        const FVector DB = VectorSubtract(Triangle.B, Center);
        const FVector DC = VectorSubtract(Triangle.C, Center);
        const FFascatPreviewPoint A{
            PreviewWidth * 0.5f + VectorDot(DA, Right) * Scale,
            PreviewHeight * 0.53f - VectorDot(DA, Up) * Scale,
            VectorDot(DA, ViewDirection)
        };
        const FFascatPreviewPoint B{
            PreviewWidth * 0.5f + VectorDot(DB, Right) * Scale,
            PreviewHeight * 0.53f - VectorDot(DB, Up) * Scale,
            VectorDot(DB, ViewDirection)
        };
        const FFascatPreviewPoint C{
            PreviewWidth * 0.5f + VectorDot(DC, Right) * Scale,
            PreviewHeight * 0.53f - VectorDot(DC, Up) * Scale,
            VectorDot(DC, ViewDirection)
        };
        DrawRasterTriangle(Pixels, DepthBuffer, A, B, C, ShadedColor(Triangle.Color, Triangle.A, Triangle.B, Triangle.C));
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

FFascatRenderResult RenderPreview(
    const FString& AssetPath,
    const FString& PreviewPath,
    const FFascatAssetCounts& Counts
)
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
    FFascatPreviewGeometry Geometry;
    FString GeometryError;
    const bool HasGeometryPreview = ReadPreviewGeometry(AssetPath, Geometry, GeometryError);
    const double BenchmarkStart = FPlatformTime::Seconds();
    for (int32 Frame = 0; Frame < PreviewBenchmarkFrames; ++Frame)
    {
        if (HasGeometryPreview)
        {
            DrawGeometryPreviewFrame(Geometry, Frame, Pixels);
        }
        else
        {
            DrawPreviewFrame(Counts, Frame, Pixels);
        }
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
    Result.Status = HasGeometryPreview ? TEXT("rendered") : TEXT("rendered_partial");
    Result.Backend = HasGeometryPreview
        ? TEXT("unreal_commandlet_geometry_rasterizer")
        : TEXT("unreal_commandlet_count_preview");
    if (HasGeometryPreview)
    {
        Result.Limitations.Add(
            TEXT("packaged Unreal commandlet rasterizes GLB triangle geometry and baseColorFactor materials; it is not a full Unreal scene renderer")
        );
        Result.Limitations.Add(
            TEXT("textures, node transforms, skinning, animation, and engine material graph behavior are not sampled")
        );
    }
    else
    {
        Result.Limitations.Add(
            FString::Printf(
                TEXT("packaged Unreal commandlet fell back to count-based preview geometry: %s"),
                *GeometryError
            )
        );
    }
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
        Payload->SetStringField(TEXT("render_backend"), TEXT("none"));
        Payload->SetArrayField(TEXT("render_limitations"), TArray<TSharedPtr<FJsonValue>>());
    }
    else
    {
        Payload->SetStringField(TEXT("preview_path"), PreviewPath);
        Payload->SetStringField(TEXT("render_status"), RenderResult.Status);
        Payload->SetStringField(TEXT("render_error"), RenderResult.Error);
        Payload->SetStringField(TEXT("render_backend"), RenderResult.Backend);
        TArray<TSharedPtr<FJsonValue>> Limitations;
        for (const FString& Limitation : RenderResult.Limitations)
        {
            Limitations.Add(MakeShared<FJsonValueString>(Limitation));
        }
        Payload->SetArrayField(TEXT("render_limitations"), Limitations);
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
        RenderResult = RenderPreview(AssetPath, PreviewPath, Counts);
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
