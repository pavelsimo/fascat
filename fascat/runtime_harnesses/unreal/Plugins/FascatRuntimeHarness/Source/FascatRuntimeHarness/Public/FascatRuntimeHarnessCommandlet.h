#pragma once

#include "Commandlets/Commandlet.h"
#include "CoreMinimal.h"

#include "FascatRuntimeHarnessCommandlet.generated.h"

UCLASS()
class FASCATRUNTIMEHARNESS_API UFascatRuntimeHarnessCommandlet : public UCommandlet
{
    GENERATED_BODY()

public:
    int32 Main(const FString& Params) override;
};
