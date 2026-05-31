using UnrealBuildTool;

public class FascatRuntimeHarness : ModuleRules
{
    public FascatRuntimeHarness(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PrivateDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "Json",
            "UnrealEd"
        });
    }
}
