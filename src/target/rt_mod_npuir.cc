// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "codegen_npuir.h"
#include "codegen_npuir_api.h"

namespace tvm {
namespace codegen {

runtime::Module BuildTileLangNPUIR(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTileLangNPUIR cg;
  cg.Init(output_ssa);

  Array<String> function_names;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIR: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();

  return CSourceModuleCreate(code, "c", function_names);
}

runtime::Module BuildTileLangNPUIRMLIRAPIs(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  CodeGenTileLangNPUIRAPI cg;
  Array<String> function_names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangNPUIRAPI: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetCurrentFunctionName());
  }
  std::string mlirCode = cg.Finish();
  return CSourceModuleCreate(mlirCode, "c", function_names);
}

TVM_REGISTER_GLOBAL("target.build.tilelang_npuir")
    .set_body_typed(BuildTileLangNPUIR);

TVM_REGISTER_TARGET_KIND("npuir", kDLExtDev);

TVM_REGISTER_GLOBAL("target.build.tilelang_npuir_apis")
    .set_body_typed(BuildTileLangNPUIRMLIRAPIs);

} // namespace codegen
} // namespace tvm
