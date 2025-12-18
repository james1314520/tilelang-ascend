// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/ascend.h
 * \brief Define ascend-related operators.
 *
 */

#ifndef TVM_TL_OP_ELEM_H_
#define TVM_TL_OP_ELEM_H_

#include "op.h"

namespace tvm {
namespace tl {

using namespace tir;

class AscendCopy : public Operator {
public:
  AscendCopy(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) final;
  static const Op &Get();

private:
  Array<PrimExpr> args_;

  Buffer src, dst;

  Array<Range> src_range, dst_range;
  Array<PrimExpr> src_extents, dst_extents;
  int srcN;
  bool enRelu;
};


TVM_DLL const Op &ascend_exp();

TVM_DLL const Op &ascend_ln();

TVM_DLL const Op &ascend_abs();

TVM_DLL const Op &ascend_reciprocal();

TVM_DLL const Op &ascend_sqrt();

TVM_DLL const Op &ascend_rsqrt();

TVM_DLL const Op &ascend_relu();

TVM_DLL const Op &ascend_not();

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_