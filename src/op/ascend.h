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

TVM_DLL const Op &ascend_add();

TVM_DLL const Op &ascend_sub();

TVM_DLL const Op &ascend_mul();

TVM_DLL const Op &ascend_div();

TVM_DLL const Op &ascend_max();

TVM_DLL const Op &ascend_min();

TVM_DLL const Op &ascend_and();

TVM_DLL const Op &ascend_or();

TVM_DLL const Op &ascend_adds();

TVM_DLL const Op &ascend_subs();

TVM_DLL const Op &ascend_muls();

TVM_DLL const Op &ascend_divs();

TVM_DLL const Op &ascend_exp();

TVM_DLL const Op &ascend_ln();

TVM_DLL const Op &ascend_abs();

TVM_DLL const Op &ascend_reciprocal();

TVM_DLL const Op &ascend_sqrt();

TVM_DLL const Op &ascend_rsqrt();

TVM_DLL const Op &ascend_relu();

TVM_DLL const Op &ascend_not();

TVM_DLL const Op &ascend_select();

TVM_DLL const Op &ascend_leaky_relu();

TVM_DLL const Op &ascend_axpy();

TVM_DLL const Op &ascend_shiftleft();

TVM_DLL const Op &ascend_shiftright();

TVM_DLL const Op &ascend_sin();

TVM_DLL const Op &ascend_cos();

TVM_DLL const Op &ascend_transpose();

TVM_DLL const Op &ascend_createvecindex();

TVM_DLL const Op &ascend_fill();

TVM_DLL const Op &ascend_arith_progression();

TVM_DLL const Op &ascend_sort();

TVM_DLL const Op &ascend_merge_sort();

TVM_DLL const Op &ascend_topk();

TVM_DLL const Op &ascend_gather_mask();

TVM_DLL const Op &ascend_gatherb();

TVM_DLL const Op &ascend_init_sort_buf();

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_ELEM_H_