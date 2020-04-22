/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file partition_graph_util.cc
 * \brief Define functions of AttrsComparor and SubgraphPartitionor class.
 */

#include "partition_graph_util.h"
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/attrs/nn.h>
#include "../../ir/attr_functor.h"
namespace tvm {
namespace relay {
namespace partition {
bool AttrsComparor::CompareNonDefault(const Attrs pself,
		const Attrs other) {
	auto p = pself.operator->();
	auto o = other.operator->();
	if (!p && !o)
		return true;
	if(!(p && o))
		return false;
	if (p->type_index() != o->type_index())
		return false;
	StructuralEqual equal;
	EqualVisitor visitor(p, o, equal);
	const_cast<BaseAttrsNode*>(p)->VisitEachNonDefaultAttrs(
			&visitor);
	return visitor.result_;
}
std::unordered_map<const Object*, size_t> RefVisitor::GetCounter(Expr expr) {
	ExprVisitor::VisitExpr(expr);
	return visit_counter_;
}
Type InferType(const Expr expr) {
	Function func = Function(FreeVars(expr), expr, Type(),
			FreeTypeVars(expr, IRModule()), { });

	auto mod = IRModule::FromExpr(func);
	mod = transform::InferType()(mod);
	auto entry_func = Downcast<Function>(mod->Lookup("main"));
	Expr new_expr =
			expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
	return new_expr->checked_type();
}
Expr Cast(const Expr expr, DataType dst_type) {
	if (dst_type.is_handle())
		return expr;
	DataType src_type;
	src_type = InferType(expr).as<TensorTypeNode>()->dtype;
	if (src_type == dst_type) {
		return expr;
	} else {
		Expr new_expr =  cast_(expr, dst_type);
		return new_expr;
	}
}
std::string ConvertOpName(std::string text) {
	for (size_t i = 0; i < text.length(); i++) {
		if (text[i] == '.')
			text[i] = '_';
	}
	return text;
}


}// namespace partition
}// namespace relay
}// namespace tvm
