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
 *
 * \file partition_subgraph_op.h
 * \brief Abstract class for partition graph.
 */
#ifndef TVM_RELAY_PASS_PARTITION_SUBGRAPH_UTIL_H_
#define TVM_RELAY_PASS_PARTITION_SUBGRAPH_UTIL_H_
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

static auto fpattern_ =
    Op::GetAttrMap<TOpPattern>("TOpPattern");
// The packed function to make cast operator.
static auto cast_ = (*tvm::runtime::Registry::Get("relay.ir.cast"));
// The packed function to make cast operator.
static auto on_device_ = (*tvm::runtime::Registry::Get(
		"relay.op.annotation._make.on_device"));
class AttrsComparor{
public:
	/*
	 * \brief Compare non default attrs
	 * \param expr the Relay expression.
	 * \return Type.
	 */
	static bool CompareNonDefault(const Attrs pself, const Attrs other);
private:
		class EqualVisitor;
		friend class EqualVisitor;
};
class AttrsComparor::EqualVisitor: public AttrVisitor {
public:
	bool result_ { true };
	EqualVisitor(const Object *lhs, const Object *rhs, const StructuralEqual &equal) :
			lhs_(lhs), rhs_(rhs), equal_(equal) {
	}
	template<typename T>
	void CompareAttr(const char *key, T *lhs_value) {
		if (!result_)
			return;
		const T *rhs_value =
				reinterpret_cast<const T*>(reinterpret_cast<const char*>(rhs_)
						+ (reinterpret_cast<const char*>(lhs_value)
								- reinterpret_cast<const char*>(lhs_)));
		if (!equal_(*lhs_value, *rhs_value)) {
			result_ = false;
		} else {
			result_ = true;
		}
	}
	void Visit(const char *key, double *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, int64_t *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, uint64_t *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, int *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, bool *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, std::string *value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, void **value) final {
		CompareAttr(key, value);
	}
	void Visit(const char *key, DataType *value) final {
		// do nothing
	}
	void Visit(const char *key, runtime::NDArray *value) final {
		CompareAttr(key, value);

	}
	void Visit(const char *key, runtime::ObjectRef *obj) final {
		CompareAttr(key, obj);
	}

private:
	const Object *lhs_;
	const Object *rhs_;
	const StructuralEqual &equal_;

};

class RefVisitor: public ExprVisitor {
public:
	std::unordered_map<const Object*, size_t> GetCounter(Expr expr);
};
/*
 * \brief Infer the type of expression.
 * \param expr the Relay expression.
 * \return Type.
 */
Type InferType(const Expr expr);
/*
 * \brief Cast the output data to the specified data type.
 * \param expr the Relay expression.
 * \param tar_type the target data type.
 * \return New expression with the cast operator.
 */
Expr Cast(const Expr expr, DataType dst_type);
/*
 * \brief Convert op name to fused name.
 * \param text op name
 * \return Fused name.
 */
std::string ConvertOpName(std::string text);

} // namespace partition
}// namespace relay
}// namespace tvm
#endif  // TVM_RELAY_PASS_PARTITION_SUBGRAPH_UTIL_H_
