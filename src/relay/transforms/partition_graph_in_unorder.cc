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
 * \file partition_graph_in_unorder.cc
 * \brief Pass to partition a Relay graph to subgraph with unordered operators.
 */
#include "partition_graph_util.h"
namespace tvm {
namespace relay {
namespace partition {
class GraphPartitionerInUnorder: public ExprMutator{
public:
	explicit GraphPartitionerInUnorder(const Array<Array<ObjectRef>> op_attrs, const String func_name, const String compiler,
				const int device_type, const DataType data_type) :
				op_attrs_(op_attrs), func_name_(func_name), compiler_(compiler), device_type_(
						device_type), data_type_(data_type) {
	}
	Expr Partition(Expr expr) {
		auto visitor = RefVisitor();
		visit_counter_ = visitor.GetCounter(expr);
		return ExprMutator::Mutate(expr);
	}
	struct Subgraph {
	public:
		// \brief The input arguments of this subgraph.
		std::vector<std::pair<Expr, Var>> args;
		// \brief Function name.
		std::stringstream func_name;
		// Get a new parameter or allocate an old one.
		Var GetOrAllocParam(const Expr &expr, const Type &type) {
			for (auto arg : args) {
				if (expr.same_as(arg.first))
					return arg.second;
			}
			// create a new parameter.
			std::stringstream os;
			os << "p" << args.size();
			auto var = Var(os.str(), type);
			args.push_back( { expr, var });
			return var;
		}
	};

	/*
	 * \brief Check if the node has attrs is in attrs_.
	 * \param Relay expression.
	 * \return ture or false.
	 */
	bool InOpAttrs(const CallNode *call_node) {
		auto op = call_node->op.as<OpNode>();
		if (!op)
			return false;
		for(size_t order=0;order<op_attrs_.size();order++){
			LOG_IF(FATAL, !op_attrs_[order].defined())
					<<"The "<<order<<"-th op name in op_attrs is not defined.";
			if (Downcast<String>(op_attrs_[order][0]) == op->name){
				if (op_attrs_[order][1].defined()) {
					auto attrs = Downcast<Attrs>(op_attrs_[order][1]);
					LOG_IF(FATAL, op->attrs_type_index!=attrs->type_index())
							<< "The input attrs type is not consistant with the op name, \""
							<< op->name
							<< "\" .";
					return AttrsComparor::CompareNonDefault(attrs,
							call_node->attrs);
				} else
					return true;
			}
		}
		return false;
	}
	  /*!
	   * \brief Get unique name for func
	   *
	   * \param name
	   * \return std::string
	   */
	  std::string GetUniqueName(const std::string& name) {
		if (!name_map_.count(name)) {
		  name_map_[name] = 1;
		  return name;
		}
		auto index = name_map_[name];
		name_map_[name] += 1;
		return GetUniqueName(name +"_"+ std::to_string(index));
	  }
	  std::string GetName(const std::string& name) {
		constexpr static size_t kMaxFuncNameLength = 80;
	    std::stringstream ss;
	    if (name.size() > kMaxFuncNameLength) {
	      ss <<  name.substr(0, kMaxFuncNameLength);
	      ss << "_" << std::hash<std::string>{}(name);
	    }else{
	      ss << name;
	    }
	    return GetUniqueName(ss.str());
	  }
	/*
	 * \brief Make a call function node, annotate its device type
	 * 	and cast the output data type to the raw data type.
	 * \param subgraph The Subgraph which the node belongs to.
	 * \param body The body of the function.
	 * \param raw_type The raw type of the original expression.
	 * \return The new expression.
	 */
	Expr MakeFunc(
			std::shared_ptr<Subgraph> &subgraph, Expr &body,
			const Type raw_type) {
		Array<Var> params;
		Array<Expr> arguments;
		for (auto pair : subgraph->args) {
			arguments.push_back(pair.first);
			params.push_back(pair.second);
		}
		auto func = Function(params, body, InferType(body), { });
		if (compiler_!="") {
			//Set defined compiler
			func = WithAttr(std::move(func), attr::kCompiler, compiler_);
			if (func_name_!="") {
				//Set defined function name
				func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, String(GetName(func_name_)));

			} else if (func_name_=="") {
				// Auto set function name.
				func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, String(GetName(subgraph->func_name.str())));
			} else {
				LOG(WARNING)
						<< "The type of func name must be string.";
			}
		}else{
			if (func_name_!="") {
				//Set defined function name
				func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, String(GetName(func_name_)));
			} else if (func_name_=="") {
				// Auto set function name.
				func = WithAttr(std::move(func), tvm::attr::kGlobalSymbol, String(GetName(subgraph->func_name.str())));
			} else {
				LOG(WARNING)
						<< "The type of func name must be string.";
			}
		}
		func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(1));
		Expr new_call = Call(func, arguments, Attrs());
		if (device_type_ != 0) {
			new_call = on_device_(new_call, device_type_);
		}
		auto cast_type = raw_type.as<TensorTypeNode>()->dtype;
		new_call = Cast(new_call, cast_type);
		return new_call;
	}
	Expr VisitExpr_(const CallNode *call_node)
	final {
		Expr new_expr;
		Array<Expr> new_args;
		if (!InOpAttrs(call_node)) {
			/*
			 * If the call node's op is not in attrs_.
			 * set current subgraph to nullptr, and visit in the general way.
			 */
			current_ = nullptr;
			new_expr = ExprMutator::VisitExpr_(call_node);
		} else {
			if (current_) {
				/* If the call node belongs to a subgraph.*/
				if(visit_counter_[call_node]>1)
				{
					/*
					* if call_node is referenced by more than one operators.
					* Make a param and return.
					*/
					auto last = current_;
					current_ = std::make_shared<Subgraph>();
					current_->func_name << "fused";
					auto new_subgraph = current_;
					for (auto arg : call_node->args) {
						Expr new_arg = this->VisitExpr(arg);
						auto is_call = new_arg.as<CallNode>();
						if (!is_call || (is_call && !InOpAttrs(is_call))) {
							new_arg = Cast(new_arg, data_type_);
							new_arg = current_->GetOrAllocParam(new_arg,
										InferType(new_arg));
						}
						new_args.push_back(new_arg);
					}
					new_expr = Call(call_node->op, new_args,
							call_node->attrs, call_node->type_args);
					current_->func_name << "_"<< ConvertOpName(call_node->op.as<OpNode>()->name);
					new_expr = MakeFunc(new_subgraph, new_expr,
							call_node->checked_type());
					new_expr = Cast(new_expr, data_type_);
					new_expr = last->GetOrAllocParam(new_expr, InferType(new_expr));
				}else{
					/*
					* Otherwise, visit its args and get or allocate new param to subgraph for each arg
					* except of the call node with op which is in attrs_.
					*/
					for (auto arg : call_node->args) {
						Expr new_arg = this->VisitExpr(arg);
						auto is_call = new_arg.as<CallNode>();
						if (!is_call || !InOpAttrs(is_call)) {
							new_arg = Cast(new_arg, data_type_);
							new_arg = current_->GetOrAllocParam(new_arg,
										InferType(new_arg));
						}
						new_args.push_back(new_arg);
					}
					new_expr = Call(call_node->op, new_args,
							call_node->attrs, call_node->type_args);
					current_->func_name << "_";
					current_->func_name
							<< ConvertOpName(call_node->op.as<OpNode>()->name);
				}
			} else {
				/* If the call node doesn't belong to any subgraph.
				 * visit its args and get or allocate new param to subgraph for each arg
				 * except of the call node with op which is in attrs_.
				 * Afterwards, make a function.
				 */
				current_ = std::make_shared<Subgraph>();
				current_->func_name << "fused";
				auto new_subgraph = current_;
				for (auto arg : call_node->args) {
					Expr new_arg = this->VisitExpr(arg);
					auto is_call = new_arg.as<CallNode>();
					if (!is_call || (is_call && !InOpAttrs(is_call))) {
						new_arg = Cast(new_arg, data_type_);
						new_arg = current_->GetOrAllocParam(new_arg,
									InferType(new_arg));
					}
					new_args.push_back(new_arg);
				}
				new_expr = Call(call_node->op, new_args,
						call_node->attrs, call_node->type_args);
				current_->func_name << "_"<< ConvertOpName(call_node->op.as<OpNode>()->name);
				new_expr = MakeFunc(new_subgraph, new_expr,
						call_node->checked_type());

			}
		}
		return new_expr;
	}
	Expr VisitExpr_(const FunctionNode* op) final{
		if (op->HasNonzeroAttr(attr::kPrimitive)) {
			return GetRef<Function>(op);
		}
		return ExprMutator::VisitExpr_(op);
	}
	Expr VisitExpr(const Expr &expr) {
		// Store the current subgraph to maintain for the next branch.
		auto subgraph = current_;
		auto it = this->memo_.find(expr);
		if (it != this->memo_.end()) {
			return it->second;
		} else {
			Expr new_expr = ExprFunctor::VisitExpr(expr);
			memo_[expr] = new_expr;
			current_ = subgraph;
			return new_expr;
		}
	}

private:
	// The pointer point to current subgraph.
	std::shared_ptr<Subgraph> current_ { nullptr };
	// The list of the specified operators.
	const Array<Array<ObjectRef>> op_attrs_;
	// The function name.
	const String func_name_;
	// The compiler.
	const String compiler_;
	// The annotated device type for subgraphs.
	int device_type_;
	// The data type which the function computes in.
	const DataType data_type_;
	// Internal visiting counter
	std::unordered_map<const Object*, size_t> visit_counter_;
	// Count the function name.
	std::unordered_map<std::string, size_t> name_map_;

}
;
Expr PartitionGraphInUnorder(const Array<Array<ObjectRef>> op_attrs, const String func_name, const String compiler, const int device_type,
		const DataType data_type, const Expr expr) {
	return partition::GraphPartitionerInUnorder(op_attrs, func_name, compiler, device_type,
			data_type).Partition(expr);
}
} // namespace partition

namespace transform {
Pass PartitionGraphInUnorder(Array<Array<ObjectRef>> op_attrs,
		String func_name, String compiler, int device_type, DataType data_type) {
	runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
			[=](Function f, IRModule m, PassContext pc) {
				return Downcast<Function>(partition::PartitionGraphInUnorder(op_attrs, func_name, compiler,
								device_type, data_type, f));
			};
	return CreateFunctionPass(pass_func, 1, "PartitionGraphInUnorder", {});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraphInUnorder")
		.set_body_typed(PartitionGraphInUnorder);
}  // namespace transform
}  // namespace relay
}  // namespace tvm

