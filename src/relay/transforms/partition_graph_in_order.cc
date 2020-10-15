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
 * \file partition_graph_in_order.cc
 * \brief Pass to partition a Relay graph to subgraph with ordered operators.
 */
#include "partition_graph_util.h"
namespace tvm {
namespace relay {
namespace partition {
class OpComparator: public ExprFunctor<bool(const Expr&, const size_t&)> {
public:
	struct Subgraph {
	public:
		// \brief The expr and sxpr of the boundaries in this subgraph.
		std::unordered_set<Expr, ObjectHash, ObjectEqual> boundaries;
		// \brief The middle expr in this subgraph.
		std::unordered_set<Expr, ObjectHash, ObjectEqual> nodes;
		// \brief The input arguments of this subgraph.
		std::vector<std::pair<Expr, Var>> args;
		// \brief The old expr and new expr of the output in this subgraph.
		std::pair<Expr, Expr> output;
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
			args.push_back({ expr, var });
			return var;
		}
	};
	bool FindOp(Array<Array<ObjectRef>> op_attrs,const OpNode* op, Attrs tar_attrs){
		for (size_t i = 0; i < op_attrs.size();i++)
		{
			if( Downcast<String>(op_attrs[i][0]) == op->name){
				if (!op_attrs[i][1].defined())
					return true;
				else{
					auto attrs = Downcast<Attrs>(op_attrs[i][1]);
					LOG_IF(FATAL, op->attrs_type_index!=attrs->type_index())
							<< "The input attrs type is not consistant with the op name, \""
							<< op->name
							<< "\" .";
					if (AttrsComparor::CompareNonDefault(attrs, tar_attrs))
						return true;
				}
			}
		}
		return false;
	}
	bool InOpAttrs(const CallNode *call_node, size_t order) {

		auto op = call_node->op.as<OpNode>();
		if(!op)
			return false;

		if (!op_attrs_[order].defined()) {
			// If op name is not defined, return true.
			if((include_.empty()|| FindOp(include_, op, call_node->attrs)) &&
					(exclude_.empty()|| !FindOp(exclude_, op, call_node->attrs)))
				return true;
			else
				return false;
		}
		if (Downcast<String>(op_attrs_[order][0]) == op->name) {
			if (op_attrs_[order][1].defined()) {
				auto attrs = Downcast<Attrs>(op_attrs_[order][1]);
				LOG_IF(FATAL, op->attrs_type_index!=attrs->type_index())
						<< "The input attrs type is not consistant with the op name, \""
						<< op->name
						<< "\" .";
				return AttrsComparor::CompareNonDefault(attrs, call_node->attrs);
			} else{
				return true;
			}
		}

		return false;
	}

	std::shared_ptr<Subgraph> Compare(const Expr &expr) {

		if (VisitExpr(expr, 0)) {
			return current_;
		} else {
			return nullptr;
		}
	}

	bool VisitExpr(const Expr &expr, const size_t &order) final {
		auto it = memo_.find(expr);
		if (it != memo_.end()) {
			return it->second;
		} else {
			bool is_in_subgraph = ExprFunctor::VisitExpr(expr, order);
			memo_[expr] = is_in_subgraph;
			return is_in_subgraph;
		}
	}


	bool VisitExpr_(const VarNode *op, const size_t &order) final {
		if (order == op_attrs_.size()) {
			current_->boundaries.insert(GetRef<Expr>(op));
			return true;
		} else {
			return false;
		}
	}
	bool VisitExpr_(const GlobalVarNode* op, const size_t &order) final {
		if (order == op_attrs_.size()) {
			current_->boundaries.insert(GetRef<Expr>(op));
			return true;
		} else {
			return false;
		}
	}

	bool VisitExpr_(const ConstantNode *op, const size_t &order) final {
		if (order == op_attrs_.size()) {
			current_->boundaries.insert(GetRef<Expr>(op));
			return true;
		} else {
			return false;
		}
	}
	std::string ConvertOpName(std::string text) {
		for (size_t i = 0; i < text.length(); i++) {
			if (text[i] == '.')
				text[i] = '_';
		}
		return text;
	}
	bool VisitExpr_(const CallNode *op, const size_t &order) final {


		if (order == op_attrs_.size()) {
			current_->boundaries.insert(GetRef<Expr>(op));
			return true;
		}
		if (order > 0 && visit_counter_[op] > 1)
			return false;
		if (!InOpAttrs(op, order)) // here we compare call nodes' ops instead of its attributes.
			return false;
		current_->nodes.insert(op->op);
		size_t idx = 0;
		for (size_t i = idx; i < op->args.size(); i++) {
			if (VisitExpr(op->args[i], order + 1)) {
				current_->nodes.insert(GetRef<Expr>(op));
				if (op->op.as<OpNode>()) {
					current_->func_name << "_"<< ConvertOpName(op->op.as<OpNode>()->name);
				}
				return true;
			}
		}
		return false;
	}
	bool VisitExpr_(const OpNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const TupleNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const FunctionNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const LetNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const IfNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const TupleGetItemNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const RefCreateNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const RefReadNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const RefWriteNode *op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const ConstructorNode* op, const size_t &order) final {
		return false;
	}
	bool VisitExpr_(const MatchNode* op, const size_t &order) final {
		return false;
	}

	explicit OpComparator(const Array<Array<ObjectRef>> op_attrs,
			const Array<Array<ObjectRef>> include,
			const Array<Array<ObjectRef>> exclude,
			const std::unordered_map<const Object*, size_t> visit_counter) :
		op_attrs_(op_attrs), include_(include), exclude_(exclude), visit_counter_(visit_counter) {
		current_ = std::make_shared<Subgraph>();
		current_->func_name << "fused";

	}
private:
	std::shared_ptr<Subgraph> current_;
	// The list of the specified operator.
	Array<Array<ObjectRef>> op_attrs_;
	// The names of the including operator.
	const Array<Array<ObjectRef>> &include_;
	// The names of the excluding operator.
	const Array<Array<ObjectRef>> exclude_;
	// Internal visiting counter
	std::unordered_map<const Object*, size_t> visit_counter_;
	/*! \brief Internal map used for memoization. */
	std::unordered_map<Expr, bool, ObjectHash, ObjectEqual> memo_;
};

class GraphPartitionerInOrder: public ExprMutator {
public:
	explicit GraphPartitionerInOrder(const IRModule module, Array<Array<ObjectRef>> op_attrs, const Array<Array<ObjectRef>> include, const Array<Array<ObjectRef>> exclude,
			const String func_name, const String compiler, const int device_type,
			const DataType data_type):module_(module), op_attrs_(op_attrs), include_(include), exclude_(exclude), func_name_(func_name), compiler_(compiler),
					device_type_(device_type), data_type_(data_type){
	}
	std::shared_ptr<OpComparator::Subgraph> Match(const Expr &expr) {

		return OpComparator(op_attrs_, include_, exclude_, visit_counter_).Compare(expr);
	}
	Expr Partition(Expr expr) {

		auto visitor = RefVisitor();
		visit_counter_ = visitor.GetCounter(expr);
		return ExprMutator::Mutate(expr);
	}
	// Get the subgraph by the id.
	bool IsMiddleNode(const Expr expr) {
		if (!current_)
			return false;
		auto it = current_->nodes.find(expr);
		if (it != current_->nodes.end()) {
			return true;
		} else {
			return false;
		}
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
			std::shared_ptr<OpComparator::Subgraph> &subgraph, Expr &body,
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

	Expr VisitExpr(const Expr &expr) {
		std::shared_ptr<OpComparator::Subgraph> last_subgraph = current_;
		auto it = this->memo_.find(expr);
		if (it != this->memo_.end()) {
			// If expr was visited before,
			auto new_expr = it->second;
			if (last_subgraph && !new_expr.as<OpNode>()
					&& !new_expr.as<FunctionNode>()) {
				// check if expr is a boundary of a subgraph, make a param for the subgraph.
				new_expr = Cast(new_expr, data_type_);
				// Get or allocate a parameter.
				return last_subgraph->GetOrAllocParam(new_expr,
						InferType(new_expr));
			}
			return new_expr;
		} else {
			Expr new_expr;
			auto new_subgraph = Match(expr);
			if (IsMiddleNode(expr)) {
				new_expr = ExprFunctor::VisitExpr(expr);
				memo_[expr] = new_expr;
			} else if (last_subgraph && new_subgraph) {
				// Check if expr is a boundaries of a graph.
				// If expr VisitExpres with subexpr_,
				// Record the nodes in the new subgraph.
				current_ = new_subgraph;
				// Tarverse the expr.
				new_expr = ExprFunctor::VisitExpr(expr);
				// Make a new device function.CallNode(Op(add)
				new_expr = MakeFunc(new_subgraph, new_expr,
						expr->checked_type());
				// Get or allocate a parameter.
				memo_[expr] = new_expr;
				new_expr = last_subgraph->GetOrAllocParam(new_expr,
						InferType(new_expr));

			} else if (last_subgraph && !new_subgraph) {
				// If expr doesn't VisitExpr with subexpr_,
				// Tarverse the expr.
				current_ = nullptr;
				new_expr = ExprFunctor::VisitExpr(expr);
				memo_[expr] = new_expr;
				new_expr = Cast(new_expr, data_type_);
				// Get or allocate a parameter.
				new_expr = last_subgraph->GetOrAllocParam(new_expr,
							InferType(new_expr));

			} else if (!last_subgraph && new_subgraph) {
				current_ = new_subgraph;
				// Tarverse the expr.
				new_expr = ExprFunctor::VisitExpr(expr);
				// Make a new device function.
				new_expr = MakeFunc(new_subgraph, new_expr,
						expr->checked_type());
				memo_[expr] = new_expr;
			} else {
				// Otherwise, tarverse the expr.
				current_ = nullptr;
				new_expr = ExprFunctor::VisitExpr(expr);
				memo_[expr] = new_expr;
			}

			current_ = last_subgraph;
			return new_expr;
		}

	}
	Expr VisitExpr_(const FunctionNode* op) final{
		if (op->HasNonzeroAttr(attr::kPrimitive)) {
				return GetRef<Function>(op);
		}
		return ExprMutator::VisitExpr_(op);
	}


private:
	// A set stores subgraphs finded in the graph.
	std::shared_ptr<OpComparator::Subgraph> current_ { nullptr };
	// Module
	const IRModule module_;
	// The list of the specified operators.
	const Array<Array<ObjectRef>> op_attrs_;
	// The names of the including operators.
	const Array<Array<ObjectRef>> include_;
	// The names of the excluding operators.
	const Array<Array<ObjectRef>> exclude_;
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
};
Expr PartitionGraphInOrder(Array<Array<ObjectRef>> op_attrs, Array<Array<ObjectRef>> include, Array<Array<ObjectRef>> exclude,
		const String func_name, const String compiler, const int device_type,
		const DataType data_type, const Expr expr, const IRModule module) {
	return partition::GraphPartitionerInOrder(module, op_attrs, include, exclude, func_name, compiler,
			device_type, data_type).Partition(expr);
}
} // namespace partition

namespace transform {
Pass PartitionGraphInOrder(Array<Array<ObjectRef>> op_attrs, Array<Array<ObjectRef>> include, Array<Array<ObjectRef>> exclude,
		String func_name, String compiler, int device_type, DataType data_type) {
	runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
			[=](Function f, IRModule m, PassContext pc) {
				return Downcast<Function>(partition::PartitionGraphInOrder(op_attrs, include, exclude, func_name, compiler,
								device_type, data_type, f, m));
			};
	return CreateFunctionPass(pass_func, 1, "PartitionGraphInOrder", {});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraphInOrder").set_body_typed(
		PartitionGraphInOrder);
}  // namespace transform
}  // namespace relay
}  // namespace tvm
