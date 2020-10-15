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
 * \file partition_graph_by_expr.cc
 * \brief Pass to partition a Relay graph to subgraph with Relay expression.
 */
#include "partition_graph_util.h"
namespace tvm {
namespace relay {
namespace partition {
class ExprComparator: public ExprFunctor<bool(const Expr&, const Expr&)> {
public:
	struct Subgraph {
	public:
		// \brief The expr and sxpr of the boundaries in this subgraph.
		std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual> boundaries;
		// \brief The middle expr in this subgraph.
		std::unordered_set<Expr, ObjectHash, ObjectEqual> nodes;
		// \brief The input arguments of this subgraph.
		std::vector<std::pair<Expr, Var>> args;
		// \brief The old expr and new expr of the output in this subgraph.
		std::pair<Expr, Expr> output;
		// \brief The name of the composite function.
		std::stringstream func_name;
		// Get a new parameter or allocate an old one.
		Var GetOrAllocParam(const Expr &expr, const Type &type) {
			for (auto arg : args) {
				if (expr.same_as(arg.first))
					return arg.second;
			}
			// create a new parameter.
			std::ostringstream os;
			os << "p" << args.size();
			auto var = Var(os.str(), type);
			args.push_back( { expr, var });
			return var;
		}
		bool AddBoundary(const Expr &expr, const Expr &sexpr) {
			// If sexpr is repeated in boundaries
			for (auto node : boundaries) {
				if (node.second == sexpr) {
					if (node.first != expr) {
						auto it = boundaries.find(expr);
						if (it != boundaries.end()) {
							node.second = it->second;
							it->second = sexpr;
						} else {
							return false;
						}
					} else {
						return true;
					}
				}
			}
			// If sexpr is not repeated in boundaries, add the new node
			boundaries[expr] = sexpr;
			return true;

		}
	};
	std::shared_ptr<Subgraph> Compare(const Expr &expr, const Expr &sexpr) {
		sexpr_ = sexpr;
		if (VisitExpr(expr, sexpr)) {
			return current_;
		} else {
			return nullptr;
		}
	}
	bool VisitExpr(const Expr &expr, const Expr &sexpr) final {
		auto it = memo_.find(expr);
		if (it != memo_.end()) {
			return it->second;
		} else {
			bool is_in_subgraph;
			if (sexpr.as<VarNode>()) {
				is_in_subgraph = current_->AddBoundary(expr, sexpr);
			}
			else{
				is_in_subgraph = ExprFunctor::VisitExpr(expr, sexpr);
			}
			memo_[expr] = is_in_subgraph;
			return is_in_subgraph;
		}
	}
	bool VisitExpr_(const VarNode *var, const Expr &sexpr) final {
		auto svar = sexpr.as<VarNode>();
		if (svar)
			return current_->AddBoundary(GetRef<Expr>(var), sexpr);
		else
			return false;
	}

	bool VisitExpr_(const GlobalVarNode* op, const Expr &sexpr) final {
		auto sglobalvar = sexpr.as<GlobalVarNode>();
		if(sglobalvar)
			return current_->AddBoundary(GetRef<Expr>(sglobalvar), sexpr);
		return false;
	}
	bool VisitExpr_(const ConstantNode *constant, const Expr &sexpr) final {
		auto sconstant = sexpr.as<ConstantNode>();

		if (sconstant) {
			current_->nodes.insert(GetRef<Expr>(constant));
			return true;
		} else {
			return false;
		}
	}
	bool VisitExpr_(const CallNode *call, const Expr &sexpr) final {
		auto scall = sexpr.as<CallNode>();
		if (!scall)
			return false;
		if (visit_counter_[call] > 1 && GetRef<Expr>(call)!= sexpr_)
			return false;
		if (!VisitExpr(call->op, scall->op)) // here we compare call nodes' ops instead of its attributes.
			return false;
		if(!AttrsComparor::CompareNonDefault(scall->attrs, call->attrs))
			return false;
		current_->nodes.insert(call->op);
		current_->func_name<<"_"<< ConvertOpName(call->op.as<OpNode>()->name);
		size_t equal_count = 0;
		size_t idx = 0;
		for (size_t i = 0; i < scall->args.size(); i++) {
			for (size_t j = idx; j < call->args.size(); j++) {
				if (VisitExpr(call->args[i], scall->args[j])) {
					equal_count += 1;
					idx = j + 1;
					break;
				}
			}
		}
		if (equal_count == scall->args.size()) {
			current_->nodes.insert(GetRef<Expr>(call));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const OpNode *op, const Expr &sexpr) final {
		auto sop = sexpr.as<OpNode>();
		if (op == sop){
			return true;
		}else
			return false;
	}
	bool VisitExpr_(const TupleNode *tuple, const Expr &sexpr) final {
		auto stuple = sexpr.as<TupleNode>();
		if (!stuple)
			return false;
		size_t equal_count = 0;
		size_t idx = 0;
		for (size_t i = 0; i < stuple->fields.size(); i++) {
			for (size_t j = idx; j < tuple->fields.size(); j++) {
				if (VisitExpr(tuple->fields[i], stuple->fields[j])) {
					equal_count += 1;
					idx = j + 1;
					break;
				}
			}
		}
		if (equal_count == stuple->fields.size()) {
			current_->nodes.insert(GetRef<Expr>(tuple));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const FunctionNode *func, const Expr &sexpr) final {
		auto sfunc = sexpr.as<FunctionNode>();
		if (!sfunc)
			return false;
		if (VisitExpr(func->body, sfunc->body)) {
			current_->nodes.insert(GetRef<Expr>(func));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const LetNode *let, const Expr &sexpr) final {
		auto slet = sexpr.as<LetNode>();
		if (!slet)
			return false;
		if (VisitExpr(let->value, slet->value)
				&& VisitExpr(let->body, slet->body)) {
			current_->nodes.insert(GetRef<Expr>(let));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const IfNode *If, const Expr &sexpr) final {
		auto sIf = sexpr.as<IfNode>();
		if (!sIf)
			return false;
		if (VisitExpr(If->cond, sIf->cond)
				&& VisitExpr(If->true_branch, sIf->true_branch)
				&& VisitExpr(If->false_branch, sIf->false_branch)) {
			current_->nodes.insert(GetRef<Expr>(If));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const TupleGetItemNode *tgi, const Expr &sexpr) final {
		auto stgi = sexpr.as<TupleGetItemNode>();
		if (!stgi)
			return false;
		if (VisitExpr(tgi->tuple, stgi->tuple)) {
			current_->nodes.insert(GetRef<Expr>(tgi));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const RefCreateNode *refcreate, const Expr &sexpr) final {
		auto srefcreate = sexpr.as<RefCreateNode>();
		if (!srefcreate)
			return false;
		if (VisitExpr(refcreate->value, srefcreate->value)) {
			current_->nodes.insert(GetRef<Expr>(refcreate));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const RefReadNode *refread, const Expr &sexpr) final {
		auto srefread = sexpr.as<RefReadNode>();
		if (!srefread)
			return false;
		if (VisitExpr(refread->ref, srefread->ref)) {
			current_->nodes.insert(GetRef<Expr>(refread));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const RefWriteNode *rewrite, const Expr &sexpr) final {
		auto srewrite = sexpr.as<RefWriteNode>();
		if (!srewrite)
			return false;
		if (VisitExpr(rewrite->ref, srewrite->ref)
				&& VisitExpr(rewrite->value, srewrite->value)) {
			current_->nodes.insert(GetRef<Expr>(rewrite));
			return true;
		} else {
			return false;
		}

	}
	bool VisitExpr_(const ConstructorNode* op, const Expr &sexpr) final {
		auto sconstructor = sexpr.as<ConstructorNode>();
		if(!sconstructor)
			return false;
		return true;
	}

	bool VisitExpr_(const MatchNode* op, const Expr &sexpr) final {
		auto smatch = sexpr.as<MatchNode>();
		if(!smatch)
			return false;
		return VisitExpr(op->data, smatch->data);
	}

	explicit ExprComparator(
			const std::unordered_map<const Object*, size_t> visit_counter) :
		visit_counter_(visit_counter) {
		current_ = std::make_shared<Subgraph>();
		current_->func_name << "fused";
	}
private:
	// Internal visiting counter.
	std::unordered_map<const Object*, size_t> visit_counter_;
	// The output subgraph.
	std::shared_ptr<Subgraph> current_;
	// Sub expr.
	Expr sexpr_;
	/*! \brief Internal map used for memoization. */
	std::unordered_map<Expr, bool, ObjectHash, ObjectEqual> memo_;
};

class GraphPartitionerByExpr: public ExprMutator{
public:
	explicit GraphPartitionerByExpr(const IRModule module, const Expr subexpr,
			const String func_name, const String compiler, const int device_type,
			const DataType &data_type) :
			module_(module), subexpr_(subexpr), func_name_(func_name), compiler_(compiler), device_type_(
					device_type), data_type_(data_type) {
	}
	Expr Partition(Expr expr) {

		auto visitor = RefVisitor();
		visit_counter_ = visitor.GetCounter(expr);
		return ExprMutator::Mutate(expr);
	}
	std::shared_ptr<ExprComparator::Subgraph> Match(const Expr &expr) {
		return ExprComparator(visit_counter_).Compare(expr, subexpr_);
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
			std::shared_ptr<ExprComparator::Subgraph> &subgraph, Expr &body,
			const Type raw_type) {
	    // If the function has no call, it is not a primitive function.
	    struct HasCallVisitor : ExprVisitor {
	      bool has_call = false;
	      void VisitExpr_(const CallNode* op) final {
	        has_call = true;
	      }
	    } visitor;
	    visitor(body);
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
		//func = WithAttr(std::move(func), attr::kPattern, tvm::Integer(pattern_));
		func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(visitor.has_call));
		Expr new_call = Call(func, arguments, Attrs());
		if (device_type_ != 0) {
			new_call = on_device_(new_call, device_type_);
		}
		auto cast_type = raw_type.as<TensorTypeNode>()->dtype;
		new_call = Cast(new_call, cast_type);
		return new_call;
	}
	Expr VisitExpr(const Expr &expr) {
		std::shared_ptr<ExprComparator::Subgraph> last_subgraph = current_;
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
				// If expr matchs with subexpr_,
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
				// If expr doesn't match with subexpr_,
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
	std::shared_ptr<ExprComparator::Subgraph> current_ { nullptr };
	// Module
	const IRModule module_;
	// A subgraph we search.
	const Expr subexpr_;
	// The function name.
	const String func_name_;
	// The function name.
	const String compiler_;
	// The annotated device type for subgraphs.
	const int device_type_;
	// The data type which the function computes in.
	const DataType data_type_;
	// Internal visiting counter
	std::unordered_map<const Object*, size_t> visit_counter_;
	// Count the function name.
	std::unordered_map<std::string, size_t> name_map_;
};
Expr PartitionGraphByExpr(const Array<Array<ObjectRef>> func_list, const int device_type, const DataType data_type, const Expr expr,
		const IRModule model) {
	Expr new_expr = expr;
	for(size_t i=0; i< func_list.size();i++)
	{
		auto func = func_list[i];
		auto len = func.size();
		if (len == 0){
			LOG(WARNING)
					<< i<<"-th func in func_list is null.";
			continue;
		}

		Expr subexpr;
		String func_name = "";
		String compiler = "";
		if (len >=1 && func[0].defined()){
			subexpr = Downcast<Expr>(func[0]);
		}else{
			LOG(WARNING)
					<< "The expr of "<<i<<"-th func is not defined.";
			continue;
		}
		//if(len >=2 && func[1].defined())
		//	pattern = int(Downcast<Integer>(func[1]));
		if(len >=2 && func[1].defined())
			func_name = std::string(Downcast<String>(func[1]));
		if(len >=3 && func[2].defined())
			compiler = std::string(Downcast<String>(func[2]));
		if(len >=4){
			LOG(WARNING)
					<< "The list of "<<i<<"-th func is too long. Can only define the subgraph to fused and the function name.";
		}
		if (auto f = subexpr.as<FunctionNode>()){
			new_expr = partition::GraphPartitionerByExpr(model, f->body, func_name, compiler, device_type,
						data_type).Partition(new_expr);
		}
		else{
			new_expr = partition::GraphPartitionerByExpr(model, subexpr, func_name, compiler, device_type,
						data_type).Partition(new_expr);
		}
	}
	return new_expr;
}
} // namespace partition

namespace transform {
Pass PartitionGraphByExpr(Array<Array<ObjectRef>> func_list, int device_type,
		DataType data_type) {
	runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
			[=](Function f, IRModule m, PassContext pc) {

				return Downcast<Function>(partition::PartitionGraphByExpr(func_list, device_type,
								data_type, f, m));
			};
	return CreateFunctionPass(pass_func, 1, "PartitionGraphByExpr", {});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraphByExpr")
.set_body_typed(PartitionGraphByExpr);
}  // namespace transform
}  // namespace relay
}  // namespace tvm
