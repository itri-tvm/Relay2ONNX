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
 * \file fuse_funcs.cc
 * \brief Fuse multiple functions to one function.
 */
#include "partition_graph_util.h"
namespace tvm{
namespace relay{
namespace partition {
class FuseFuncsMutator: public ExprMutator {
public:
	explicit FuseFuncsMutator(){}
	Expr Transform(Expr expr) {
		auto visitor = RefVisitor();
		visit_counter_ = visitor.GetCounter(expr);
		return ExprMutator::Mutate(expr);
	}
	struct Subgraph {
	public:
		// \brief Function name.
		std::ostringstream func_name;
		// \brief The map mapping var to body.
		std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual> var_body_map_;
		// \brief The map mapping param to new param.
		std::unordered_map<Expr, Var, ObjectHash, ObjectEqual> replace_params_;

		Array<Expr> new_args;
		Array<Var> new_params;
		Expr new_body;
	};
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
	Expr VisitExpr_(const VarNode* op) {
		if (current_){
			auto it = current_->var_body_map_.find(GetRef<Var>(op));
			if(it != current_->var_body_map_.end()){
				return this->Mutate(it->second);
			}
			else{
				std::ostringstream os;
				os << "p" << current_->replace_params_.size();
				Var var = Var(os.str(), op->type_annotation);
				current_->replace_params_[GetRef<Var>(op)] = var;
				return var;
			}
		}else{
			return GetRef<Var>(op);
		}
	}
	Expr VisitExpr_(const CallNode* call_node) {
		if(auto f = call_node->op.as<FunctionNode>()){
			if(!current_)
				current_ = std::make_shared<Subgraph>();
			auto current = current_;
			for(size_t i=0; i<call_node->args.size(); i++){
				auto arg_call = call_node->args[i].as<CallNode>();
				if( arg_call && visit_counter_[arg_call]==1 && arg_call->op.as<FunctionNode>()){
					auto arg_f = arg_call->op.as<FunctionNode>();
					current->var_body_map_[f->params[i]] = arg_f->body;
				}
			}
			if(current->var_body_map_.size() > 0){
				current->func_name<<std::string(Downcast<String>(f->attrs->dict[attr::kName]));
				current->new_body = this->Mutate(f->body);
				for(size_t i=0; i<call_node->args.size(); i++){
					auto arg_call = call_node->args[i].as<CallNode>();
					if(arg_call && visit_counter_[arg_call]==1 && arg_call->op.as<FunctionNode>()){
						const FunctionNode* arg_f = arg_call->op.as<FunctionNode>();
						current->func_name<<std::string(Downcast<String>(arg_f->attrs->dict[attr::kName])).substr(5);
						for(size_t j=0; j<arg_call->args.size(); j++){
							current->new_params.push_back(current_->replace_params_[arg_f->params[j]]);
							current_ = nullptr;
							current->new_args.push_back(this->Mutate(arg_call->args[j]));
							current_ = current;
						}
					}
					else{
						current->new_params.push_back(current->replace_params_[f->params[i]]);
						current_ = nullptr;
						current->new_args.push_back(this->Mutate(call_node->args[i]));
						current_ = current;
					}
				}
				auto new_f = Function(current->new_params, current->new_body, f->ret_type, {});
				new_f = WithAttr(new_f, attr::kName, tvm::String(current_->func_name.str()));
				new_f = WithAttr(new_f, attr::kPrimitive, tvm::Integer(1));
				return Call(new_f, current->new_args, call_node->attrs, call_node->type_args);
			}
			else{
				current_ = nullptr;
			}
		}
		return ExprMutator::VisitExpr_(call_node);

	}
private:
	// The pointer point to current subgraph.
	std::shared_ptr<Subgraph> current_ { nullptr };
	// Internal visiting counter
	std::unordered_map<const Object*, size_t> visit_counter_;
	// Check if the current node in the body of function node.
	bool in_body_{false};
};
Expr FuseFuncs(int limit_num, Expr expr){
	Expr curr_expr;
	Expr next_expr = expr;
	if (limit_num <= 0)
	{
		do{
			curr_expr = next_expr;
			next_expr = FuseFuncsMutator().Transform(curr_expr);
		}while(next_expr != curr_expr);
	}
	else{
		for(int i=0; i<limit_num; i++)
			next_expr = FuseFuncsMutator().Transform(next_expr);
	}
	return next_expr;
}
} // namespace partition
namespace transform{
Pass FuseFuncs(int limit_num) {
	runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
			[=](Function f, IRModule m, PassContext pc) {
				return Downcast<Function>(partition::FuseFuncs(limit_num, f));
			};
	return CreateFunctionPass(pass_func, 1, "FuseFuncs", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FuseFuncs").set_body_typed(FuseFuncs);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
