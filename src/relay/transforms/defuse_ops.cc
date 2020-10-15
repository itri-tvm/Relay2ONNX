#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

class DefuseMutator: public ExprMutator{
public:

	Expr VisitExpr_(const VarNode* op) final{
		auto it = args_map_.find(GetRef<Var>(op));
		if(it != args_map_.end()){
			return it->second;
		}

		return ExprMutator::VisitExpr_(op);
	}
	Expr VisitExpr_(const CallNode* call_node) final{
		auto f = call_node->op.as<FunctionNode>();
		if (f && f->HasNonzeroAttr(attr::kPrimitive))
		{
			for(size_t i = 0; i< call_node->args.size();i++){
				args_map_[f->params[i]] = ExprMutator::VisitExpr(call_node->args[i]);
			}
			auto new_expr = ExprMutator::VisitExpr(f->body);
			return new_expr;
		}
		else {
			return ExprMutator::VisitExpr_(call_node);
		}
	}
	explicit DefuseMutator(){}

	Expr Transform(Expr expr) {
		return ExprMutator::Mutate(expr);
	};
private:
	std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> args_map_;
};
Expr DefuseOps(const Expr &expr) {
	return DefuseMutator().Transform(expr);
}

namespace transform {
Pass DefuseOps() {
	runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
			[=](Function f, IRModule m, PassContext pc) {
				return Downcast<Function>(
						DefuseOps(f));
			};
	return CreateFunctionPass(pass_func, 1, "DefuseOps",{});
}

TVM_REGISTER_GLOBAL("relay._transform.DefuseOps")
.set_body_typed(DefuseOps);
}  // namespace transform
}  // namespace relay
}  // namespace tvm
