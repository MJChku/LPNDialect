#include "LPN/Dialect/LPNDialect.h"
#include "LPN/Dialect/LPNOps.h"
#include "LPN/Dialect/LPNTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

using namespace mlir;
using namespace mlir::lpn;

#include "GuardSynthesis.h"

namespace {

class Compiler {
public:
  Compiler() : ir(context), module(std::make_unique<llvm::Module>("lpn-native", context)) {
    i1 = llvm::Type::getInt1Ty(context);
    i32 = llvm::Type::getInt32Ty(context);
    i64 = llvm::Type::getInt64Ty(context);
    f64 = llvm::Type::getDoubleTy(context);
    ptr = llvm::PointerType::getUnqual(context);
    keyType = llvm::StructType::get(context, {i64, i64});
    declareRuntime();
  }

  void compile(ModuleOp source) {
    SmallVector<NetOp> nets;
    source.walk([&](NetOp net) { nets.push_back(net); });
    if (nets.size() != 1)
      fail("expected exactly one lpn.net");
    NetOp net = nets.front();
    for (PlaceOp place : net.getOps<PlaceOp>()) {
      if (place.getCapacity())
        fail("place capacity is not implemented by the native runtime");
      std::string name = place.getSymName().str();
      if (placeIds.count(name))
        fail("duplicate place: " + name);
      placeIds[name] = places.size();
      places.push_back({name, static_cast<int64_t>(
                                   place.getInitialTokens().value_or(0))});
    }
    source.walk([&](func::FuncOp function) {
      if (!function.getBody().hasOneBlock())
        fail("native compilation requires single-block helper functions");
      auto name = function.getSymName().str();
      if (helpers.count(name))
        fail("duplicate helper function: " + name);
      std::vector<llvm::Type *> inputs;
      for (mlir::Type input : function.getFunctionType().getInputs())
        inputs.push_back(lowerType(input));
      std::vector<llvm::Type *> outputs;
      for (mlir::Type output : function.getFunctionType().getResults())
        outputs.push_back(lowerType(output));
      llvm::Type *result = llvm::Type::getVoidTy(context);
      if (outputs.size() == 1)
        result = outputs.front();
      else if (outputs.size() > 1)
        result = llvm::StructType::get(context, outputs);
      auto *type = llvm::FunctionType::get(result, inputs, false);
      helpers[name] = llvm::Function::Create(type, llvm::GlobalValue::InternalLinkage,
                                              "helper_" + name, module.get());
    });
    for (TransitionOp transition : net.getOps<TransitionOp>()) {
      if (!transition.getBody().hasOneBlock())
        fail("native compilation requires single-block transitions");
      std::string name = transition.getSymName().str();
      transitionNames.push_back(name);
      auto *type = llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
      transitions.push_back(llvm::Function::Create(
          type, llvm::GlobalValue::InternalLinkage,
          "transition_" + std::to_string(transitions.size()) + "_" + name,
          module.get()));
    }
    source.walk([&](func::FuncOp function) { emitHelper(function); });
    unsigned index = 0;
    for (TransitionOp transition : net.getOps<TransitionOp>()) {
      GuardPlan plan = GuardInference().analyze(transition);
      emitEnabled(transition, plan, index);
      emitTransition(transition, transitions[index++]);
    }
    emitMain();
    if (llvm::verifyModule(*module, &llvm::errs()))
      fail("generated LLVM IR failed verification");
  }

  llvm::Module &getModule() { return *module; }

private:
  struct PlaceSpec {
    std::string name;
    int64_t initial;
  };

  [[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
  }

  llvm::Value *value(mlir::Value source) {
    auto it = values.find(source);
    if (it == values.end())
      fail("use of an unmapped SSA value");
    return it->second;
  }

  void put(mlir::Value source, llvm::Value *lowered) {
    values[source] = lowered;
  }

  llvm::ConstantInt *integer(int64_t number) {
    return llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(i64),
                                  static_cast<uint64_t>(number));
  }

  llvm::Type *lowerType(mlir::Type type) {
    if (type.isInteger(1)) return i1;
    if (type.isInteger(64) || type.isIndex()) return i64;
    if (type.isF64()) return f64;
    if (isa<TokenType, PlaceType, ArrayType>(type)) return ptr;
    if (isa<KeyType>(type)) return keyType;
    fail("unsupported MLIR type for native compilation");
  }

  llvm::FunctionCallee runtime(StringRef name, llvm::Type *result,
                                std::initializer_list<llvm::Type *> args) {
    return module->getOrInsertFunction(name, llvm::FunctionType::get(result, args, false));
  }

  void declareRuntime() {
    placeFn = runtime("lpn_place", ptr, {i64});
    takeFn = runtime("lpn_take", ptr, {ptr});
    peekFn = runtime("lpn_peek", ptr, {ptr});
    guardFn = runtime("lpn_guard", llvm::Type::getVoidTy(context), {i1});
    emitFn = runtime("lpn_emit", llvm::Type::getVoidTy(context), {ptr, ptr, f64});
    countFn = runtime("lpn_count", i64, {ptr});
    createFn = runtime("lpn_token_create", ptr, {});
    cloneFn = runtime("lpn_token_clone", ptr, {ptr});
    setFn = runtime("lpn_token_set", ptr, {ptr, i64, i64, i64});
    getFn = runtime("lpn_token_get", i64, {ptr, i64, i64});
    // Let LLVM remove unused predicate calculations from transition bodies.
    for (auto callee : {placeFn, countFn, getFn}) {
      auto *function = llvm::cast<llvm::Function>(callee.getCallee());
      function->setOnlyReadsMemory();
      function->setDoesNotThrow();
      function->addFnAttr(llvm::Attribute::WillReturn);
    }
    arrayCreateFn = runtime("lpn_array_create", ptr, {i64, ptr});
    arrayStaticFn = runtime("lpn_array_static", ptr, {i64, ptr});
    arrayAllocFn = runtime("lpn_array_alloc", ptr, {i64, i64});
    arrayGetFn = runtime("lpn_array_get", i64, {ptr, i64});
    arraySetFn = runtime("lpn_array_set", ptr, {ptr, i64, i64});
    arrayLenFn = runtime("lpn_array_len", i64, {ptr});
    checkStepFn = runtime("lpn_check_step", llvm::Type::getVoidTy(context),
                          {i64});
    runFn = runtime("lpn_run", i32,
                    {i32, ptr, i64, ptr, ptr, i64, ptr, ptr, ptr, i64, ptr});
  }

  int64_t literalId(StringRef name) {
    auto it = literalIds.find(name.str());
    if (it != literalIds.end()) return it->second;
    int64_t id = literalIds.size();
    literalIds[name.str()] = id;
    return id;
  }

  llvm::Value *keyPart(llvm::Value *key, unsigned index) {
    return ir.CreateExtractValue(key, {index});
  }

  llvm::Value *asBits(llvm::Value *input) {
    if (input->getType()->isPointerTy()) return ir.CreatePtrToInt(input, i64);
    if (input->getType()->isDoubleTy()) return ir.CreateBitCast(input, i64);
    if (input->getType()->isIntegerTy(1)) return ir.CreateZExt(input, i64);
    if (input->getType() == i64) return input;
    fail("arrays of this element type are not supported");
  }

  llvm::Value *fromBits(llvm::Value *bits, mlir::Type type) {
    llvm::Type *target = lowerType(type);
    if (target->isPointerTy()) return ir.CreateIntToPtr(bits, ptr);
    if (target->isDoubleTy()) return ir.CreateBitCast(bits, f64);
    if (target->isIntegerTy(1)) return ir.CreateTrunc(bits, i1);
    if (target == i64) return bits;
    fail("arrays of this element type are not supported");
  }

  std::vector<llvm::Value *> emitBlock(mlir::Block &block) {
    for (Operation &op : block) {
      if (auto yield = dyn_cast<scf::YieldOp>(&op)) {
        std::vector<llvm::Value *> result;
        for (mlir::Value operand : yield.getResults())
          result.push_back(value(operand));
        return result;
      }
      if (auto ret = dyn_cast<func::ReturnOp>(&op)) {
        if (ret.getNumOperands() == 0) {
          ir.CreateRetVoid();
        } else if (ret.getNumOperands() == 1) {
          ir.CreateRet(value(ret.getOperand(0)));
        } else {
          llvm::Value *packed = llvm::UndefValue::get(
              ir.GetInsertBlock()->getParent()->getReturnType());
          for (auto [index, operand] : llvm::enumerate(ret.getOperands()))
            packed = ir.CreateInsertValue(packed, value(operand),
                                          {static_cast<unsigned>(index)});
          ir.CreateRet(packed);
        }
        return {};
      }
      if (isa<ScheduleReturnOp>(&op)) {
        ir.CreateRetVoid();
        return {};
      }
      emitOperation(op);
    }
    fail("block has no supported terminator");
  }

  void emitIf(scf::IfOp op) {
    llvm::Function *function = ir.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBlock = llvm::BasicBlock::Create(context, "if.then", function);
    llvm::BasicBlock *elseBlock = llvm::BasicBlock::Create(context, "if.else", function);
    llvm::BasicBlock *joinBlock = llvm::BasicBlock::Create(context, "if.end", function);
    ir.CreateCondBr(value(op.getCondition()), thenBlock, elseBlock);
    ir.SetInsertPoint(thenBlock);
    std::vector<llvm::Value *> thenValues = emitBlock(op.getThenRegion().front());
    llvm::BasicBlock *thenEnd = ir.GetInsertBlock();
    ir.CreateBr(joinBlock);
    ir.SetInsertPoint(elseBlock);
    std::vector<llvm::Value *> elseValues;
    if (!op.getElseRegion().empty())
      elseValues = emitBlock(op.getElseRegion().front());
    llvm::BasicBlock *elseEnd = ir.GetInsertBlock();
    ir.CreateBr(joinBlock);
    ir.SetInsertPoint(joinBlock);
    if (thenValues.size() != op.getNumResults() ||
        elseValues.size() != op.getNumResults())
      fail("scf.if result count mismatch");
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      auto *phi = ir.CreatePHI(lowerType(result.getType()), 2);
      phi->addIncoming(thenValues[index], thenEnd);
      phi->addIncoming(elseValues[index], elseEnd);
      put(result, phi);
    }
  }

  void emitFor(scf::ForOp op) {
    llvm::Function *function = ir.GetInsertBlock()->getParent();
    llvm::Value *lower = value(op.getLowerBound());
    llvm::Value *upper = value(op.getUpperBound());
    llvm::Value *step = value(op.getStep());
    ir.CreateCall(checkStepFn, {step});
    std::vector<llvm::Value *> initial;
    for (mlir::Value init : op.getInitArgs()) initial.push_back(value(init));
    llvm::BasicBlock *before = ir.GetInsertBlock();
    llvm::BasicBlock *header = llvm::BasicBlock::Create(context, "for.header", function);
    llvm::BasicBlock *body = llvm::BasicBlock::Create(context, "for.body", function);
    llvm::BasicBlock *after = llvm::BasicBlock::Create(context, "for.end", function);
    ir.CreateBr(header);
    ir.SetInsertPoint(header);
    auto *induction = ir.CreatePHI(i64, 2, "iv");
    induction->addIncoming(lower, before);
    std::vector<llvm::PHINode *> carried;
    for (auto [index, iterArg] : llvm::enumerate(op.getRegionIterArgs())) {
      auto *phi = ir.CreatePHI(lowerType(iterArg.getType()), 2, "iter");
      phi->addIncoming(initial[index], before);
      carried.push_back(phi);
    }
    ir.CreateCondBr(ir.CreateICmpSLT(induction, upper), body, after);
    ir.SetInsertPoint(body);
    put(op.getInductionVar(), induction);
    for (auto [index, iterArg] : llvm::enumerate(op.getRegionIterArgs()))
      put(iterArg, carried[index]);
    std::vector<llvm::Value *> yielded = emitBlock(*op.getBody());
    if (yielded.size() != carried.size())
      fail("scf.for yield count mismatch");
    llvm::Value *next = ir.CreateAdd(induction, step);
    llvm::BasicBlock *bodyEnd = ir.GetInsertBlock();
    ir.CreateBr(header);
    induction->addIncoming(next, bodyEnd);
    for (auto [index, phi] : llvm::enumerate(carried))
      phi->addIncoming(yielded[index], bodyEnd);
    ir.SetInsertPoint(after);
    for (auto [index, result] : llvm::enumerate(op.getResults()))
      put(result, carried[index]);
  }

  void emitOperation(Operation &op) {
    if (auto peek = dyn_cast<PeekOp>(&op)) {
      if (checkingGuard)
        emitGuardTest(ir.CreateICmpSGT(
            ir.CreateCall(countFn, {value(peek.getPlace())}), integer(0)));
      put(peek.getResult(), ir.CreateCall(peekFn, {value(peek.getPlace())}));
      return;
    }
    if (auto guard = dyn_cast<GuardOp>(&op)) {
      if (checkingGuard)
        emitGuardTest(value(guard.getCondition()));
      else if (!skipManualGuardChecks)
        ir.CreateCall(guardFn, {value(guard.getCondition())});
      return;
    }
    if (auto ref = dyn_cast<PlaceRefOp>(&op)) {
      auto it = placeIds.find(ref.getPlaceAttr().getValue().str());
      if (it == placeIds.end()) fail("unknown place reference");
      put(ref.getResult(), ir.CreateCall(placeFn, {integer(it->second)}));
      return;
    }
    if (auto take = dyn_cast<TakeOp>(&op)) {
      put(take.getResult(), ir.CreateCall(takeFn, {value(take.getPlace())}));
      return;
    }
    if (auto emit = dyn_cast<EmitOp>(&op)) {
      ir.CreateCall(emitFn, {value(emit.getPlace()), value(emit.getToken()),
                             value(emit.getDelay())});
      return;
    }
    if (auto count = dyn_cast<CountOp>(&op)) {
      put(count.getResult(), ir.CreateCall(countFn, {value(count.getPlace())}));
      return;
    }
    if (auto create = dyn_cast<TokenCreateOp>(&op)) {
      llvm::Value *token = ir.CreateCall(createFn);
      if (auto prefix = create.getLogPrefixAttr()) {
        for (NamedAttribute attr : prefix) {
          auto number = dyn_cast<IntegerAttr>(attr.getValue());
          if (!number) fail("token.create log_prefix must contain integers");
          token = ir.CreateCall(setFn,
              {token, integer(0), integer(literalId(attr.getName())),
               integer(number.getValue().getSExtValue())});
        }
      }
      put(create.getResult(), token);
      return;
    }
    if (auto clone = dyn_cast<TokenCloneOp>(&op)) {
      put(clone.getResult(), ir.CreateCall(cloneFn, {value(clone.getToken())}));
      return;
    }
    if (auto literal = dyn_cast<KeyLiteralOp>(&op)) {
      put(literal.getResult(), llvm::ConstantStruct::get(
          llvm::cast<llvm::StructType>(keyType),
          {integer(0), integer(literalId(literal.getValue()))}));
      return;
    }
    if (auto reg = dyn_cast<KeyRegOp>(&op)) {
      llvm::Value *key = llvm::UndefValue::get(keyType);
      key = ir.CreateInsertValue(key, integer(1), {0});
      key = ir.CreateInsertValue(key, value(reg.getId()), {1});
      put(reg.getResult(), key);
      return;
    }
    if (auto set = dyn_cast<TokenSetOp>(&op)) {
      llvm::Value *key = value(set.getKey());
      put(set.getResult(), ir.CreateCall(setFn,
          {value(set.getToken()), keyPart(key, 0), keyPart(key, 1),
           value(set.getValue())}));
      return;
    }
    if (auto get = dyn_cast<TokenGetOp>(&op)) {
      llvm::Value *key = value(get.getKey());
      put(get.getResult(), ir.CreateCall(getFn,
          {value(get.getToken()), keyPart(key, 0), keyPart(key, 1)}));
      return;
    }
    if (auto array = dyn_cast<ArrayOp>(&op)) {
      auto elements = array.getElements();
      std::vector<llvm::Constant *> literals;
      for (auto element : elements) {
        auto *literal = llvm::dyn_cast<llvm::Constant>(asBits(value(element)));
        if (!literal)
          break;
        literals.push_back(literal);
      }
      if (!elements.empty() && literals.size() == elements.size()) {
        auto *type = llvm::ArrayType::get(i64, literals.size());
        auto *global = new llvm::GlobalVariable(*module,type,true,
            llvm::GlobalValue::PrivateLinkage,llvm::ConstantArray::get(type,literals),
            "lpn.array.literal");
        global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
        std::array<llvm::Constant *,2> indices{integer(0),integer(0)};
        auto *storage = llvm::ConstantExpr::getInBoundsGetElementPtr(type,global,indices);
        put(array.getResult(),ir.CreateCall(arrayStaticFn,{integer(literals.size()),storage}));
        return;
      }
      llvm::Value *storage = llvm::ConstantPointerNull::get(
          llvm::cast<llvm::PointerType>(ptr));
      if (!elements.empty()) {
        auto *arrayType = llvm::ArrayType::get(i64, elements.size());
        llvm::Function *function = ir.GetInsertBlock()->getParent();
        llvm::IRBuilder<> entryBuilder(context);
        entryBuilder.SetInsertPoint(&function->getEntryBlock(),
                                    function->getEntryBlock().begin());
        storage = entryBuilder.CreateAlloca(arrayType);
        for (auto [index, element] : llvm::enumerate(elements)) {
          auto *slot = ir.CreateInBoundsGEP(
              arrayType, storage, {integer(0), integer(index)});
          ir.CreateStore(asBits(value(element)), slot);
        }
      }
      put(array.getResult(), ir.CreateCall(arrayCreateFn,
          {integer(elements.size()), storage}));
      return;
    }
    if (auto alloc = dyn_cast<ArrayAllocOp>(&op)) {
      put(alloc.getResult(), ir.CreateCall(arrayAllocFn,
          {value(alloc.getSize()), asBits(value(alloc.getFillValue()))}));
      return;
    }
    if (auto get = dyn_cast<ArrayGetOp>(&op)) {
      llvm::Value *bits = ir.CreateCall(arrayGetFn,
          {value(get.getArray()), value(get.getIndex())});
      put(get.getResult(), fromBits(bits, get.getResult().getType()));
      return;
    }
    if (auto set = dyn_cast<ArraySetOp>(&op)) {
      put(set.getResult(), ir.CreateCall(arraySetFn,
          {value(set.getArray()), value(set.getIndex()),
           asBits(value(set.getValue()))}));
      return;
    }
    if (auto len = dyn_cast<ArrayLenOp>(&op)) {
      put(len.getResult(), ir.CreateCall(arrayLenFn, {value(len.getArray())}));
      return;
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(&op)) {
      Attribute attribute = constant.getValue();
      if (auto number = dyn_cast<FloatAttr>(attribute)) {
        put(constant.getResult(), llvm::ConstantFP::get(f64, number.getValueAsDouble()));
      } else if (auto number = dyn_cast<IntegerAttr>(attribute)) {
        if (constant.getType().isInteger(1))
          put(constant.getResult(), llvm::ConstantInt::get(i1,
              number.getValue().getBoolValue()));
        else
          put(constant.getResult(), integer(number.getValue().getSExtValue()));
      } else {
        fail("unsupported arith.constant attribute");
      }
      return;
    }
    if (auto add = dyn_cast<arith::AddIOp>(&op)) {
      put(add.getResult(), ir.CreateAdd(value(add.getLhs()), value(add.getRhs())));
      return;
    }
    if (auto sub = dyn_cast<arith::SubIOp>(&op)) {
      put(sub.getResult(), ir.CreateSub(value(sub.getLhs()), value(sub.getRhs())));
      return;
    }
    if (auto mul = dyn_cast<arith::MulIOp>(&op)) {
      put(mul.getResult(), ir.CreateMul(value(mul.getLhs()), value(mul.getRhs())));
      return;
    }
    if (auto xorOp = dyn_cast<arith::XOrIOp>(&op)) {
      put(xorOp.getResult(), ir.CreateXor(value(xorOp.getLhs()),
                                          value(xorOp.getRhs())));
      return;
    }
    if (auto add = dyn_cast<arith::AddFOp>(&op)) {
      put(add.getResult(), ir.CreateFAdd(value(add.getLhs()), value(add.getRhs())));
      return;
    }
    if (auto sub = dyn_cast<arith::SubFOp>(&op)) {
      put(sub.getResult(), ir.CreateFSub(value(sub.getLhs()), value(sub.getRhs())));
      return;
    }
    if (auto div = dyn_cast<arith::DivFOp>(&op)) {
      put(div.getResult(), ir.CreateFDiv(value(div.getLhs()), value(div.getRhs())));
      return;
    }
    if (auto rem = dyn_cast<arith::RemFOp>(&op)) {
      put(rem.getResult(), ir.CreateFRem(value(rem.getLhs()), value(rem.getRhs())));
      return;
    }
    if (auto div = dyn_cast<arith::DivSIOp>(&op)) {
      put(div.getResult(), ir.CreateSDiv(value(div.getLhs()), value(div.getRhs())));
      return;
    }
    if (auto div = dyn_cast<arith::DivUIOp>(&op)) {
      put(div.getResult(), ir.CreateUDiv(value(div.getLhs()), value(div.getRhs())));
      return;
    }
    if (auto rem = dyn_cast<arith::RemSIOp>(&op)) {
      put(rem.getResult(), ir.CreateSRem(value(rem.getLhs()), value(rem.getRhs())));
      return;
    }
    if (auto rem = dyn_cast<arith::RemUIOp>(&op)) {
      put(rem.getResult(), ir.CreateURem(value(rem.getLhs()), value(rem.getRhs())));
      return;
    }
    if (auto compare = dyn_cast<arith::CmpIOp>(&op)) {
      llvm::Value *lhs = value(compare.getLhs());
      llvm::Value *rhs = value(compare.getRhs());
      llvm::Value *result = nullptr;
      switch (compare.getPredicate()) {
      case arith::CmpIPredicate::eq: result = ir.CreateICmpEQ(lhs, rhs); break;
      case arith::CmpIPredicate::ne: result = ir.CreateICmpNE(lhs, rhs); break;
      case arith::CmpIPredicate::slt: result = ir.CreateICmpSLT(lhs, rhs); break;
      case arith::CmpIPredicate::sle: result = ir.CreateICmpSLE(lhs, rhs); break;
      case arith::CmpIPredicate::sgt: result = ir.CreateICmpSGT(lhs, rhs); break;
      case arith::CmpIPredicate::sge: result = ir.CreateICmpSGE(lhs, rhs); break;
      case arith::CmpIPredicate::ult: result = ir.CreateICmpULT(lhs, rhs); break;
      case arith::CmpIPredicate::ule: result = ir.CreateICmpULE(lhs, rhs); break;
      case arith::CmpIPredicate::ugt: result = ir.CreateICmpUGT(lhs, rhs); break;
      case arith::CmpIPredicate::uge: result = ir.CreateICmpUGE(lhs, rhs); break;
      }
      if (!result) fail("unsupported arith.cmpi predicate");
      put(compare.getResult(), result);
      return;
    }
    if (auto select = dyn_cast<arith::SelectOp>(&op)) {
      put(select.getResult(), ir.CreateSelect(value(select.getCondition()),
          value(select.getTrueValue()), value(select.getFalseValue())));
      return;
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(&op)) {
      put(cast.getResult(), value(cast.getIn()));
      return;
    }
    if (auto cast = dyn_cast<arith::SIToFPOp>(&op)) {
      put(cast.getResult(), ir.CreateSIToFP(value(cast.getIn()), f64));
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(&op)) {
      emitFor(loop);
      return;
    }
    if (auto conditional = dyn_cast<scf::IfOp>(&op)) {
      emitIf(conditional);
      return;
    }
    if (auto call = dyn_cast<func::CallOp>(&op)) {
      auto it = helpers.find(call.getCallee().str());
      if (it == helpers.end()) fail("unknown helper: " + call.getCallee().str());
      std::vector<llvm::Value *> args;
      for (mlir::Value operand : call.getOperands()) args.push_back(value(operand));
      llvm::Value *result = ir.CreateCall(it->second, args);
      if (call.getNumResults() == 1)
        put(call.getResult(0), result);
      else if (call.getNumResults() > 1)
        for (auto [index, output] : llvm::enumerate(call.getResults()))
          put(output, ir.CreateExtractValue(result,
                                            {static_cast<unsigned>(index)}));
      return;
    }
    fail("unsupported operation in native compilation: " +
         op.getName().getStringRef().str());
  }
  void emitHelper(func::FuncOp function) {
    auto *lowered = helpers.at(function.getSymName().str());
    auto *entry = llvm::BasicBlock::Create(context, "entry", lowered);
    ir.SetInsertPoint(entry);
    values.clear();
    for (auto [source, target] : llvm::zip(function.getArguments(), lowered->args()))
      put(source, &target);
    emitBlock(function.front());
  }
  void emitGuardTest(llvm::Value *condition) {
    auto *next = llvm::BasicBlock::Create(context, "guard.next",
                                         ir.GetInsertBlock()->getParent());
    ir.CreateCondBr(condition, next, guardDisabled);
    ir.SetInsertPoint(next);
  }

  void emitEnabled(TransitionOp transition, const GuardPlan &plan, unsigned index) {
    auto *type = llvm::FunctionType::get(i1, false);
    auto *function = llvm::Function::Create(type, llvm::GlobalValue::InternalLinkage,
        "enabled_" + std::to_string(index) + "_" + transition.getSymName().str(),
        module.get());
    enabled.push_back(function);
    ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
    guardDisabled = llvm::BasicBlock::Create(context, "guard.disabled", function);
    values.clear();
    checkingGuard = true;
    if (plan.lastManualGuard) {
      for (Operation &op : transition.getBody().front()) {
        emitOperation(op);
        if (&op == plan.lastManualGuard)
          break;
      }
    } else {
      for (const auto &[name, required] : plan.required) {
        auto it = placeIds.find(name);
        if (it == placeIds.end())
          fail("unknown guard place: " + name);
        auto *place = ir.CreateCall(placeFn, {integer(it->second)});
        emitGuardTest(ir.CreateICmpSGE(ir.CreateCall(countFn, {place}), integer(required)));
      }
    }
    ir.CreateRet(llvm::ConstantInt::getTrue(context));
    ir.SetInsertPoint(guardDisabled);
    ir.CreateRet(llvm::ConstantInt::getFalse(context));
    checkingGuard = false;
    guardDisabled = nullptr;
  }

  void emitTransition(TransitionOp transition, llvm::Function *lowered) {
    auto *entry = llvm::BasicBlock::Create(context, "entry", lowered);
    ir.SetInsertPoint(entry);
    values.clear();
    // Retain prefix SSA definitions needed by the body; the explicit
    // predicates themselves were checked before entering this function.
    skipManualGuardChecks = true;
    emitBlock(transition.getBody().front());
    skipManualGuardChecks = false;
  }

  llvm::Constant *stringPointer(StringRef text) {
    auto *data = llvm::ConstantDataArray::getString(context, text, true);
    auto *global = new llvm::GlobalVariable(
        *module, data->getType(), true, llvm::GlobalValue::PrivateLinkage,
        data, "lpn.name");
    std::array<llvm::Constant *, 2> indices{integer(0), integer(0)};
    return llvm::ConstantExpr::getInBoundsGetElementPtr(
        data->getType(), global, indices);
  }

  void emitMain() {
    auto *type = llvm::FunctionType::get(i32, {i32, ptr}, false);
    auto *main = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                        "main", module.get());
    ir.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", main));
    std::vector<llvm::Constant *> nameConstants;
    std::vector<llvm::Constant *> initConstants;
    std::vector<llvm::Constant *> transitionConstants;
    std::vector<llvm::Constant *> enabledConstants;
    std::vector<llvm::Constant *> transitionNameConstants;
    std::vector<llvm::Constant *> literalConstants(literalIds.size());
    for (const PlaceSpec &place : places) {
      nameConstants.push_back(stringPointer(place.name));
      initConstants.push_back(integer(place.initial));
    }
    for (llvm::Function *transition : transitions)
      transitionConstants.push_back(transition);
    for (llvm::Function *predicate : enabled)
      enabledConstants.push_back(predicate);
    for (const auto &name : transitionNames)
      transitionNameConstants.push_back(stringPointer(name));
    for (const auto &[name, id] : literalIds)
      literalConstants[id] = stringPointer(name);
    auto makeArray = [&](llvm::Type *elementType,
                         const std::vector<llvm::Constant *> &constants,
                         const char *name) -> llvm::Constant * {
      auto *arrayType = llvm::ArrayType::get(elementType, constants.size());
      auto *data = llvm::ConstantArray::get(arrayType, constants);
      auto *global = new llvm::GlobalVariable(
          *module, arrayType, true, llvm::GlobalValue::PrivateLinkage,
          data, name);
      std::array<llvm::Constant *, 2> indices{integer(0), integer(0)};
      return llvm::ConstantExpr::getInBoundsGetElementPtr(
          arrayType, global, indices);
    };
    llvm::Value *nameArray = makeArray(ptr, nameConstants, "lpn.place.names");
    llvm::Value *initArray = makeArray(i64, initConstants, "lpn.place.initial");
    llvm::Value *transitionArray =
        makeArray(ptr, transitionConstants, "lpn.transitions");
    llvm::Value *enabledArray = makeArray(ptr, enabledConstants, "lpn.enabled");
    llvm::Value *transitionNameArray = makeArray(ptr, transitionNameConstants, "lpn.transition.names");
    llvm::Value *literalArray =
        makeArray(ptr, literalConstants, "lpn.literal.names");
    auto arguments = main->arg_begin();
    llvm::Value *argc = &*arguments++;
    llvm::Value *argv = &*arguments;
    auto *status = ir.CreateCall(runFn,
        {argc, argv, integer(places.size()), nameArray, initArray,
         integer(transitions.size()), transitionArray, enabledArray, transitionNameArray,
         integer(literalConstants.size()), literalArray});
    ir.CreateRet(status);
  }

  llvm::LLVMContext context;
  llvm::IRBuilder<> ir;
  std::unique_ptr<llvm::Module> module;
  llvm::Type *i1, *i32, *i64, *f64, *ptr, *keyType;
  llvm::FunctionCallee placeFn, takeFn, emitFn, countFn, createFn, cloneFn;
  llvm::FunctionCallee peekFn, guardFn;
  llvm::FunctionCallee setFn, getFn, arrayCreateFn, arrayAllocFn, arrayGetFn;
  llvm::FunctionCallee arrayStaticFn;
  llvm::FunctionCallee arraySetFn, arrayLenFn, checkStepFn, runFn;
  llvm::DenseMap<mlir::Value, llvm::Value *> values;
  std::map<std::string, int64_t> literalIds;
  std::map<std::string, int64_t> placeIds;
  std::map<std::string, llvm::Function *> helpers;
  std::vector<PlaceSpec> places;
  std::vector<llvm::Function *> transitions;
  std::vector<llvm::Function *> enabled;
  std::vector<std::string> transitionNames;
  bool checkingGuard = false;
  bool skipManualGuardChecks = false;
  llvm::BasicBlock *guardDisabled = nullptr;
};

} // namespace

int main(int argc, char **argv) {
  if (argc != 4 || std::string(argv[2]) != "-o") {
    llvm::errs() << "Usage: lpn-compile input.mlir -o output-binary\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::lpn::LPNDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext mlirContext(registry);
  mlirContext.loadAllAvailableDialects();
  auto source = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &mlirContext);
  if (!source)
    return 1;

  try {
    Compiler compiler;
    compiler.compile(*source);

    int llvmFd = -1;
    llvm::SmallString<128> llvmPath;
    if (auto error = llvm::sys::fs::createTemporaryFile(
            "lpn-native", "ll", llvmFd, llvmPath))
      throw std::runtime_error("cannot create temporary LLVM IR file: " +
                               error.message());
    {
      llvm::raw_fd_ostream output(llvmFd, true);
      compiler.getModule().print(output, nullptr);
    }
    int objectFd = -1;
    llvm::SmallString<128> objectPath;
    if (auto error = llvm::sys::fs::createTemporaryFile(
            "lpn-native", "o", objectFd, objectPath))
      throw std::runtime_error("cannot create temporary object file: " +
                               error.message());
    close(objectFd);

    auto optimizer = llvm::sys::findProgramByName("opt-20");
    auto llc = llvm::sys::findProgramByName("llc-20");
    auto linker = llvm::sys::findProgramByName("clang++-20");
    if (!optimizer || !llc || !linker)
      throw std::runtime_error("opt-20, llc-20 and clang++-20 are required");
    llvm::SmallVector<llvm::StringRef> optimizeArgs{
        *optimizer, "-O3", "-S", llvmPath, "-o", llvmPath};
    if (llvm::sys::ExecuteAndWait(*optimizer, optimizeArgs) != 0)
      throw std::runtime_error("LLVM optimization failed");
    llvm::SmallVector<llvm::StringRef> codegenArgs{
        *llc, "-O3", "-filetype=obj", llvmPath, "-o", objectPath};
    int codegenStatus = llvm::sys::ExecuteAndWait(*llc, codegenArgs);
    if (codegenStatus != 0)
      throw std::runtime_error("LLVM object generation failed");
    llvm::SmallVector<llvm::StringRef> linkArgs{
        *linker, objectPath, LPN_NATIVE_RUNTIME_PATH, "-O3", "-no-pie",
        "-o", argv[3]};
    int linkStatus = llvm::sys::ExecuteAndWait(*linker, linkArgs);
    llvm::sys::fs::remove(llvmPath);
    llvm::sys::fs::remove(objectPath);
    if (linkStatus != 0)
      throw std::runtime_error("native linker failed");
  } catch (const std::exception &error) {
    llvm::errs() << "lpn-compile: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
