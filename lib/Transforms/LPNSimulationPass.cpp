#include "LPN/Conversion/LPNPasses.h"
#include "LPN/Dialect/LPNOps.h"
#include "LPN/Dialect/LPNTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>
#include <variant>

using namespace mlir;
using namespace mlir::lpn;

namespace {

enum class ExecStatus { Ok, Blocked, Failed };

struct RuntimeKey {
  enum class Kind { Literal, Register };

  Kind kind = Kind::Literal;
  std::string literal;
  int64_t reg = 0;

  static RuntimeKey literalKey(StringRef value) {
    RuntimeKey key;
    key.kind = Kind::Literal;
    key.literal = value.str();
    return key;
  }

  static RuntimeKey regKey(int64_t identifier) {
    RuntimeKey key;
    key.kind = Kind::Register;
    key.reg = identifier;
    return key;
  }

  bool operator==(const RuntimeKey &rhs) const {
    if (kind != rhs.kind)
      return false;
    if (kind == Kind::Literal)
      return literal == rhs.literal;
    return reg == rhs.reg;
  }
};

} // namespace

namespace llvm {
template <> struct DenseMapInfo<RuntimeKey> {
  static RuntimeKey getEmptyKey() {
    RuntimeKey key;
    key.kind = RuntimeKey::Kind::Literal;
    key.literal = "";
    return key;
  }
  static RuntimeKey getTombstoneKey() {
    RuntimeKey key;
    key.kind = RuntimeKey::Kind::Literal;
    key.literal = "\xff";
    return key;
  }
  static unsigned getHashValue(const RuntimeKey &key) {
    if (key.kind == RuntimeKey::Kind::Literal)
      return hash_value(key.literal);
    return hash_value(key.reg);
  }
  static bool isEqual(const RuntimeKey &lhs, const RuntimeKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {

struct RuntimeToken {
  llvm::DenseMap<RuntimeKey, int64_t> fields;

  int64_t get(const RuntimeKey &key) const {
    auto it = fields.find(key);
    if (it == fields.end())
      return 0;
    return it->second;
  }

  void set(const RuntimeKey &key, int64_t value) { fields[key] = value; }
};

struct RuntimePlace {
  struct ScheduledEntry {
    double readyTime;
    uint64_t order;
    RuntimeToken token;
  };

  struct CompareEntry {
    bool operator()(const ScheduledEntry &lhs,
                    const ScheduledEntry &rhs) const {
      if (lhs.readyTime != rhs.readyTime)
        return lhs.readyTime > rhs.readyTime;
      return lhs.order > rhs.order;
    }
  };

  explicit RuntimePlace(StringRef name, int64_t initialTokens)
      : name(name.str()) {
    for (int64_t idx = 0; idx < initialTokens; ++idx)
      ready.emplace_back();
  }

  std::optional<RuntimeToken> takeImmediate() {
    if (ready.empty())
      return std::nullopt;
    RuntimeToken token = ready.front();
    ready.pop_front();
    return token;
  }

  void restoreFront(const RuntimeToken &token) {
    ready.push_front(token);
  }

  void schedule(RuntimeToken token, double readyTime) {
    scheduled.push(
        ScheduledEntry{readyTime, nextSequence++, std::move(token)});
  }

  bool commitReady(double currentTime) {
    bool changed = false;
    while (!scheduled.empty() &&
           scheduled.top().readyTime <= currentTime + 1e-12) {
      ready.push_back(std::move(scheduled.top().token));
      scheduled.pop();
      changed = true;
    }
    return changed;
  }

  std::optional<double> nextScheduledTime() const {
    if (scheduled.empty())
      return std::nullopt;
    return scheduled.top().readyTime;
  }

  size_t scheduledLen() const { return scheduled.size(); }
  size_t readyLen() const { return ready.size(); }

  std::string name;
  std::deque<RuntimeToken> ready;
  std::priority_queue<ScheduledEntry, std::vector<ScheduledEntry>, CompareEntry>
      scheduled;
  uint64_t nextSequence = 0;
};

struct RuntimeArray;

struct RuntimeValue {
  using ArrayPtr = std::shared_ptr<RuntimeArray>;
  using Storage =
      std::variant<std::monostate, int64_t, double, bool, RuntimeToken,
                   RuntimeKey, RuntimePlace *, ArrayPtr>;

  Storage storage;

  static RuntimeValue makeInt(int64_t value) {
    RuntimeValue v;
    v.storage = value;
    return v;
  }

  static RuntimeValue makeFloat(double value) {
    RuntimeValue v;
    v.storage = value;
    return v;
  }

  static RuntimeValue makeBool(bool value) {
    RuntimeValue v;
    v.storage = value;
    return v;
  }

  static RuntimeValue makeToken(const RuntimeToken &token) {
    RuntimeValue v;
    v.storage = token;
    return v;
  }

  static RuntimeValue makeKey(const RuntimeKey &key) {
    RuntimeValue v;
    v.storage = key;
    return v;
  }

  static RuntimeValue makePlace(RuntimePlace *place) {
    RuntimeValue v;
    v.storage = place;
    return v;
  }

  static RuntimeValue makeArray(ArrayPtr array) {
    RuntimeValue v;
    v.storage = array;
    return v;
  }

  bool isIntLike() const {
    return std::holds_alternative<int64_t>(storage) ||
           std::holds_alternative<bool>(storage);
  }

  int64_t asInt() const {
    if (auto *value = std::get_if<int64_t>(&storage))
      return *value;
    if (auto *flag = std::get_if<bool>(&storage))
      return *flag ? 1 : 0;
    llvm_unreachable("value does not hold an integer");
  }

  double asFloat() const {
    if (auto *value = std::get_if<double>(&storage))
      return *value;
    if (auto *intValue = std::get_if<int64_t>(&storage))
      return static_cast<double>(*intValue);
    llvm_unreachable("value does not hold a float");
  }

  bool asBool() const {
    if (auto *flag = std::get_if<bool>(&storage))
      return *flag;
    if (auto *intValue = std::get_if<int64_t>(&storage))
      return *intValue != 0;
    llvm_unreachable("value does not hold a bool");
  }

  RuntimeToken asToken() const {
    if (auto *token = std::get_if<RuntimeToken>(&storage))
      return *token;
    llvm_unreachable("value does not hold a token");
  }

  RuntimeKey asKey() const {
    if (auto *key = std::get_if<RuntimeKey>(&storage))
      return *key;
    llvm_unreachable("value does not hold a key");
  }

  RuntimePlace *asPlace() const {
    if (auto *place = std::get_if<RuntimePlace *>(&storage))
      return *place;
    llvm_unreachable("value does not hold a place");
  }

  ArrayPtr asArray() const {
    if (auto *array = std::get_if<ArrayPtr>(&storage))
      return *array;
    llvm_unreachable("value does not hold an array");
  }
};

struct RuntimeArray {
  llvm::SmallVector<RuntimeValue, 4> elements;
};

struct TransactionContext {
  llvm::SmallVector<std::pair<RuntimePlace *, RuntimeToken>, 4> takes;
  llvm::SmallVector<std::tuple<RuntimePlace *, RuntimeToken, double>, 4> emits;

  void clear() {
    takes.clear();
    emits.clear();
  }

  void recordTake(RuntimePlace *place, RuntimeToken token) {
    takes.emplace_back(place, std::move(token));
  }

  void recordEmit(RuntimePlace *place, RuntimeToken token, double readyTime) {
    emits.emplace_back(place, std::move(token), readyTime);
  }

  void rollback() {
    for (auto it = takes.rbegin(); it != takes.rend(); ++it)
      it->first->restoreFront(it->second);
    clear();
  }
};

struct NetState {
  llvm::StringMap<std::unique_ptr<RuntimePlace>> placeMap;
  llvm::SmallVector<RuntimePlace *> placeOrder;
  double currentTime = 0.0;

  RuntimePlace *addPlace(StringRef name, int64_t initialTokens) {
    auto place = std::make_unique<RuntimePlace>(name, initialTokens);
    RuntimePlace *ptr = place.get();
    placeOrder.push_back(ptr);
    placeMap.try_emplace(name, std::move(place));
    return ptr;
  }

  RuntimePlace *getPlace(StringRef name) const {
    auto it = placeMap.find(name);
    if (it == placeMap.end())
      return nullptr;
    return it->second.get();
  }

  bool commitReady(double uptoTime) {
    bool changed = false;
    for (RuntimePlace *place : placeOrder)
      changed |= place->commitReady(uptoTime);
    return changed;
  }

  std::optional<double> earliestScheduledTime() const {
    std::optional<double> next;
    for (RuntimePlace *place : placeOrder) {
      auto candidate = place->nextScheduledTime();
      if (!candidate)
        continue;
      if (!next || *candidate < *next)
        next = candidate;
    }
    return next;
  }

  bool hasPendingScheduled() const {
    for (RuntimePlace *place : placeOrder)
      if (place->scheduledLen() > 0)
        return true;
    return false;
  }
};

struct EvaluationContext {
  NetState &net;
  llvm::StringMap<func::FuncOp> &functions;
  TransactionContext *txn = nullptr;
  RuntimePlace *blockedPlace = nullptr;
};

//===----------------------------------------------------------------------===//
// Evaluator helpers
//===----------------------------------------------------------------------===//

static LogicalResult assignBlockArguments(Block &block,
                                          ArrayRef<RuntimeValue> values,
                                          llvm::DenseMap<Value, RuntimeValue> &map,
                                          Location loc) {
  if (block.getNumArguments() != values.size()) {
    emitError(loc) << "expected " << block.getNumArguments()
                   << " block arguments but got " << values.size();
    return failure();
  }
  for (auto it : llvm::zip(block.getArguments(), values))
    map[std::get<0>(it)] = std::get<1>(it);
  return success();
}

class Evaluator {
public:
  Evaluator(EvaluationContext &ctx) : ctx(ctx) {}

  ExecStatus runBlock(Block &block, llvm::DenseMap<Value, RuntimeValue> &values,
                      SmallVector<RuntimeValue> *yieldValues = nullptr) {
    for (Operation &op : block) {
      bool shouldStop = false;
      ExecStatus status =
          executeOperation(op, values, shouldStop, yieldValues);
      if (status != ExecStatus::Ok)
        return status;
      if (shouldStop)
        break;
    }
    return ExecStatus::Ok;
  }

  ExecStatus runRegion(Region &region, ArrayRef<RuntimeValue> args,
                       llvm::DenseMap<Value, RuntimeValue> &values,
                       SmallVector<RuntimeValue> *yieldValues = nullptr) {
    Block &entry = region.front();
    Location loc = region.getParentOp()
                        ? region.getParentOp()->getLoc()
                        : UnknownLoc::get(region.getContext());
    if (failed(assignBlockArguments(entry, args, values, loc)))
      return ExecStatus::Failed;
    return runBlock(entry, values, yieldValues);
  }

  ExecStatus runFunction(func::FuncOp func, ArrayRef<RuntimeValue> args,
                         SmallVector<RuntimeValue> &results) {
    llvm::DenseMap<Value, RuntimeValue> values;
    if (failed(assignBlockArguments(func.front(), args, values,
                                    func.getLoc())))
      return ExecStatus::Failed;
    return runBlock(func.front(), values, &results);
  }

private:
  RuntimeValue lookup(Value value,
                      const llvm::DenseMap<Value, RuntimeValue> &values) {
    auto it = values.find(value);
    assert(it != values.end() && "value not initialized");
    return it->second;
  }

  void store(Value value, const RuntimeValue &runtime,
             llvm::DenseMap<Value, RuntimeValue> &values) {
    values[value] = runtime;
  }

  ExecStatus handlePlaceRef(PlaceRefOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimePlace *place = ctx.net.getPlace(op.getPlaceAttr().getValue());
    if (!place) {
      emitError(op.getLoc()) << "unknown place '"
                             << op.getPlaceAttr().getValue() << "'";
      return ExecStatus::Failed;
    }
    store(op.getResult(), RuntimeValue::makePlace(place), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleTake(TakeOp op,
                        llvm::DenseMap<Value, RuntimeValue> &values) {
    assert(ctx.txn && "transaction required for take");
    RuntimePlace *place = lookup(op.getPlace(), values).asPlace();
    auto maybeToken = place->takeImmediate();
    if (!maybeToken) {
      ctx.blockedPlace = place;
      return ExecStatus::Blocked;
    }
    ctx.txn->recordTake(place, *maybeToken);
    store(op.getResult(), RuntimeValue::makeToken(*maybeToken), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleEmit(EmitOp op,
                        llvm::DenseMap<Value, RuntimeValue> &values) {
    assert(ctx.txn && "transaction required for emit");
    RuntimePlace *place = lookup(op.getPlace(), values).asPlace();
    RuntimeToken token = lookup(op.getToken(), values).asToken();
    double delay = lookup(op.getDelay(), values).asFloat();
    double readyTime =
        std::max(0.0, delay) + ctx.net.currentTime;
    ctx.txn->recordEmit(place, std::move(token), readyTime);
    return ExecStatus::Ok;
  }

  ExecStatus handleCount(CountOp op,
                         llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimePlace *place = lookup(op.getPlace(), values).asPlace();
    store(op.getResult(), RuntimeValue::makeInt(place->readyLen()), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleTokenCreate(TokenCreateOp op,
                               llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimeToken token;
    if (auto dict = op.getLogPrefixAttr()) {
      for (auto attr : dict) {
        auto intAttr = llvm::dyn_cast<IntegerAttr>(attr.getValue());
        if (!intAttr) {
          emitError(op.getLoc())
              << "token.create log_prefix must hold integer attributes";
          return ExecStatus::Failed;
        }
        RuntimeKey key = RuntimeKey::literalKey(attr.getName());
        token.set(key, intAttr.getValue().getSExtValue());
      }
    }
    store(op.getResult(), RuntimeValue::makeToken(token), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleTokenClone(TokenCloneOp op,
                              llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimeToken token = lookup(op.getToken(), values).asToken();
    store(op.getResult(), RuntimeValue::makeToken(token), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleTokenSet(TokenSetOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimeToken token = lookup(op.getToken(), values).asToken();
    RuntimeKey key = lookup(op.getKey(), values).asKey();
    int64_t value = lookup(op.getValue(), values).asInt();
    token.set(key, value);
    store(op.getResult(), RuntimeValue::makeToken(token), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleTokenGet(TokenGetOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimeToken token = lookup(op.getToken(), values).asToken();
    RuntimeKey key = lookup(op.getKey(), values).asKey();
    store(op.getResult(), RuntimeValue::makeInt(token.get(key)), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleKeyLiteral(KeyLiteralOp op,
                              llvm::DenseMap<Value, RuntimeValue> &values) {
    store(op.getResult(),
          RuntimeValue::makeKey(RuntimeKey::literalKey(op.getValue())),
          values);
    return ExecStatus::Ok;
  }

  ExecStatus handleKeyReg(KeyRegOp op,
                          llvm::DenseMap<Value, RuntimeValue> &values) {
    int64_t identifier = lookup(op.getId(), values).asInt();
    store(op.getResult(), RuntimeValue::makeKey(RuntimeKey::regKey(identifier)),
          values);
    return ExecStatus::Ok;
  }

  ExecStatus handleArray(ArrayOp op,
                         llvm::DenseMap<Value, RuntimeValue> &values) {
    auto array = std::make_shared<RuntimeArray>();
    for (Value operand : op.getElements())
      array->elements.push_back(lookup(operand, values));
    store(op.getResult(), RuntimeValue::makeArray(array), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleArrayGet(ArrayGetOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    auto array = lookup(op.getArray(), values).asArray();
    int64_t index = lookup(op.getIndex(), values).asInt();
    if (index < 0 || static_cast<size_t>(index) >= array->elements.size()) {
      emitError(op.getLoc()) << "array.get index out of range";
      return ExecStatus::Failed;
    }
    store(op.getResult(), array->elements[index], values);
    return ExecStatus::Ok;
  }

  ExecStatus handleArraySet(ArraySetOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    auto array = lookup(op.getArray(), values).asArray();
    int64_t index = lookup(op.getIndex(), values).asInt();
    if (index < 0 || static_cast<size_t>(index) >= array->elements.size()) {
      emitError(op.getLoc()) << "array.set index out of range";
      return ExecStatus::Failed;
    }
    auto updated = std::make_shared<RuntimeArray>(*array);
    updated->elements[index] = lookup(op.getValue(), values);
    store(op.getResult(), RuntimeValue::makeArray(updated), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleArrayLen(ArrayLenOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    auto array = lookup(op.getArray(), values).asArray();
    store(op.getResult(),
          RuntimeValue::makeInt(static_cast<int64_t>(array->elements.size())),
          values);
    return ExecStatus::Ok;
  }

  ExecStatus handleConstant(arith::ConstantOp op,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    Attribute attr = op.getValue();
    Type type = op.getType();
    RuntimeValue runtime;
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr)) {
      runtime = RuntimeValue::makeFloat(floatAttr.getValueAsDouble());
    } else if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      if (type.isInteger(1))
      runtime =
          RuntimeValue::makeBool(intAttr.getValue().getBoolValue());
      else
        runtime =
            RuntimeValue::makeInt(intAttr.getValue().getSExtValue());
    } else {
      emitError(op.getLoc()) << "unsupported constant attribute: " << attr;
      return ExecStatus::Failed;
    }
    store(op.getResult(), runtime, values);
    return ExecStatus::Ok;
  }

  template <typename OpTy, typename FuncTy>
  ExecStatus handleBinaryOp(OpTy op, FuncTy fn,
                            llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimeValue lhs = lookup(op.getLhs(), values);
    RuntimeValue rhs = lookup(op.getRhs(), values);
    auto result = fn(lhs, rhs);
    store(op.getResult(), result, values);
    return ExecStatus::Ok;
  }

  ExecStatus handleCmp(arith::CmpIOp op,
                       llvm::DenseMap<Value, RuntimeValue> &values) {
    int64_t lhs = lookup(op.getLhs(), values).asInt();
    int64_t rhs = lookup(op.getRhs(), values).asInt();
    bool result = false;
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      result = lhs == rhs;
      break;
    case arith::CmpIPredicate::ne:
      result = lhs != rhs;
      break;
    case arith::CmpIPredicate::slt:
      result = lhs < rhs;
      break;
    case arith::CmpIPredicate::sle:
      result = lhs <= rhs;
      break;
    case arith::CmpIPredicate::sgt:
      result = lhs > rhs;
      break;
    case arith::CmpIPredicate::sge:
      result = lhs >= rhs;
      break;
    default:
      emitError(op.getLoc()) << "unsupported integer predicate";
      return ExecStatus::Failed;
    }
    store(op.getResult(), RuntimeValue::makeBool(result), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleSelect(arith::SelectOp op,
                          llvm::DenseMap<Value, RuntimeValue> &values) {
    bool cond = lookup(op.getCondition(), values).asBool();
    RuntimeValue chosen =
        cond ? lookup(op.getTrueValue(), values)
             : lookup(op.getFalseValue(), values);
    store(op.getResult(), chosen, values);
    return ExecStatus::Ok;
  }

  ExecStatus handleIndexCast(arith::IndexCastOp op,
                             llvm::DenseMap<Value, RuntimeValue> &values) {
    RuntimeValue input = lookup(op.getIn(), values);
    if (op.getIn().getType().isIndex())
      store(op.getResult(), RuntimeValue::makeInt(input.asInt()), values);
    else
      store(op.getResult(), RuntimeValue::makeInt(input.asInt()), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleSIToFP(arith::SIToFPOp op,
                          llvm::DenseMap<Value, RuntimeValue> &values) {
    double value = static_cast<double>(lookup(op.getIn(), values).asInt());
    store(op.getResult(), RuntimeValue::makeFloat(value), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleDivSI(arith::DivSIOp op,
                         llvm::DenseMap<Value, RuntimeValue> &values) {
    int64_t lhs = lookup(op.getLhs(), values).asInt();
    int64_t rhs = lookup(op.getRhs(), values).asInt();
    if (rhs == 0) {
      emitError(op.getLoc()) << "division by zero";
      return ExecStatus::Failed;
    }
    store(op.getResult(), RuntimeValue::makeInt(lhs / rhs), values);
    return ExecStatus::Ok;
  }

  ExecStatus handleFor(scf::ForOp op,
                       llvm::DenseMap<Value, RuntimeValue> &values) {
    if (!op.getInitArgs().empty() || op.getNumResults() != 0) {
      emitError(op.getLoc()) << "only simple scf.for loops are supported";
      return ExecStatus::Failed;
    }
    int64_t lower = lookup(op.getLowerBound(), values).asInt();
    int64_t upper = lookup(op.getUpperBound(), values).asInt();
    int64_t step = lookup(op.getStep(), values).asInt();
    if (step == 0) {
      emitError(op.getLoc()) << "scf.for step cannot be zero";
      return ExecStatus::Failed;
    }
    for (int64_t iv = lower; iv < upper; iv += step) {
      store(op.getInductionVar(), RuntimeValue::makeInt(iv), values);
      ExecStatus status = runBlock(*op.getBody(), values);
      if (status != ExecStatus::Ok)
        return status;
    }
    return ExecStatus::Ok;
  }

  ExecStatus handleIf(scf::IfOp op,
                      llvm::DenseMap<Value, RuntimeValue> &values) {
    bool cond = lookup(op.getCondition(), values).asBool();
    Region &region = cond ? op.getThenRegion() : op.getElseRegion();
    SmallVector<RuntimeValue> results;
    if (!region.empty()) {
      ExecStatus status = runRegion(region, {}, values, &results);
      if (status != ExecStatus::Ok)
        return status;
    }
    if (op.getNumResults() != results.size()) {
      emitError(op.getLoc()) << "if region yielded "
                             << results.size() << " results but "
                             << op.getNumResults() << " were expected";
      return ExecStatus::Failed;
    }
    for (auto it : llvm::enumerate(op.getResults()))
      store(it.value(), results[it.index()], values);
    return ExecStatus::Ok;
  }

  ExecStatus handleCall(func::CallOp op,
                        llvm::DenseMap<Value, RuntimeValue> &values) {
    auto funcIt = ctx.functions.find(op.getCallee());
    if (funcIt == ctx.functions.end()) {
      emitError(op.getLoc()) << "unknown callee '" << op.getCallee() << "'";
      return ExecStatus::Failed;
    }
    SmallVector<RuntimeValue> args;
    for (Value operand : op.getOperands())
      args.push_back(lookup(operand, values));
    SmallVector<RuntimeValue> results;
    Evaluator calleeEvaluator(ctx);
    ExecStatus status = calleeEvaluator.runFunction(funcIt->second, args, results);
    if (status != ExecStatus::Ok) {
      ctx.blockedPlace = calleeEvaluator.ctx.blockedPlace;
      return status;
    }
    if (results.size() != op.getNumResults()) {
      emitError(op.getLoc()) << "call produced " << results.size()
                             << " results but " << op.getNumResults()
                             << " expected";
      return ExecStatus::Failed;
    }
    for (auto it : llvm::enumerate(op.getResults()))
      store(it.value(), results[it.index()], values);
    return ExecStatus::Ok;
  }

  ExecStatus handleYield(scf::YieldOp op,
                         llvm::DenseMap<Value, RuntimeValue> &values,
                         SmallVector<RuntimeValue> *yieldValues,
                         bool &shouldStop) {
    shouldStop = true;
    if (yieldValues) {
      yieldValues->clear();
      for (Value operand : op.getResults())
        yieldValues->push_back(lookup(operand, values));
    } else if (!op.getResults().empty()) {
      emitError(op.getLoc()) << "yield with values not expected here";
      return ExecStatus::Failed;
    }
    return ExecStatus::Ok;
  }

  ExecStatus handleFuncReturn(func::ReturnOp op,
                              llvm::DenseMap<Value, RuntimeValue> &values,
                              SmallVector<RuntimeValue> *yieldValues,
                              bool &shouldStop) {
    shouldStop = true;
    if (!yieldValues) {
      emitError(op.getLoc()) << "return outside function evaluation";
      return ExecStatus::Failed;
    }
    yieldValues->clear();
    for (Value operand : op.getOperands())
      yieldValues->push_back(lookup(operand, values));
    return ExecStatus::Ok;
  }

  ExecStatus executeOperation(Operation &op,
                              llvm::DenseMap<Value, RuntimeValue> &values,
                              bool &shouldStop,
                              SmallVector<RuntimeValue> *yieldValues) {
    shouldStop = false;
    if (auto placeRef = dyn_cast<PlaceRefOp>(op))
      return handlePlaceRef(placeRef, values);
    if (auto take = dyn_cast<TakeOp>(op))
      return handleTake(take, values);
    if (auto emit = dyn_cast<EmitOp>(op))
      return handleEmit(emit, values);
    if (auto count = dyn_cast<CountOp>(op))
      return handleCount(count, values);
    if (auto create = dyn_cast<TokenCreateOp>(op))
      return handleTokenCreate(create, values);
    if (auto clone = dyn_cast<TokenCloneOp>(op))
      return handleTokenClone(clone, values);
    if (auto set = dyn_cast<TokenSetOp>(op))
      return handleTokenSet(set, values);
    if (auto get = dyn_cast<TokenGetOp>(op))
      return handleTokenGet(get, values);
    if (auto literal = dyn_cast<KeyLiteralOp>(op))
      return handleKeyLiteral(literal, values);
    if (auto keyReg = dyn_cast<KeyRegOp>(op))
      return handleKeyReg(keyReg, values);
    if (auto array = dyn_cast<ArrayOp>(op))
      return handleArray(array, values);
    if (auto arrayGet = dyn_cast<ArrayGetOp>(op))
      return handleArrayGet(arrayGet, values);
    if (auto arraySet = dyn_cast<ArraySetOp>(op))
      return handleArraySet(arraySet, values);
    if (auto arrayLen = dyn_cast<ArrayLenOp>(op))
      return handleArrayLen(arrayLen, values);
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      return handleConstant(constant, values);
    if (auto addi = dyn_cast<arith::AddIOp>(op)) {
      return handleBinaryOp(addi,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeInt(lhs.asInt() + rhs.asInt());
                            },
                            values);
    }
    if (auto subi = dyn_cast<arith::SubIOp>(op)) {
      return handleBinaryOp(subi,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeInt(lhs.asInt() - rhs.asInt());
                            },
                            values);
    }
    if (auto muli = dyn_cast<arith::MulIOp>(op)) {
      return handleBinaryOp(muli,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeInt(lhs.asInt() * rhs.asInt());
                            },
                            values);
    }
    if (auto xori = dyn_cast<arith::XOrIOp>(op)) {
      return handleBinaryOp(xori,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeInt(lhs.asInt() ^ rhs.asInt());
                            },
                            values);
    }
    if (auto addf = dyn_cast<arith::AddFOp>(op)) {
      return handleBinaryOp(addf,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeFloat(lhs.asFloat() + rhs.asFloat());
                            },
                            values);
    }
    if (auto subf = dyn_cast<arith::SubFOp>(op)) {
      return handleBinaryOp(subf,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeFloat(lhs.asFloat() - rhs.asFloat());
                            },
                            values);
    }
    if (auto divf = dyn_cast<arith::DivFOp>(op)) {
      return handleBinaryOp(divf,
                            [&](RuntimeValue lhs, RuntimeValue rhs) {
                              return RuntimeValue::makeFloat(lhs.asFloat() / rhs.asFloat());
                            },
                            values);
    }
    if (auto divsi = dyn_cast<arith::DivSIOp>(op))
      return handleDivSI(divsi, values);
    if (auto cmpi = dyn_cast<arith::CmpIOp>(op))
      return handleCmp(cmpi, values);
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return handleSelect(select, values);
    if (auto sitofp = dyn_cast<arith::SIToFPOp>(op))
      return handleSIToFP(sitofp, values);
    if (auto indexCast = dyn_cast<arith::IndexCastOp>(op))
      return handleIndexCast(indexCast, values);
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      return handleFor(forOp, values);
    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      return handleIf(ifOp, values);
    if (auto call = dyn_cast<func::CallOp>(op))
      return handleCall(call, values);
    if (auto yield = dyn_cast<scf::YieldOp>(op))
      return handleYield(yield, values, yieldValues, shouldStop);
    if (auto ret = dyn_cast<func::ReturnOp>(op))
      return handleFuncReturn(ret, values, yieldValues, shouldStop);
    if (isa<ScheduleReturnOp>(op)) {
      shouldStop = true;
      return ExecStatus::Ok;
    }
    emitError(op.getLoc()) << "unsupported operation: " << op.getName();
    return ExecStatus::Failed;
  }

public:
  EvaluationContext &ctx;
};

//===----------------------------------------------------------------------===//
// Simulation driver
//===----------------------------------------------------------------------===//

struct TransitionDriver {
  TransitionDriver(NetState &state,
                   llvm::StringMap<func::FuncOp> &functions)
      : ctx{state, functions, nullptr, nullptr} {}

  ExecStatus tryFire(lpn::TransitionOp op) {
    TransactionContext txn;
    ctx.txn = &txn;
    ctx.blockedPlace = nullptr;
    Evaluator evaluator(ctx);
    llvm::DenseMap<Value, RuntimeValue> values;
    ExecStatus status = evaluator.runBlock(op.getBody().front(), values, nullptr);
    if (status == ExecStatus::Ok) {
      for (auto &entry : txn.emits)
        std::get<0>(entry)->schedule(std::move(std::get<1>(entry)),
                                     std::get<2>(entry));
      txn.clear();
      ctx.net.commitReady(ctx.net.currentTime);
      ctx.txn = nullptr;
      return ExecStatus::Ok;
    }
    txn.rollback();
    ctx.blockedPlace = evaluator.ctx.blockedPlace;
    ctx.txn = nullptr;
    return status;
  }

  EvaluationContext ctx;
};

LogicalResult simulateNet(NetOp netOp, ModuleOp module, double maxTime) {
  llvm::StringMap<func::FuncOp> functions;
  module.walk([&](func::FuncOp func) {
    functions.try_emplace(func.getSymName(), func);
  });

  NetState state;
  SmallVector<TransitionOp> transitions;

  Block &body = netOp.getBody().front();
  for (Operation &op : body) {
    if (auto place = dyn_cast<PlaceOp>(op)) {
      int64_t initialTokens = 0;
      if (auto attr = place.getInitialTokens())
        initialTokens = static_cast<int64_t>(*attr);
      state.addPlace(place.getSymName(), initialTokens);
      continue;
    }
    if (auto transition = dyn_cast<TransitionOp>(op)) {
      transitions.push_back(transition);
      continue;
    }
  }

  state.commitReady(state.currentTime);
  TransitionDriver driver(state, functions);

  while (state.currentTime <= maxTime) {
    bool progress = false;
    for (TransitionOp transition : transitions) {
      while (true) {
        ExecStatus status = driver.tryFire(transition);
        if (status == ExecStatus::Ok) {
          progress = true;
          continue;
        }
        if (status == ExecStatus::Blocked) {
          break;
        }
        return failure();
      }
    }
    if (progress)
      continue;
    auto nextTime = state.earliestScheduledTime();
    if (!nextTime || *nextTime > maxTime)
      break;
    state.currentTime = *nextTime;
    state.commitReady(state.currentTime);
  }

  llvm::outs() << "Simulation for lpn.net at " << netOp.getLoc()
               << " finished at t=" << state.currentTime << "\n";
  for (RuntimePlace *place : state.placeOrder) {
    llvm::outs() << "  place " << place->name << ": "
                 << place->readyLen() << " token(s)\n";
    for (const RuntimeToken &token : place->ready) {
      llvm::outs() << "    token {";
      bool first = true;
      for (const auto &entry : token.fields) {
        if (!first)
          llvm::outs() << ", ";
        first = false;
        if (entry.first.kind == RuntimeKey::Kind::Literal)
          llvm::outs() << entry.first.literal;
        else
          llvm::outs() << "reg(" << entry.first.reg << ")";
        llvm::outs() << ": " << entry.second;
      }
      llvm::outs() << "}\n";
    }
  }

  return success();
}

struct LPNSimulationPass
    : public PassWrapper<LPNSimulationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LPNSimulationPass)

  LPNSimulationPass() = default;
  LPNSimulationPass(const LPNSimulationPass &other) : PassWrapper(other) {}

  StringRef getArgument() const final { return "lpn-simulate"; }
  StringRef getDescription() const final {
    return "Simulate LPN networks";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (Operation &op : module.getBody()->getOperations()) {
      auto net = dyn_cast<NetOp>(op);
      if (!net)
        continue;
      if (failed(simulateNet(net, module, static_cast<double>(maxTime))))
        signalPassFailure();
    }
  }

  Option<int64_t> maxTime{
      *this, "max-time", llvm::cl::desc("Max simulation time"),
      llvm::cl::init(1000)};
};

} // namespace

std::unique_ptr<Pass> mlir::lpn::createLPNSimulationPass() {
  return std::make_unique<LPNSimulationPass>();
}
