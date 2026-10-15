#include <algorithm>
#include <cmath>
#include <chrono>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Key {
  uint64_t kind;
  uint64_t value;
  bool operator==(const Key &other) const {
    return kind == other.kind && value == other.value;
  }
};

struct KeyHash {
  size_t operator()(const Key &key) const {
    uint64_t value = key.value ^ (key.kind * 0x9e3779b97f4a7c15ULL);
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    return static_cast<size_t>(value ^ (value >> 33));
  }
};

struct Token {
  using Fields = std::unordered_map<Key, int64_t, KeyHash>;
  Fields fields;
  uint64_t references = 1;
};

// Token payloads are immutable. Queues and transactions own handles, not
// copies of the field map. Empty tokens share one allocation-free sentinel.
Token emptyToken;
uint64_t payloadAllocations = 0, fieldCopies = 0, emptyTakes = 0;

Token *retain(Token *token) {
  if (token != &emptyToken)
    ++token->references;
  return token;
}

void release(Token *token) {
  if (token != &emptyToken && --token->references == 0)
    delete token;
}

struct Array {
  std::vector<uint64_t> elements;
  const uint64_t *borrowed = nullptr;
  size_t borrowedSize = 0;
  size_t size() const { return borrowed ? borrowedSize : elements.size(); }
  const uint64_t *data() const { return borrowed ? borrowed : elements.data(); }
};

struct Place {
  struct Scheduled {
    double time;
    uint64_t order;
    Token *token;
  };
  struct LaterFirst {
    bool operator()(const Scheduled &a, const Scheduled &b) const {
      return a.time != b.time ? a.time > b.time : a.order > b.order;
    }
  };
  std::string name;
  struct Batch {
    Token *token;
    uint64_t count;
  };
  // When a queue contains only empty tokens, no queue entries are needed.
  // Mixed queues use runs of identical immutable handles to preserve FIFO.
  uint64_t emptyReady = 0, readyCount = 0;
  std::deque<Batch> ready;
  std::priority_queue<Scheduled, std::vector<Scheduled>, LaterFirst> scheduled;
  uint64_t nextOrder = 0;

  ~Place() {
    for (const Batch &batch : ready)
      release(batch.token);
    while (!scheduled.empty()) {
      release(scheduled.top().token);
      scheduled.pop();
    }
  }

  void materializeEmpties() {
    if (emptyReady) {
      ready.push_back({&emptyToken, emptyReady});
      emptyReady = 0;
    }
  }

  void pushBack(Token *token) {
    ++readyCount;
    if (token == &emptyToken && ready.empty()) {
      ++emptyReady;
      return;
    }
    materializeEmpties();
    if (!ready.empty() && ready.back().token == token)
      ++ready.back().count;
    else
      ready.push_back({retain(token), 1});
  }

  void pushFront(Token *token) {
    ++readyCount;
    if (token == &emptyToken && ready.empty()) {
      ++emptyReady;
      return;
    }
    materializeEmpties();
    if (!ready.empty() && ready.front().token == token)
      ++ready.front().count;
    else
      ready.push_front({retain(token), 1});
  }

  Token *front() {
    return emptyReady ? &emptyToken : ready.front().token;
  }

  // Transfer one owned handle out of the queue.
  Token *popFront() {
    --readyCount;
    if (emptyReady) {
      --emptyReady;
      ++emptyTakes;
      return &emptyToken;
    }
    Token *token = ready.front().token;
    if (--ready.front().count == 0)
      ready.pop_front();
    else
      retain(token);
    if (token == &emptyToken)
      ++emptyTakes;
    return token;
  }

  void commit(double time) {
    while (!scheduled.empty() && scheduled.top().time <= time + 1e-12) {
      // priority_queue::top returns a const reference.
      Token *token = scheduled.top().token;
      pushBack(token);
      scheduled.pop();
      release(token);
    }
  }
};

struct Transaction {
  std::vector<std::pair<Place *, Token *>> taken;
  std::vector<std::tuple<Place *, Token *, double>> emitted;
  std::vector<Token *> tokens;
  std::vector<std::unique_ptr<Array>> arrays;
  std::jmp_buf blocked;

  ~Transaction() {
    for (Token *token : tokens)
      release(token);
    for (auto &entry : taken)
      release(entry.second);
  }
  Token *makeToken(Token::Fields fields = {}) {
    if (fields.empty())
      return &emptyToken;
    Token *token = new Token{std::move(fields)};
    ++payloadAllocations;
    tokens.push_back(token);
    return token;
  }
  Token *holdToken(Token *token) {
    if (token != &emptyToken)
      tokens.push_back(retain(token));
    return token;
  }
  Array *makeArray(Array array = {}) {
    arrays.push_back(std::make_unique<Array>(std::move(array)));
    return arrays.back().get();
  }
};

std::vector<std::unique_ptr<Place>> places;
std::unordered_map<const uint64_t *, std::unique_ptr<Array>> staticArrays;
std::unordered_map<const uint64_t *, std::unique_ptr<Array>> staticPlaceArrays;
std::unordered_map<std::string, std::unique_ptr<Array>> inputArrays;
bool inRun = false;
Transaction *active = nullptr;
double currentTime = 0.0;
uint64_t guardChecks = 0, bodyExecutions = 0, rollbacks = 0;

[[noreturn]] void invalid(const char *message) {
  std::fprintf(stderr, "lpn native runtime: %s\n", message);
  std::exit(2);
}

Transaction &transaction() {
  if (!active)
    invalid("operation outside a transition");
  return *active;
}

Place &placeAt(int64_t index) {
  if (index < 0 || static_cast<size_t>(index) >= places.size())
    invalid("invalid place index");
  return *places[static_cast<size_t>(index)];
}

void commitReady() {
  for (auto &place : places)
    place->commit(currentTime);
}

// Portable little-endian table format; producers need no model-specific API.
// Header: LPNTBL1\0, u64 table count. Each table: u64 name bytes, u64 element
// count, UTF-8 name bytes, then raw little-endian u64 elements.
void loadInputArrays(const char *path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) invalid("cannot open input-table file");
  auto end = stream.tellg();
  if (end < 16 || end > (int64_t(1) << 30)) invalid("invalid input-table file size");
  uint64_t remaining = static_cast<uint64_t>(end);
  stream.seekg(0);
  auto read = [&](char *output, size_t count) {
    if (count > remaining || !stream.read(output, count)) invalid("truncated input-table file");
    remaining -= count;
  };
  auto u64 = [&]() {
    unsigned char bytes[8];
    read(reinterpret_cast<char *>(bytes), 8);
    uint64_t value = 0;
    for (int n = 0; n < 8; ++n) value |= uint64_t(bytes[n]) << (8*n);
    return value;
  };
  char magic[8]; read(magic, 8);
  if (std::memcmp(magic, "LPNTBL1\0", 8)) invalid("unsupported input-table format");
  uint64_t tables = u64();
  if (tables > 10000) invalid("too many input tables");
  inputArrays.clear();
  for (uint64_t n = 0; n < tables; ++n) {
    uint64_t nameSize = u64(), count = u64();
    if (!nameSize || nameSize > 1024 || nameSize > remaining)
      invalid("invalid input-table name length");
    std::string name(nameSize, '\0'); read(name.data(), nameSize);
    if (name.find('\0') != std::string::npos || inputArrays.count(name))
      invalid("duplicate or invalid input-table name");
    if (count > remaining / 8) invalid("truncated input-table array");
    auto array = std::make_unique<Array>();
    array->elements.reserve(count);
    for (uint64_t k = 0; k < count; ++k) array->elements.push_back(u64());
    inputArrays.emplace(name, std::move(array));
  }
  if (remaining) invalid("trailing bytes in input-table file");
}

bool tryFire(void (*transition)(), bool (*enabled)()) {
  ++guardChecks;
  {
    // Count-only predicates allocate no transaction on the heap. A scratch
    // arena supports immutable token/array expressions in manual predicates.
    Transaction scratch;
    active = &scratch;
    volatile bool ready = false;
    if (setjmp(scratch.blocked) == 0)
      ready = enabled();
    active = nullptr;
    if (!ready)
      return false;
  }
  ++bodyExecutions;
  auto *txn = new Transaction();
  active = txn;
  bool fired = setjmp(txn->blocked) == 0;
  if (fired) {
    transition();
    for (auto &entry : txn->emitted) {
      Place *place = std::get<0>(entry);
      place->scheduled.push(Place::Scheduled{std::get<2>(entry),
                                             place->nextOrder++,
                                             retain(std::get<1>(entry))});
    }
  } else {
    ++rollbacks;
    for (auto it = txn->taken.rbegin(); it != txn->taken.rend(); ++it)
      it->first->pushFront(it->second);
  }
  active = nullptr;
  delete txn;
  if (fired)
    commitReady();
  return fired;
}

double nextScheduledTime() {
  double next = std::numeric_limits<double>::infinity();
  for (const auto &place : places)
    if (!place->scheduled.empty())
      next = std::min(next, place->scheduled.top().time);
  return next;
}

} // namespace

extern "C" {

int lpn_bind_input_array(const char *name, int64_t size, const uint64_t *values) {
  if (inRun || !name || !*name || std::strlen(name)>1024 || size<0 || (size && !values))
    return 2;
  auto array = std::make_unique<Array>();
  array->borrowed = values;
  array->borrowedSize = static_cast<size_t>(size);
  inputArrays[name] = std::move(array);
  return 0;
}

int lpn_clear_input_arrays() {
  if (inRun) return 2;
  inputArrays.clear();
  return 0;
}

void *lpn_input_array(const char *name) {
  auto found = inputArrays.find(name);
  if (found==inputArrays.end()) invalid("missing named input-array binding");
  return found->second.get();
}

double lpn_simulated_time() { return currentTime; }

void *lpn_place(int64_t index) { return &placeAt(index); }

void lpn_check_step(int64_t step) {
  if (step <= 0)
    invalid("scf.for step must be positive");
}

void *lpn_take(void *rawPlace) {
  auto &txn = transaction();
  auto *place = static_cast<Place *>(rawPlace);
  if (!place->readyCount)
    longjmp(txn.blocked, 1);
  Token *token = place->popFront();
  txn.taken.emplace_back(place, token);
  return token;
}

void *lpn_peek(void *rawPlace) {
  auto &txn = transaction();
  auto *place = static_cast<Place *>(rawPlace);
  if (!place->readyCount)
    longjmp(txn.blocked, 1);
  return txn.holdToken(place->front());
}

// Borrowing is restricted by the compiler to enabling predicates and scalar
// helpers with no token-flow effects. A general body peek retains snapshot
// semantics, including when a later take removes the original queue entry.
void *lpn_peek_readonly(void *rawPlace) {
  auto *place = static_cast<Place *>(rawPlace);
  if (!place->readyCount)
    longjmp(transaction().blocked, 1);
  return place->front();
}

void lpn_guard(bool condition) {
  if (!condition)
    longjmp(transaction().blocked, 1);
}

void lpn_emit(void *rawPlace, void *rawToken, double delay) {
  auto &txn = transaction();
  double ready = currentTime + std::max(0.0, delay);
  if (!std::isfinite(delay) || !std::isfinite(ready))
    invalid("emit delay must produce a finite ready time");
  txn.emitted.emplace_back(static_cast<Place *>(rawPlace),
                           static_cast<Token *>(rawToken), ready);
}

int64_t lpn_count(void *rawPlace) {
  return static_cast<int64_t>(static_cast<Place *>(rawPlace)->readyCount);
}

void *lpn_token_create() { return transaction().makeToken(); }

void *lpn_token_clone(void *rawToken) {
  return transaction().holdToken(static_cast<Token *>(rawToken));
}

void *lpn_token_set(void *rawToken, int64_t kind, int64_t key,
                    int64_t value) {
  const auto &original = static_cast<Token *>(rawToken)->fields;
  Token::Fields copy = original;
  fieldCopies += !original.empty();
  copy[Key{static_cast<uint64_t>(kind), static_cast<uint64_t>(key)}] =
      value;
  return transaction().makeToken(std::move(copy));
}

int64_t lpn_token_get(void *rawToken, int64_t kind, int64_t key) {
  auto &fields = static_cast<Token *>(rawToken)->fields;
  auto it = fields.find(Key{static_cast<uint64_t>(kind),
                            static_cast<uint64_t>(key)});
  return it == fields.end() ? 0 : it->second;
}

void *lpn_array_create(int64_t size, const uint64_t *elements) {
  if (size < 0)
    invalid("negative array size");
  Array array;
  if (size)
    array.elements.assign(elements, elements + size);
  return transaction().makeArray(std::move(array));
}

void *lpn_array_static(int64_t size, const uint64_t *elements) {
  auto &array = staticArrays[elements];
  if (!array) {
    array = std::make_unique<Array>();
    if (size)
      array->elements.assign(elements,elements+size);
  }
  return array.get();
}

// Place addresses are stable for a run. Resolve an immutable list of indices
// once, rather than rebuilding a list of handles in every guard and firing.
void *lpn_array_places_static(int64_t size, const uint64_t *indices) {
  // LLVM may merge identical integer literals and place-index globals. Their
  // addresses must not alias cache entries with different interpretations.
  auto &array = staticPlaceArrays[indices];
  if (!array) {
    array = std::make_unique<Array>();
    for (int64_t index = 0; index < size; ++index)
      array->elements.push_back(reinterpret_cast<uint64_t>(&placeAt(indices[index])));
  }
  return array.get();
}

void *lpn_array_alloc(int64_t size, uint64_t fill) {
  if (size < 0)
    invalid("negative array size");
  Array array;
  array.elements.assign(static_cast<size_t>(size), fill);
  return transaction().makeArray(std::move(array));
}

uint64_t lpn_array_get(void *rawArray, int64_t index) {
  auto &array = *static_cast<Array *>(rawArray);
  if (index < 0 || static_cast<size_t>(index) >= array.size())
    invalid("array.get index out of range");
  return array.data()[static_cast<size_t>(index)];
}

void *lpn_array_set(void *rawArray, int64_t index, uint64_t value) {
  Array copy = *static_cast<Array *>(rawArray);
  if (copy.borrowed) {
    copy.elements.assign(copy.borrowed, copy.borrowed+copy.borrowedSize);
    copy.borrowed = nullptr;
    copy.borrowedSize = 0;
  }
  if (index < 0 || static_cast<size_t>(index) >= copy.elements.size())
    invalid("array.set index out of range");
  copy.elements[static_cast<size_t>(index)] = value;
  return transaction().makeArray(std::move(copy));
}

int64_t lpn_array_len(void *rawArray) {
  return static_cast<int64_t>(static_cast<Array *>(rawArray)->size());
}

int lpn_run(int argc, char **argv, int64_t placeCount, const char **placeNames,
            const int64_t *initialCounts, int64_t transitionCount,
            void (**transitions)(), bool (**enabled)(), const char **transitionNames,
            int64_t literalCount,
            const char **literalNames) {
  double maxTime = 100000000.0;
  bool printTokens = false;
  bool printStats = false;
  bool trace = false;
  const char *inputPath = nullptr;
  std::string wallStartName,wallEndName;
  for (int idx = 1; idx < argc; ++idx) {
    if (std::strncmp(argv[idx], "--max-time=", 11) == 0) {
      char *end = nullptr;
      maxTime = std::strtod(argv[idx] + 11, &end);
      if (!end || *end || !std::isfinite(maxTime) || maxTime < 0)
        invalid("invalid --max-time value");
    } else if (std::strcmp(argv[idx], "--tokens") == 0) {
      printTokens = true;
    } else if (std::strcmp(argv[idx], "--stats") == 0) {
      printStats = true;
    } else if (std::strcmp(argv[idx], "--trace") == 0) {
      trace = true;
    } else if (std::strncmp(argv[idx], "--inputs=", 9) == 0) {
      if (inputPath) invalid("duplicate --inputs option");
      inputPath = argv[idx]+9;
    } else if (std::strncmp(argv[idx],"--wall-start=",13) == 0) {
      wallStartName = argv[idx]+13;
    } else if (std::strncmp(argv[idx],"--wall-end=",11) == 0) {
      wallEndName = argv[idx]+11;
    } else {
      std::fprintf(stderr,
                   "Usage: %s [--max-time=NUMBER] [--tokens] [--stats] [--trace] "
                   "[--wall-start=TRANSITION --wall-end=TRANSITION] [--inputs=FILE]\n", argv[0]);
      return 2;
    }
  }
  if (inRun) invalid("nested/concurrent simulation is unsupported");
  if (inputPath) loadInputArrays(inputPath);
  inRun = true;
  places.clear();
  staticArrays.clear();
  staticPlaceArrays.clear();
  places.reserve(static_cast<size_t>(placeCount));
  payloadAllocations = fieldCopies = emptyTakes = 0;
  for (int64_t idx = 0; idx < placeCount; ++idx) {
    if (initialCounts[idx] < 0)
      invalid("negative initial token count");
    auto place = std::make_unique<Place>();
    place->name = placeNames[idx];
    place->emptyReady = place->readyCount = initialCounts[idx];
    places.push_back(std::move(place));
  }
  currentTime = 0.0;
  int64_t wallStart = -1,wallEnd = -1;
  bool wallStarted = false,wallFinished = false;
  std::chrono::steady_clock::time_point wallClock;
  for (int64_t idx = 0; idx < transitionCount; ++idx) {
    if (!wallStartName.empty() && wallStartName == transitionNames[idx]) wallStart = idx;
    if (!wallEndName.empty() && wallEndName == transitionNames[idx]) wallEnd = idx;
  }
  if ((!wallStartName.empty() || !wallEndName.empty()) && (wallStart < 0 || wallEnd < 0))
    invalid("wall-time profiling needs valid --wall-start and --wall-end transitions");
  guardChecks = bodyExecutions = rollbacks = 0;
  while (currentTime <= maxTime) {
    bool progress = false;
    for (int64_t idx = 0; idx < transitionCount; ++idx)
      while (tryFire(transitions[idx], enabled[idx])) {
        if (idx == wallStart && !wallStarted) {
          wallClock = std::chrono::steady_clock::now();
          wallStarted = true;
        }
        if (idx == wallEnd && wallStarted && !wallFinished) {
          double seconds = std::chrono::duration<double>(
              std::chrono::steady_clock::now()-wallClock).count();
          std::printf("Wall window elapsed: %.9f seconds\n",seconds);
          wallFinished = true;
        }
        if (trace)
          std::printf("[t=%.3f] fired %s\n", currentTime, transitionNames[idx]);
        progress = true;
      }
    if (progress)
      continue;
    double next = nextScheduledTime();
    if (!std::isfinite(next) || next > maxTime)
      break;
    currentTime = next;
    commitReady();
  }
  std::printf("Simulation finished at t=%.3f ns\n", currentTime);
  if (printStats)
    std::printf("Guard checks: %llu; body executions: %llu; rollbacks: %llu\n",
        static_cast<unsigned long long>(guardChecks),
        static_cast<unsigned long long>(bodyExecutions),
        static_cast<unsigned long long>(rollbacks));
  if (printStats)
    std::printf("Token payload allocations: %llu; field copies on set: %llu; empty-token takes: %llu\n",
        static_cast<unsigned long long>(payloadAllocations),
        static_cast<unsigned long long>(fieldCopies),
        static_cast<unsigned long long>(emptyTakes));
  for (const auto &place : places) {
    std::printf("  place %s: %llu token(s)\n", place->name.c_str(),
                static_cast<unsigned long long>(place->readyCount));
    if (!printTokens)
      continue;
    auto printToken = [&](const Token *token) {
      std::printf("    token {");
      bool first = true;
      for (const auto &entry : token->fields) {
        if (!first)
          std::printf(", ");
        first = false;
        if (entry.first.kind == 0 && entry.first.value <
                                         static_cast<uint64_t>(literalCount))
          std::printf("'%s': %lld", literalNames[entry.first.value],
                      static_cast<long long>(entry.second));
        else
          std::printf("reg(%lld): %lld",
                      static_cast<long long>(entry.first.value),
                      static_cast<long long>(entry.second));
      }
      std::printf("}\n");
    };
    for (uint64_t n = 0; n < place->emptyReady; ++n)
      printToken(&emptyToken);
    for (const Place::Batch &batch : place->ready)
      for (uint64_t n = 0; n < batch.count; ++n)
        printToken(batch.token);
  }
  inRun = false;
  return 0;
}

} // extern "C"
