#include <algorithm>
#include <cmath>
#include <chrono>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
  std::unordered_map<Key, int64_t, KeyHash> fields;
};

struct Array {
  std::vector<uint64_t> elements;
};

struct Place {
  struct Scheduled {
    double time;
    uint64_t order;
    Token token;
  };
  struct LaterFirst {
    bool operator()(const Scheduled &a, const Scheduled &b) const {
      return a.time != b.time ? a.time > b.time : a.order > b.order;
    }
  };
  std::string name;
  std::deque<Token> ready;
  std::priority_queue<Scheduled, std::vector<Scheduled>, LaterFirst> scheduled;
  uint64_t nextOrder = 0;

  void commit(double time) {
    while (!scheduled.empty() && scheduled.top().time <= time + 1e-12) {
      // priority_queue::top returns a const reference.
      ready.push_back(scheduled.top().token);
      scheduled.pop();
    }
  }
};

struct Transaction {
  std::vector<std::pair<Place *, Token>> taken;
  std::vector<std::tuple<Place *, Token, double>> emitted;
  std::vector<std::unique_ptr<Token>> tokens;
  std::vector<std::unique_ptr<Array>> arrays;
  std::jmp_buf blocked;

  Token *makeToken(Token token = {}) {
    tokens.push_back(std::make_unique<Token>(std::move(token)));
    return tokens.back().get();
  }
  Array *makeArray(Array array = {}) {
    arrays.push_back(std::make_unique<Array>(std::move(array)));
    return arrays.back().get();
  }
};

std::vector<std::unique_ptr<Place>> places;
std::unordered_map<const uint64_t *, std::unique_ptr<Array>> staticArrays;
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

bool tryFire(void (*transition)(), bool (*enabled)()) {
  ++guardChecks;
  {
    // Count-only predicates allocate no transaction on the heap. A scratch
    // arena supports immutable token/array expressions in manual predicates.
    Transaction scratch;
    active = &scratch;
    bool ready = enabled();
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
                                             std::move(std::get<1>(entry))});
    }
  } else {
    ++rollbacks;
    for (auto it = txn->taken.rbegin(); it != txn->taken.rend(); ++it)
      it->first->ready.push_front(std::move(it->second));
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

void *lpn_place(int64_t index) { return &placeAt(index); }

void lpn_check_step(int64_t step) {
  if (step <= 0)
    invalid("scf.for step must be positive");
}

void *lpn_take(void *rawPlace) {
  auto &txn = transaction();
  auto *place = static_cast<Place *>(rawPlace);
  if (place->ready.empty())
    longjmp(txn.blocked, 1);
  Token token = std::move(place->ready.front());
  place->ready.pop_front();
  txn.taken.emplace_back(place, token);
  return txn.makeToken(std::move(token));
}

void *lpn_peek(void *rawPlace) {
  auto &txn = transaction();
  auto *place = static_cast<Place *>(rawPlace);
  if (place->ready.empty())
    longjmp(txn.blocked, 1);
  return txn.makeToken(place->ready.front());
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
                           *static_cast<Token *>(rawToken), ready);
}

int64_t lpn_count(void *rawPlace) {
  return static_cast<int64_t>(static_cast<Place *>(rawPlace)->ready.size());
}

void *lpn_token_create() { return transaction().makeToken(); }

void *lpn_token_clone(void *rawToken) {
  return transaction().makeToken(*static_cast<Token *>(rawToken));
}

void *lpn_token_set(void *rawToken, int64_t kind, int64_t key,
                    int64_t value) {
  Token copy = *static_cast<Token *>(rawToken);
  copy.fields[Key{static_cast<uint64_t>(kind), static_cast<uint64_t>(key)}] =
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

void *lpn_array_alloc(int64_t size, uint64_t fill) {
  if (size < 0)
    invalid("negative array size");
  Array array;
  array.elements.assign(static_cast<size_t>(size), fill);
  return transaction().makeArray(std::move(array));
}

uint64_t lpn_array_get(void *rawArray, int64_t index) {
  auto &elements = static_cast<Array *>(rawArray)->elements;
  if (index < 0 || static_cast<size_t>(index) >= elements.size())
    invalid("array.get index out of range");
  return elements[static_cast<size_t>(index)];
}

void *lpn_array_set(void *rawArray, int64_t index, uint64_t value) {
  Array copy = *static_cast<Array *>(rawArray);
  if (index < 0 || static_cast<size_t>(index) >= copy.elements.size())
    invalid("array.set index out of range");
  copy.elements[static_cast<size_t>(index)] = value;
  return transaction().makeArray(std::move(copy));
}

int64_t lpn_array_len(void *rawArray) {
  return static_cast<int64_t>(static_cast<Array *>(rawArray)->elements.size());
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
    } else if (std::strncmp(argv[idx],"--wall-start=",13) == 0) {
      wallStartName = argv[idx]+13;
    } else if (std::strncmp(argv[idx],"--wall-end=",11) == 0) {
      wallEndName = argv[idx]+11;
    } else {
      std::fprintf(stderr,
                   "Usage: %s [--max-time=NUMBER] [--tokens] [--stats] [--trace] "
                   "[--wall-start=TRANSITION --wall-end=TRANSITION]\n", argv[0]);
      return 2;
    }
  }
  places.clear();
  staticArrays.clear();
  places.reserve(static_cast<size_t>(placeCount));
  for (int64_t idx = 0; idx < placeCount; ++idx) {
    if (initialCounts[idx] < 0)
      invalid("negative initial token count");
    auto place = std::make_unique<Place>();
    place->name = placeNames[idx];
    for (int64_t count = 0; count < initialCounts[idx]; ++count)
      place->ready.emplace_back();
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
  for (const auto &place : places) {
    std::printf("  place %s: %zu token(s)\n", place->name.c_str(),
                place->ready.size());
    if (!printTokens)
      continue;
    for (const Token &token : place->ready) {
      std::printf("    token {");
      bool first = true;
      for (const auto &entry : token.fields) {
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
    }
  }
  return 0;
}

} // extern "C"
