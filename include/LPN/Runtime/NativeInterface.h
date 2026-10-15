#ifndef LPN_NATIVE_INTERFACE_H
#define LPN_NATIVE_INTERFACE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bind a borrowed immutable array. Its storage must remain alive and unchanged
// until lpn_simulate returns. Bindings persist between sequential runs until
// replaced or cleared. Returns 0 on success, 2 on an invalid argument/state.
int lpn_bind_input_array(const char *name, int64_t size, const uint64_t *values);
int lpn_clear_input_arrays(void);

// Exported by every lpn-compile output; --shared makes it dynamically linkable.
// Uses the normal executable CLI arguments, e.g. --max-time=1000000000000.
// Calls on one loaded model/runtime must be serialized. Runtime errors have
// the same fatal-error behavior as the standalone native executable.
int lpn_simulate(int argc, char **argv);

// Completion time from the most recent run, in the model's time units.
double lpn_simulated_time(void);

#ifdef __cplusplus
}
#endif
#endif
