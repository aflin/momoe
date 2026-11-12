#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace strata::core {
// A committed input row must correspond to an output the caller can expose.
// The last exposed output remains the next unconsumed token, as in the scalar loop.
template<class Token, class IsEos>
int serve_stop_bound(int accepted, int64_t remaining, const Token* outputs, IsEos is_eos) {
    const int count = static_cast<int>(std::min<int64_t>(accepted, std::max<int64_t>(0, remaining)));
    for (int i = 0; i < count; ++i)
        if (is_eos(outputs[i])) return i + 1;
    return count;
}

// #4677: the bound is on by default on every backend; STRATA_SERVE_STOP_BOUND=0 (the variable's value, passed in)
// turns it off.  It only limits which accepted rows enter the cached conversation: the tokens a request emits are the
// same either way.
inline bool serve_stop_bound_on(const char* env_value) {
    return env_value == nullptr || std::strcmp(env_value, "0") != 0;
}

// A window that finishes after the server's STOP has arrived (the client left, a stop string was found, the server ended its
// pass) is dropped whole: no commit, no tokens printed.  The server throws away every token after its STOP (it drains the
// engine to DONE), so nothing a client receives changes; what changes is the state the engine keeps: it ends at the rows
// the client was sent, not a window or two past them, and the next request that continues the conversation resumes from it
// instead of falling back to the newest checkpoint and reading the reply again.  Dropped only when its commit can be left
// undone: not a merged window (its rows are prompt rows), not the request's first window, and not a window whose state the
// one-token path advanced already (Verifier::commit_deferred()).  A request that ends by its cap or EOS
// never reaches this (the stop flag is not set).  STRATA_SERVE_STOP_SKIP=0 turns it off.
inline bool serve_skip_stopped_window(bool enabled, bool stop_seen, bool merged, bool first_window, bool commit_deferred) {
    return enabled && stop_seen && !merged && !first_window && commit_deferred;
}
inline bool serve_stop_skip_on(const char* env_value) {
    return env_value == nullptr || std::strcmp(env_value, "0") != 0;
}
}
