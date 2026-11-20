#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// The RAM a runtime allocation (parking a conversation, saving or restoring a session file) may still take.
//
// available_host_bytes() is the smallest of
//   * MemAvailable (Windows: the available physical memory);
//   * the room under every memory limit of this process's cgroup and of each ancestor visible under
//     /sys/fs/cgroup (cgroup v2: memory.max and memory.high, each minus usage; cgroup v1:
//     memory.limit_in_bytes minus memory.usage_in_bytes);
//   * the operator's own cap, when one is set (--memory-limit-mib / STRATA_MEMORY_LIMIT_MIB): that many MiB minus
//     what the container uses now - the top visible cgroup's memory.current, or MemTotal - MemAvailable when the
//     cgroup files cannot be read.  For a container whose real limit sits on a parent cgroup it cannot see.
//
// Usage is memory.current as the kernel reports it, less one reclaimable kind: the clean inactive file cache of the
// same group (memory.stat inactive_file, v1 total_inactive_file, less dirty and writeback pages, never more than
// memory.current), which the kernel gives back at memory.high or the limit before it kills anything.  Nothing else is
// credited: active_file holds mlocked pages, and shmem is the pinned complements and the KV pool.  A memory.stat that
// is missing or unparsable gives no credit (it does not make the sample unknown).
// Pinned, locked and shared memory stays charged in full: a hard-limit kill is worse than a skipped park.
//
// Unknown telemetry is deliberately distinct from a measured zero.  A limit file that exists but cannot be parsed
// makes the sample unknown (fail closed); a file that is absent (no such controller, not mounted) is ignored.
// Not a reservation: other writers can take the room after the sample.

enum class MemorySource { meminfo, cgroup, flag };

const char* memory_source_name(MemorySource source);

/// One sample.  `available` is the smallest term; `source`, `limit` and `current` describe the term that was the
/// smallest (meminfo: MemTotal and MemTotal - MemAvailable; cgroup: that group's limit and memory.current; flag: the
/// operator's cap and the usage it was measured against).  `mem_available` is MemAvailable by itself (0 if unknown).
struct HostMemoryReading {
    uint64_t available = 0;
    MemorySource source = MemorySource::meminfo;
    uint64_t limit = 0;
    uint64_t current = 0;        ///< memory.current (meminfo: MemTotal - MemAvailable), before the credit
    uint64_t credit = 0;         ///< the clean inactive file cache counted as reclaimable (0 for meminfo)
    uint64_t mem_available = 0;
};

/// The files the sampler reads; the defaults are the real ones (tests point them at a fixture directory).
struct MemoryProbePaths {
    std::string meminfo = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
};

/// The operator's cap in bytes for the container/cgroup this engine runs in (0 = none).  Set once at start.
void set_memory_limit_bytes(uint64_t bytes);
uint64_t memory_limit_bytes();

/// Linux: reads `paths`.  `explicit_limit` 0 = no operator cap.  nullopt when the RAM cannot be determined.
std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit);

/// The real files (Windows: GlobalMemoryStatusEx) and the cap set by set_memory_limit_bytes().
std::optional<HostMemoryReading> sample_host_memory();

std::optional<uint64_t> available_host_bytes();

/// #3493: the conversation cache budget (MiB) --serve uses when --conversation-cache-mib was not given, from the RAM that is available
/// once the model is loaded: 40% of what is left above the parking floor, at most 8192 (four 100K-token agent conversations were
/// parked in 8 GiB in the report), in steps of 256, and 0 (off) when less than 1024 would be left to give.  Unknown telemetry: 0.
inline int64_t auto_conversation_cache_mib(std::optional<uint64_t> available, int64_t min_free_mib) {
    if (!available) return 0;
    const int64_t usable = (int64_t) (*available >> 20) - min_free_mib;
    if (usable <= 0) return 0;
    int64_t mib = usable * 2 / 5;
    if (mib > 8192) mib = 8192;
    mib -= mib % 256;
    return mib >= 1024 ? mib : 0;
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

/// #5697: the physical-RAM gate's eviction loop (generate.cpp, park_source).  `admit()` samples the memory reading and
/// says whether the park fits; `available()` is the reading admit() took last (nullopt: unknown); `evict()` drops the
/// oldest parked conversation that may go and returns its bytes (0: none could go).  Evicts until the park fits, at most
/// `max_evictions` entries, and stops at the first eviction whose memory did not show up in the reading: a gain under
/// half of the entry's bytes AND under 64 MiB (the C library kept it, or another process took it).  The field log of
/// #5697: the loop used to go on through every parked conversation, read no change, and refuse anyway with an empty
/// cache.  Now one entry is the most such a gate costs, and the conversations after it stay parked.
struct EvictForRam {
    bool fits = false;
    size_t evicted = 0;
    bool no_progress = false;   // stopped because an eviction gave nothing back to the reading
};
template <class Admit, class Available, class Evict>
EvictForRam evict_for_ram(Admit admit, Available available, Evict evict, size_t max_evictions) {
    EvictForRam r;
    r.fits = admit();
    while (!r.fits && r.evicted < max_evictions) {
        const std::optional<uint64_t> before = available();
        const size_t freed = evict();
        if (freed == 0) break;
        ++r.evicted;
        r.fits = admit();
        const std::optional<uint64_t> after = available();
        if (!r.fits && before && after) {
            const uint64_t gain = *after > *before ? *after - *before : 0;
            if (gain < freed / 2 && gain < (uint64_t(64) << 20)) {
                r.no_progress = true;
                break;
            }
        }
    }
    return r;
}

} // namespace strata::core
