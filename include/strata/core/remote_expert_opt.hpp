#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    bool init(std::string& err);        // after all helper caches are open
    /// The model's experts a token and width (default Flash-Next's 10 and 2560; Qwen3.6: 8 and 2048), before init.
    bool set_shape(int k, int64_t n_embd, std::string& err);
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    /// STRATA_HELPER_EXCHANGE: swaps whose evicted expert took its replacement's place in the RAM copy, and those that
    /// could not (cumulative).
    int64_t helper_exchanged() const { return helper_exchanged_; }
    int64_t helper_exchange_failed() const { return helper_exchange_failed_; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

private:
    friend class RemoteExperts;
    static size_t metadata_bytes();
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    std::vector<Peer> peers_;
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
    int k_ = 10;                         ///< experts a token (set_shape)
    int64_t w_ = 2560;                   ///< the model's width (set_shape)
    int64_t helper_exchanged_ = 0, helper_exchange_failed_ = 0;
    // STRATA_HELPER_ASYNC: the page-locked copy-back buffers and the queued round's RAM exchanges
    struct PendingExchange { int32_t layer, in, out; uint8_t* bytes; };
    std::vector<uint8_t*> pool_;
    std::vector<PendingExchange> px_;
    bool live_ = false;
};
} // namespace strata::core
