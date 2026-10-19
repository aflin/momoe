#pragma once

#include <cstdint>

namespace strata::core {

/// A GPU expert tier the prompt path can hand rows to (Prefill::set_peer): the card it lives on, whether it holds
/// (layer, expert), and that blob's device address.  The --peer-device tier (PeerExperts) and, with
/// STRATA_PREFILL_HELPER_ROWS, a helper cache (RemoteExperts) are both one.
class ExpertHolder {
public:
    virtual ~ExpertHolder() = default;
    virtual bool valid() const = 0;
    virtual int device() const = 0;
    virtual bool has(int64_t layer, int64_t expert) const = 0;
    /// The device address of (layer, expert)'s blob on that card, or null when it is not held.
    virtual const uint8_t* slot_ptr(int64_t layer, int64_t expert) = 0;
    /// Whether the primary and this card can read each other's memory directly (NVLink / PCIe P2P).
    virtual bool p2p() const = 0;
};

}  // namespace strata::core
