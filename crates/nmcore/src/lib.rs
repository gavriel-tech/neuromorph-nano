//! Library side of the Neuromorph Nano program crate: code shared by the product binary
//! `neuromorph-nano` and the lab tool `nano-lab`.
/// Comment-tolerant JSON reading and byte-exact editing of settings files (moved from the program
/// so the MCP onboarding's tests run with the library; Roadmap Phase 4.6 (2026-11-25)).
pub mod jsonc;
/// The MCP clients `setup` and `connect` write for (Roadmap Phase 4.6 (2026-11-25)).
pub mod mcp_clients;
// 2026-11-26: the engine self-test lives in nmengn, so the lab (nmlab) can use it without depending
// on nmcore, and nmcore can carry the lab inside the one program (`neuromorph-nano lab`).
pub use nmengn::selftest;
