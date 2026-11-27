# Neuromorph Nano Contribution Reward Distribution System

**Neuromorph Nano** rewards the humans who actively build, refine, and advance it through a **dynamic, merit-based commercial licensing reward distribution system** scoped **exclusively to Neuromorph Nano**.

---

## 1. Neuromorph Nano Licensing Profit Split (Product-Scoped Only)

Net commercial licensing **profit** generated specifically by **Neuromorph Nano** (see [`COMMERCIAL.md`](COMMERCIAL.md)) is allocated across three pools. **Zero fixed cash or out-of-pocket payouts exist; every contributor monetary reward scales strictly inside 100% of this 30% Neuromorph Nano licensing profit pool.**

| Pool | Share | Scope & Purpose |
|---|---|---|
| **Gavriel Technologies** | **55%** | Founding stewardship and operations (Luis Jake Gabriel III) |
| **Active Human Contributors (Reward Distribution Pool)** | **30%** | Distributed dynamically at each payout period proportionally to each human contributor's audited active share for that period |
| **Investors** | **15%** | Capital partners backing Neuromorph Nano |

> **Strictly Neuromorph Nano Licensing Profit Only:** This pool comes **exclusively from the net commercial licensing profit of Neuromorph Nano** (never Gavriel Technologies as a whole). It grants **no ownership, equity, stock, profit share, or claim** in Gavriel Technologies as a company or in any other Gavriel Technologies product, research line, or project.

---

## 2. How the Dynamic Per-Period Reward Distribution Works

At the close of each commercial licensing payout period, this repository (`gavriel-tech/neuromorph-nano` on GitHub, where pull requests are merged into `main`; see [How this repository works](CONTRIBUTING.md#how-this-repository-works)) is audited fresh at `HEAD` of `main` via [`tools/audit-pr-credit.ps1`](tools/audit-pr-credit.ps1). Payout shares apply **for that payout period only**; the next payout period is re-audited fresh against the live codebase, so contributor shares evolve dynamically as code grows, gets refined, or gets replaced:

1. **Founder Baseline (`v0.1.0`):** Because **Luis Jake Gabriel III** wrote 100% of the initial `v0.1.0` production codebase (21 zero-dependency Rust crates, 12 multi-arch CUDA fatbins, 4 parameter-mapped base cartridges, 25 MCP tools, continuous learning engine, and 11-page local interface), Luis starts with **100% of the 30% Contributor Pool** (plus **55%** for Gavriel Technologies).
2. **Surviving Functional Production Code (Automatic `git blame` + Fixed Folder Weights):**
   - Every surviving functional line of production code at `HEAD` is measured automatically by `tools/audit-pr-credit.ps1 -CodeShare` using **deterministic folder criticality weights** (zero subjective quality scoring, 100% reproducible by anyone):

   | Layer | Repository Paths | Folder Weight |
   |---|---|---|
   | **Core Rust Engine** | `crates/` (`std`-only organism, retrieval, memory, learning, MCP) | **`2.0x`** |
   | **Compute Kernels** | `kernels/` (CUDA kernels, quantization, compute backends) | **`2.0x`** |
   | **Cartridges & Parameter Maps** | `designs/`, `library/` | **`2.0x`** |
   | **Functional Local Interface** | `interface/src/` (live visualizations, UX/UI engineering) | **`1.0x`** |
   | **Build & Release Tooling** | `tools/` (installers, packaging, verification) | **`0.5x`** |
   | **Excluded (`0.0x`)** | `tests/`, `interface/tests/`, `interface/scripts/`, `.md`, `.txt`, `docs/`, **all code comments**, blank lines, `part.json`, `INTEGRITY.md`, lockfiles | **`0.0x`** |

3. **Refining Old Lines & Denser Logic (Never Punish Compression):**
   - When a contributor refactors `500` older lines into `120` cleaner, faster, denser lines that pass all checks, `git blame` automatically transfers the `120` surviving weighted lines to them, and the maintainer may record a period **Refinement Credit** in [`contributors-ledger.json`](contributors-ledger.json) up to the weighted lines cleanly eliminated so making the codebase smaller and faster is rewarded fairly.
4. **Hard Problems & Verified Binary Cartridges (Proportional Period Credit Inside the 30% Pool):**
   - For a compact, high-difficulty fix (such as a 30-line CUDA optimization) or a verified binary base cartridge / parameter-mapping technique (which has no source lines in git), the maintainer may record a transparent period **Breakthrough Credit** in [`contributors-ledger.json`](contributors-ledger.json) (measured in weighted code units and backed by a merged PR and reproducible held-out benchmark proof).
   - Every refinement and breakthrough credit is **strictly proportional and percentage-based inside the 30% licensing pool** (zero fixed-dollar rewards, zero non-scaling payouts) and applies to that payout period's audit.
5. **Direct Period Payout Formula (Always Sums to 100% of the 30% Pool):**
   - `contributor_period_score = surviving_weighted_production_lines + audited_period_refinement_and_breakthrough_credits`
   - `raw_pool_share = contributor_period_score / sum(all_contributors_period_scores)`
   - Contributors meeting the minimum eligibility threshold (`raw_pool_share >= 0.25%`) share **100.00% of the 30% Neuromorph Nano licensing profit pool** proportionally over the sum of eligible scores (`contributor_period_score / sum(eligible_period_scores)`), so **every cent of the 30% pool is cleanly allocated** with zero unallocated dust (`$0` paid if net Neuromorph Nano licensing profit is `$0` or negative).

---

## 3. Human-Controller Rule & Anti-Cheat Enforcement

- **Humans Exclusively:** Only verified human contributors can participate in the reward distribution pool and receive disbursements.
- **AI & Coding Agents Welcome:** Contributors may use Claude Code, Antigravity, Cursor, Codex, or any other AI tool. **100% of verified credit goes to the human controller** who directed, reviewed, tested, verified, and owns the merged pull request. Autonomous spam bots receive zero credit.
- **Production Functional Code Only:**
  - Every merged contribution must pass all required gates (`cargo clippy -D warnings`, `cargo test`, engine `check`, `e2e`, and held-out split gates where applicable).
  - Non-functional text (`.md`/`.txt`, docs, **code comments**, blank lines, generated files, and `tests/` fixtures) is automatically scored `0.0x` by `tools/audit-pr-credit.ps1`; useless or bloated code is rejected at PR review and never merged.
- **Zero-Tolerance Anti-Cheat Rules:**
  - **No Bloat, Churn Farming, or PR Splitting:** Useless, verbose, or mechanically rewritten code does not get merged; attempting to game metrics by splitting PRs or adding filler disqualifies the contributor.
  - **No Sybil / Multi-Account Gaming:** One human controller equals one contributor identity. Using multiple or proxy accounts results in immediate forfeiture.
  - **No Benchmark Gaming:** All performance, accuracy, and cartridge claims must reproduce on held-out gates (`20/20` decoy withholding, zero regressions).
  - **Active Survival:** Code reverted due to regressions, bugs, or security defects, or replaced by better code, drops off automatically in the next period's live `git blame` audit.
- **Minimum Payout Threshold:** To prevent micro-payout fragmentation, a contributor must hold at least **`0.25%`** of active pool share (and meet standard minimum payout transfer amounts) to receive a disbursement for that period.

---

## 4. Program Terms, Tax Responsibility & US/EU Legal Safeguards

1. **Merit-Based Contribution Reward Distribution System, Not Equity, Contest/Lottery, or Employment:** This program is a **discretionary, merit-based contribution reward distribution system** awarding periodic monetary rewards/prizes funded exclusively from collected net commercial licensing **profit** of **Neuromorph Nano**. It is **not** an offer of stock, shares, equity, tokens, securities, investment contracts, partnership, joint venture, sweepstakes, lottery, or employment in Gavriel Technologies or any affiliate, and grants zero voting, governance, ownership, or revenue/profit rights in Gavriel Technologies as a whole or in any other Gavriel Technologies product or project. **No purchase or entry fee is required and no element of chance is involved**; rewards are determined strictly by objective technical merit passing automated engineering gates.
2. **Voluntary Independent Participation (No Employment Relationship):** Participation is 100% voluntary and independent. Contributors have zero obligation to contribute, zero set hours, zero exclusivity, and zero management direction from Gavriel Technologies. Nothing in this program creates an employer-employee, worker, agency, or contractor relationship under U.S. federal/state law or EU member-state employment directives.
3. **Eligibility (18+, Sanctions & Void Where Prohibited):** Open solely to natural persons who have reached **age 18** (or the legal age of majority in their jurisdiction of residence, whichever is higher). Individuals or entities located in, or subject to, **U.S. OFAC** or **European Union Consolidated Sanctions** lists are ineligible. **Void where prohibited or restricted by applicable law.**
4. **Recipient Solely Responsible for All U.S., EU & Local Taxes:** Every monetary reward recipient is **solely and exclusively responsible** for reporting and paying all income, reward/prize, self-employment, social security, VAT, GST, or other taxes, duties, or filings arising from any distribution in their jurisdiction (including U.S. federal/state/local taxes and EU/national taxes). Luis Jake Gabriel III and Gavriel Technologies assume **zero liability** for any recipient's taxes and provide no tax or legal advice.
5. **GDPR-Compliant Payout Processing, Compliance Forms & Forfeiture:** Personal identity, payout, and tax details (such as IRS W-9 for U.S. persons or W-8BEN for non-U.S./EU persons) are **not** collected during normal GitHub contribution. They are collected solely when an eligible recipient claims a monetary reward, processed via a GDPR-compliant third-party payout provider solely for reward disbursement, U.S. OFAC / EU sanctions screening, and statutory tax reporting (withholding only where strictly required by law), and retained only as required by applicable law. Any distribution where required compliance or payout details are not provided within 90 days of notice is forfeited and returned to the pool or Gavriel Technologies.
6. **Neuromorph Nano Licensing Profit Only, Sole Discretion & Right to Modify:** Monetary reward disbursements are funded **exclusively from actual realized net commercial licensing profit of Neuromorph Nano** (collected Neuromorph Nano commercial licensing revenue minus Neuromorph Nano's direct operating, infrastructure, hosting, legal, compliance, payment-processing, payout-platform, refund, chargeback, and direct tax costs) and explicitly **exclude** Gavriel Technologies as a whole, consulting, custom integration services, hardware, grants, investor capital, or any other Gavriel Technologies product or project. There is **no guarantee** of any profit or reward amount in any period. Luis Jake Gabriel III (Gavriel Technologies) retains sole and final discretion over merge acceptance, per-period active-survival audits, refinement/breakthrough credits, anti-cheat enforcement, payout timing, and the right to modify, suspend, or terminate this reward distribution system at any time without liability or obligation. Participation creates no fiduciary duty and no right to audit or inspect Gavriel Technologies' private financial records.
7. **Independent IP & Licensing Terms:** All pull requests, code, kernels, cartridges, and parameter maps merged into this repository remain irrevocably governed by [`LICENSE`](LICENSE), [`COMMERCIAL.md`](COMMERCIAL.md), and the Contributor Licensing terms in [`CONTRIBUTING.md`](CONTRIBUTING.md#contributor-licensing), completely independent of contributor standing or reward distributions.

Questions or commercial licensing inquiries: `gavriel.techno@gmail.com`.
