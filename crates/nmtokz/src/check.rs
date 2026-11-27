//! Section check of the tokenizer (added in the mini; the rest of `src/` is a byte copy of
//! emma-labs `native/rscore/tokenz`, build record runs/2026-11-07-build-record-copied.md): the first eight texts of the copied Qwen3
//! tokenizer golden encode to the exact ids and decode to the exact bytes. The full golden
//! (64 texts) is the S6 gate run, not this check.
use crate::Tokenizer;
use crate::json;

/// One section check: (sub-component id, check returning ok or one fault line).
pub type Check = (&'static str, fn() -> Result<(), String>);

/// Section checks of this crate.
#[must_use]
pub fn checks() -> Vec<Check> {
    vec![("nmtokz.model", model)]
}

fn model() -> Result<(), String> {
    let home = nmtype::nano_root();
    let tok = Tokenizer::open(&nmtype::models_dir(&home).join("qwen3-0.6b/files/tokenizer.json")).map_err(|e| e.to_string())?;
    let golden = home.join("tests/engine-reference/qwen3-0.6b/runs/A/tokenizer");
    let inputs = json::parse(&std::fs::read(golden.join("inputs.u8")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let texts = inputs.array().map_err(|e| e.to_string())?;
    for (i, t) in texts.iter().take(8).enumerate() {
        let ids = tok.encode(t.string().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let want = std::fs::read(golden.join(format!("text{i}-ids.i32"))).map_err(|e| e.to_string())?;
        let got: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        if got != want {
            return Err(format!("nmtokz.model: ids differ on golden text {i}"));
        }
        let dec = std::fs::read(golden.join(format!("text{i}-decode.u8"))).map_err(|e| e.to_string())?;
        if tok.decode(&ids).as_bytes() != dec.as_slice() {
            return Err(format!("nmtokz.model: decode differs on golden text {i}"));
        }
    }
    Ok(())
}
