//! The reader's side of the request's drone in its model form (lane R, 2026-11-18; master plan
//! the 2026-11-18 tentacle master plan, 3.1, Phase B2). The request's drone
//! (`nmpod::intake`) asks, per short prompt, the probability of each of a few labels as the next
//! token; this answers with one forward pass of the reader MicroModel (the base population the
//! drones read with, `crate::neural::model`) per prompt, reading only the label rows of the head
//! (`RunOpts::head_rows`), no generation. On the card the prompts go in one packed run (one
//! sequence each, `RunOpts::segments`); on the processor one run each. Neither takes the device
//! gate: the request is read inside callers that may hold it (the gate is not re-entrant), as the
//! dense lane's question embeddings run ungated. Installed only when the model form is on (`NANO_REQ_MODEL=on`).
use nmengn::model::{Model, RunOpts, values};
use nmengn::placement::Device;
use nmpod::intake::RequestModel;
use nmtype::{Fault, FaultKind};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn bad(m: impl Into<String>) -> Fault {
    Fault::new("nmapi.request_read", FaultKind::Config, m)
}

/// Label reads over the reader MicroModel.
pub struct ReaderLabels {
    model: Arc<Model>,
    dev: Device,
}
impl std::fmt::Debug for ReaderLabels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderLabels").field("dev", &self.dev).finish_non_exhaustive()
    }
}
impl ReaderLabels {
    /// Over `model` on `dev`.
    #[must_use]
    pub fn new(model: Arc<Model>, dev: Device) -> Self {
        Self { model, dev }
    }
    /// The first tokens of each label's spellings (as written, capitalised, after a space), each
    /// with its label's index; a token two labels share is refused.
    fn label_rows(&self, labels: &[&str]) -> Result<Vec<(usize, usize)>, Fault> {
        let mut rows: Vec<(usize, usize)> = Vec::new();
        for (k, l) in labels.iter().enumerate() {
            let cap: String = l.chars().take(1).flat_map(char::to_uppercase).chain(l.chars().skip(1)).collect();
            for v in [(*l).to_owned(), cap, format!(" {l}")] {
                let ids = self.model.pop.tokenizer.encode(&v).map_err(|e| bad(e.to_string()))?;
                let r = ids.first().and_then(|x| usize::try_from(*x).ok()).ok_or_else(|| bad(format!("no token for {v:?}")))?;
                match rows.iter().find(|(_, x)| *x == r) {
                    Some((j, _)) if *j != k => return Err(bad(format!("labels share a first token: {labels:?}"))),
                    Some(_) => {}
                    None => rows.push((k, r)),
                }
            }
        }
        Ok(rows)
    }
}

/// Each label's probability: a softmax over every spelling's logit, summed per label.
fn label_mass(rows: &[(usize, usize)], logit: impl Fn(usize) -> f32, labels: usize) -> Vec<f32> {
    let xs: Vec<f32> = rows.iter().map(|(_, r)| logit(*r)).collect();
    let m = xs.iter().copied().fold(f32::MIN, f32::max);
    let es: Vec<f64> = xs.iter().map(|x| f64::from(x - m).exp()).collect();
    let s: f64 = es.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    let mut out = vec![0.0_f64; labels];
    for ((k, _), e) in rows.iter().zip(&es) {
        if let Some(o) = out.get_mut(*k) {
            *o += e / s;
        }
    }
    #[expect(clippy::cast_possible_truncation, reason = "probabilities")]
    out.into_iter().map(|x| x as f32).collect()
}

impl RequestModel for ReaderLabels {
    fn label_probs(&self, items: &[(String, Vec<&'static str>)]) -> Result<(Vec<Vec<f32>>, f64), Fault> {
        let _job = nmengn::chanrec::job("request");
        let m = &self.model;
        let head = (m.role_node("head")?, "logits".to_owned());
        let per: Vec<Vec<(usize, usize)>> = items.iter().map(|(_, l)| self.label_rows(l)).collect::<Result<_, _>>()?;
        let ids: Vec<Vec<i32>> = items.iter().map(|(p, _)| m.pop.tokenizer.encode(p).map_err(|e| bad(e.to_string()))).collect::<Result<_, _>>()?;
        let t = Instant::now();
        let mut out = Vec::with_capacity(items.len());
        if self.dev == Device::Gpu {
            // One packed run: every prompt a sequence, the head reading the union of label rows.
            let mut union: Vec<usize> = Vec::new();
            for (_, r) in per.iter().flatten() {
                if !union.contains(r) {
                    union.push(*r);
                }
            }
            let all: Vec<i32> = ids.iter().flatten().copied().collect();
            let opts = RunOpts { segments: Some(Arc::new(ids.iter().map(Vec::len).collect())), head_rows: Arc::new(union.clone()), ..RunOpts::default() };
            let res = m.forward(self.dev, &all, std::slice::from_ref(&head), &opts)?;
            let l = values(res.get(&head).ok_or_else(|| bad("no logits"))?)?;
            for (k, ((_, labels), rows)) in items.iter().zip(&per).enumerate() {
                let row = l.get(k * union.len()..(k + 1) * union.len()).ok_or_else(|| bad("short logits"))?;
                out.push(label_mass(rows, |r| union.iter().position(|u| *u == r).and_then(|i| row.get(i)).copied().unwrap_or(f32::MIN), labels.len()));
            }
        } else {
            for (((_, labels), rows), ids) in items.iter().zip(&per).zip(&ids) {
                let want: Vec<usize> = rows.iter().map(|(_, r)| *r).collect();
                let opts = RunOpts { head_rows: Arc::new(want.clone()), ..RunOpts::default() };
                let res = m.forward(self.dev, ids, std::slice::from_ref(&head), &opts)?;
                let l = values(res.get(&head).ok_or_else(|| bad("no logits"))?)?;
                out.push(label_mass(rows, |r| want.iter().position(|u| *u == r).and_then(|i| l.get(i)).copied().unwrap_or(f32::MIN), labels.len()));
            }
        }
        Ok((out, t.elapsed().as_secs_f64() * 1000.0))
    }
    /// On the card: the shared head prefilled once into a key and value cache, then every item's
    /// own tokens packed side by side in one run reading that head (as `card_rerank_batch`). An
    /// item whose whole prompt does not start with the head's tokens sends the set the plain way.
    fn label_probs_shared(&self, prefix: &str, items: &[(String, Vec<&'static str>)]) -> Result<(Vec<Vec<f32>>, f64), Fault> {
        let _job = nmengn::chanrec::job("request");
        let m = &self.model;
        let whole: Vec<(String, Vec<&'static str>)> = items.iter().map(|(s, l)| (format!("{prefix}{s}"), l.clone())).collect();
        if self.dev != Device::Gpu {
            return self.label_probs(&whole);
        }
        let tok = |s: &str| m.pop.tokenizer.encode(s).map_err(|e| bad(e.to_string()));
        let head_ids = tok(prefix)?;
        let full: Vec<Vec<i32>> = whole.iter().map(|(p, _)| tok(p)).collect::<Result<_, _>>()?;
        if head_ids.is_empty() || full.iter().any(|f| f.len() <= head_ids.len() || !f.starts_with(&head_ids)) {
            return self.label_probs(&whole);
        }
        let per: Vec<Vec<(usize, usize)>> = items.iter().map(|(_, l)| self.label_rows(l)).collect::<Result<_, _>>()?;
        let mut union: Vec<usize> = Vec::new();
        for (_, r) in per.iter().flatten() {
            if !union.contains(r) {
                union.push(*r);
            }
        }
        let head = (m.role_node("head")?, "logits".to_owned());
        let fin = (m.role_node("final_norm")?, "y".to_owned());
        let t = Instant::now();
        let base = Arc::new(nmengn::model::Cache::default());
        m.forward_at(self.dev, &head_ids, 0, std::slice::from_ref(&fin), &RunOpts { cache: Some(Arc::clone(&base)), ..RunOpts::default() })?;
        let rests: Vec<&[i32]> = full.iter().map(|f| f.get(head_ids.len()..).unwrap_or(&[])).collect();
        let packed: Vec<i32> = rests.iter().flat_map(|r| r.iter().copied()).collect();
        let opts = RunOpts {
            cache: Some(Arc::clone(&base)),
            segments: Some(Arc::new(rests.iter().map(|r| r.len()).collect())),
            head_rows: Arc::new(union.clone()),
            ..RunOpts::default()
        };
        let res = m.forward_at(self.dev, &packed, head_ids.len(), std::slice::from_ref(&head), &opts)?;
        let l = values(res.get(&head).ok_or_else(|| bad("no logits"))?)?;
        let mut out = Vec::with_capacity(items.len());
        for (k, ((_, labels), rows)) in items.iter().zip(&per).enumerate() {
            let row = l.get(k * union.len()..(k + 1) * union.len()).ok_or_else(|| bad("short logits"))?;
            out.push(label_mass(rows, |r| union.iter().position(|u| *u == r).and_then(|i| row.get(i)).copied().unwrap_or(f32::MIN), labels.len()));
        }
        Ok((out, t.elapsed().as_secs_f64() * 1000.0))
    }
}

/// Install the reader for the request's drone when its model form is on (`NANO_REQ_MODEL=on`).
/// Returns whether it was installed now.
/// # Errors
/// The reader cannot be loaded.
pub fn install(home: &Path) -> Result<bool, Fault> {
    if !nmpod::intake::model_form_on() || nmpod::intake::model_installed() {
        return Ok(false);
    }
    let model = crate::neural::model(home)?;
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    Ok(nmpod::intake::install_model(Arc::new(ReaderLabels::new(model, dev))))
}

/// Install over a model already loaded (the bench and the served path hold one).
pub fn install_with(model: &Arc<Model>) -> bool {
    if !nmpod::intake::model_form_on() || nmpod::intake::model_installed() {
        return false;
    }
    let dev = if model.pop.gpu.is_some() { Device::Gpu } else { Device::Cpu };
    nmpod::intake::install_model(Arc::new(ReaderLabels::new(Arc::clone(model), dev)))
}

/// The embedder's side of the request's kind read (`nmpod::intake::TextEmbedder`): the released
/// embedder MicroModel (`crate::neural::embedder_model`); on the card the texts packed in one run
/// (`card_embed_batch`), on the processor one at a time. Ungated, as the dense lane's questions.
pub struct EmbedderText {
    model: Arc<Model>,
}
impl std::fmt::Debug for EmbedderText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedderText").finish_non_exhaustive()
    }
}
impl EmbedderText {
    #[must_use]
    pub fn new(model: Arc<Model>) -> Self {
        Self { model }
    }
}
/// Most tokens of one text of the kind read (questions are short).
const KIND_TEXT_CAP: usize = 256;
impl nmpod::intake::TextEmbedder for EmbedderText {
    fn embed(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, f64), Fault> {
        let t = Instant::now();
        let v = if self.model.pop.gpu.is_some() {
            nmengn::roles::card_embed_batch(&self.model, texts, KIND_TEXT_CAP, 4096)?
        } else {
            texts.iter().map(|x| nmengn::roles::card_embed_on_capped(&self.model, Device::Cpu, x, Some(KIND_TEXT_CAP))).collect::<Result<_, _>>()?
        };
        Ok((v, t.elapsed().as_secs_f64() * 1000.0))
    }
}

/// Install the embedder for the request's kind read when it is on (`NANO_REQ_KIND=embed`).
pub fn install_embedder_with(model: &Arc<Model>) -> bool {
    if !nmpod::intake::embed_kind_on() || nmpod::intake::embedder_installed() {
        return false;
    }
    nmpod::intake::install_embedder(Arc::new(EmbedderText::new(Arc::clone(model))))
}
