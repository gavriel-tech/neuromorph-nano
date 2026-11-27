//! Inspect (guide 9.3 responsibility 3, metadata; PLAN 4c `nm_weights`): a population read
//! only: tensors, shapes, precision, clusters, inseparable groups, bytes, and where each
//! cluster lives now.
use crate::acquire::{self, WeightCard};
use crate::decompose::{ParameterCluster, TensorDesc};
use crate::provision::{Provisioner, Storage};
use nmjson::Value;
use std::collections::{BTreeMap, BTreeSet};

/// The population as JSON. `limit` caps the cluster list (the counts are always complete).
#[must_use]
pub fn inspect(
    card: &WeightCard,
    tensors: &[TensorDesc],
    clusters: &[ParameterCluster],
    prov: Option<&Provisioner>,
    limit: usize,
) -> Value {
    let groups: BTreeSet<&str> = clusters.iter().map(|c| c.group.as_str()).collect();
    let heads = groups.iter().filter(|g| g.contains(".head")).count();
    let kv = groups.iter().filter(|g| g.contains(".kv")).count();
    let ffn = groups.iter().filter(|g| g.ends_with(".ffn")).count();
    let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
    for t in tensors {
        *dtypes.entry(t.dtype.clone()).or_default() += 1;
    }
    let mut m = BTreeMap::new();
    m.insert("card".into(), acquire::to_value(card));
    m.insert("tensors".into(), Value::Number(tensors.len().to_string()));
    m.insert("tensor_bytes".into(), Value::Number(tensors.iter().map(|t| t.end - t.start).sum::<u64>().to_string()));
    m.insert("dtypes".into(), Value::Object(dtypes.into_iter().map(|(k, v)| (k, Value::Number(v.to_string()))).collect()));
    m.insert("clusters".into(), Value::Number(clusters.len().to_string()));
    m.insert(
        "groups".into(),
        Value::Object(
            [("head", heads), ("kv", kv), ("ffn", ffn), ("all", groups.len())]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), Value::Number(v.to_string())))
                .collect(),
        ),
    );
    let list: Vec<Value> = clusters
        .iter()
        .take(limit)
        .map(|c| {
            let mut v = c.to_value();
            if let (Value::Object(o), Some(p)) = (&mut v, prov) {
                let s = match p.storage_of(c) {
                    Storage::Mapped { .. } => "file".to_owned(),
                    Storage::Resident => "ram".to_owned(),
                    Storage::Device { bytes } if p.host_held(c) => format!("ram+gpu({bytes})"),
                    Storage::Device { bytes } => format!("gpu({bytes})"),
                };
                o.insert("storage".into(), Value::String(s));
            }
            v
        })
        .collect();
    m.insert("cluster_list".into(), Value::Array(list));
    if let Some(p) = prov {
        m.insert("residency".into(), p.to_value());
        m.insert(
            "lifecycle".into(),
            Value::Object(p.lifecycle.counts().into_iter().map(|(k, v)| (k, Value::Number(v.to_string()))).collect()),
        );
    }
    Value::Object(m)
}

pub(crate) fn check() -> Result<(), String> {
    let card = WeightCard {
        id: "toy".into(),
        family: "qwen3".into(),
        tokenizer: "qwen3-10".into(),
        dir: std::path::PathBuf::new(),
        main: String::new(),
        sha_main: String::new(),
        bytes: 0,
        revision: String::new(),
        config: Value::Null,
    };
    let t = vec![TensorDesc {
        name: "model.norm.weight".into(),
        dtype: "BF16".into(),
        shape: vec![8],
        file: "toy.safetensors".into(),
        start: 0,
        end: 16,
        sha256: String::new(),
    }];
    let cs = crate::decompose::decompose("toy", 1, 1, 1, &t).map_err(|e| e.to_string())?;
    let v = inspect(&card, &t, &cs, None, 10);
    if v.get("clusters").and_then(Value::integer).ok() != Some(1) {
        return Err("nmnpcs.inspect: cluster count".into());
    }
    Ok(())
}
