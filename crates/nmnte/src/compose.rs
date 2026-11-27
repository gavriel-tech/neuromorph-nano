//! Composition Engine (guide 11.1 item 2, 11.2; PLAN 4d items 4, 5, 12): builds an
//! execution assembly from data. The objective kind picks a row of
//! `designs/compositions.json` (unlisted kinds use the `context` row); the core assembly is
//! taken as is, and the tentacle template is instantiated once per tentacle the row asks for,
//! `{n}` replaced by the index. No topology is written in code.
use crate::graph::Graph;
use nmjson::Value;
use nmtype::{Budget, Fault, FaultKind};

fn bad(e: impl std::fmt::Display) -> Fault {
    Fault::new("nmnte.compose", FaultKind::Config, e.to_string())
}

/// The composition row chosen for an objective kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub objective_kind: String,
    pub tentacles: usize,
    pub views: Vec<String>,
    pub slots: Vec<String>,
    pub max_depth: u8,
    pub budget: Budget,
}

/// Pick the row for a kind (falls back to `context`).
/// # Errors
/// No row for the kind and no `context` row, or a malformed row.
pub fn row(compositions: &Value, kind: &str) -> Result<Row, Fault> {
    let rows = compositions.get("rows").and_then(Value::array).map_err(bad)?;
    let pick = |k: &str| rows.iter().find(|r| r.get("objective_kind").and_then(Value::string).ok() == Some(k));
    let r = pick(kind).or_else(|| pick("context")).ok_or_else(|| bad(format!("no composition row for {kind} and no context row")))?;
    let strings = |k: &str| {
        r.get(k).and_then(Value::array).map(|a| a.iter().filter_map(|x| x.string().ok().map(str::to_owned)).collect()).unwrap_or_default()
    };
    let int = |v: &Value, k: &str| v.get(k).and_then(Value::integer).unwrap_or(0);
    let b = r.get("budget").map_err(bad)?;
    let num = |k: &str| u32::try_from(int(b, k)).unwrap_or(u32::MAX);
    Ok(Row {
        objective_kind: r.get("objective_kind").and_then(Value::string).map_err(bad)?.to_owned(),
        tentacles: usize::try_from(int(r, "tentacles")).map_err(bad)?,
        views: strings("views"),
        slots: strings("slots"),
        max_depth: u8::try_from(int(r, "max_depth")).unwrap_or(1),
        budget: Budget {
            tokens: num("tokens"),
            ms: num("ms"),
            depth: u8::try_from(int(b, "depth")).unwrap_or(1),
            units: u8::try_from(int(b, "units")).unwrap_or(16),
        },
    })
}

/// One tentacle from the template, `{n}` replaced by `n`.
/// # Errors
/// A template that does not encode, or does not parse back as a graph.
pub fn tentacle(template: &Value, n: usize) -> Result<Graph, Fault> {
    let text = template.encode().map_err(bad)?.replace("{n}", &n.to_string());
    Graph::from_value(&nmjson::parse(text.as_bytes()).map_err(bad)?)
}

/// Compose the assembly for an objective kind: the core graph plus `row.tentacles`
/// instances of the tentacle template.
/// # Errors
/// As `row` and `tentacle`.
pub fn compose(kind: &str, compositions: &Value, core: &Graph, template: &Value) -> Result<(Graph, Row), Fault> {
    let r = row(compositions, kind)?;
    let mut g = core.clone();
    for n in 0..r.tentacles {
        let t = tentacle(template, n)?;
        g.nodes.extend(t.nodes);
        g.edges.extend(t.edges);
    }
    Ok((g, r))
}

/// Self-check of the `compose` sub-component: a two-row table picks by kind and falls back
/// to `context`; the template is instantiated once per tentacle with unique ids.
pub(crate) fn check() -> Result<(), String> {
    let table = nmjson::parse(br#"{"rows":[{"objective_kind":"context","tentacles":3,"views":["global"],"slots":["router"],"max_depth":1,"budget":{"tokens":100,"ms":1000,"depth":1,"units":4}},{"objective_kind":"build","tentacles":2,"views":["operational"],"slots":["writer"],"max_depth":1,"budget":{"tokens":50,"ms":1000,"depth":1,"units":4}}]}"#).map_err(|e| e.to_string())?;
    let template = nmjson::parse(
        br#"{"nodes":[{"id":"x#t{n}","kind":"tentacle","parent":null,"ports_in":[],"ports_out":[],"config":{}}],"edges":[]}"#,
    )
    .map_err(|e| e.to_string())?;
    let (g, r) = compose("build", &table, &Graph::default(), &template).map_err(|e| e.to_string())?;
    if r.tentacles != 2 || g.nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>() != ["x#t0", "x#t1"] {
        return Err("nmnte.compose: build row".into());
    }
    if row(&table, "anatomy").map_err(|e| e.to_string())?.objective_kind != "context" {
        return Err("nmnte.compose: fallback to the context row".into());
    }
    Ok(())
}
