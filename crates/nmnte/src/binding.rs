//! Parameter Binding Manager (guide 11.1 item 4, 9.4; PLAN 4c): which parameter clusters
//! each node is bound to. The table is what anatomy zoom reads (cluster, component,
//! structure). Clusters arrive with NPCS at S6; the table and its queries exist now, so
//! RNAM changes keep bindings consistent from the start.
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BindingTable {
    by_node: BTreeMap<String, Vec<String>>,
}
impl BindingTable {
    /// Bind a node to clusters (replaces that node's previous binding).
    pub fn bind(&mut self, node: &str, clusters: Vec<String>) {
        self.by_node.insert(node.to_owned(), clusters);
    }
    /// Drop a node's binding (node removed); returns what it held.
    pub fn unbind(&mut self, node: &str) -> Vec<String> {
        self.by_node.remove(node).unwrap_or_default()
    }
    #[must_use]
    pub fn clusters(&self, node: &str) -> &[String] {
        self.by_node.get(node).map_or(&[], Vec::as_slice)
    }
    /// Nodes bound to a cluster (one cluster may serve several nodes, shared, not copied).
    #[must_use]
    pub fn nodes_of(&self, cluster: &str) -> Vec<&str> {
        self.by_node.iter().filter(|(_, cs)| cs.iter().any(|c| c == cluster)).map(|(n, _)| n.as_str()).collect()
    }
}

/// Self-check of the `binding` sub-component: one cluster shared by two nodes; unbinding one
/// leaves the other.
pub(crate) fn check() -> Result<(), String> {
    let mut b = BindingTable::default();
    b.bind("q#0", vec!["layer0.head0".into()]);
    b.bind("k#0", vec!["layer0.head0".into()]);
    let _ = b.unbind("q#0");
    if b.nodes_of("layer0.head0") != ["k#0"] || !b.clusters("q#0").is_empty() {
        return Err("nmnte.binding: shared cluster bookkeeping".into());
    }
    Ok(())
}
