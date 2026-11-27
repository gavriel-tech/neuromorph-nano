//! The key of what made an encoded read kept on disk: the dense store's header names the values a
//! model runs on and its precision, so a different cartridge or precision reuses nothing.
//!
//! Processor only, no graphics card: the embedder bound at 16 and at 8 bits from its open weights.
//! Run: `cargo test --release -p nmapi --test parameters_key`.
use nmapi::neural::{dense_header, parameters_key, values_key};
use nmengn::model::{Model, Population};

#[test]
fn values_key_names_the_values() {
    let open = values_key("aa", None);
    assert_eq!(open, "open weights aa");
    // An exact cartridge proven bit-identical runs on the open weights' values.
    assert_eq!(values_key("aa", Some(("bb", true))), open);
    // A cartridge with values of its own is keyed by its parameters; two such differ.
    assert_eq!(values_key("aa", Some(("bb", false))), "cartridge bb");
    assert_ne!(values_key("aa", Some(("bb", false))), values_key("aa", Some(("cc", false))));
}

#[test]
fn dense_header_changes_with_precision() {
    let home = nmtype::nano_root();
    let pop = Population::open(&home, "qwen3-embedding-0.6b", true, false).expect("embedder");
    let bind = |form: &str| {
        let mut a = nmengn::Assembly::load(&home.join("designs/assemblies"), "qwen3-embedding").expect("assembly");
        for n in a.nodes.iter_mut().filter(|n| n.op == "linear") {
            form.clone_into(&mut n.precision);
        }
        Model::bind(&pop, a).expect("bind")
    };
    let (m16, m8) = (bind("f16"), bind("q8"));
    assert_eq!(parameters_key(&m16), format!("open weights {}", pop.card.sha_main));
    let (h16, h8) = (dense_header(&m16), dense_header(&m8));
    assert_eq!(h16.get("parameters"), h8.get("parameters"));
    assert_ne!(h16, h8, "a different precision must give a different header");
    assert_ne!(h16.get("precision"), h8.get("precision"));
}
