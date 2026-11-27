//! Tests of the cartridge library: a cartridge is one file holding its parameters and the label
//! needed to load them; stacked filters give a defined, repeatable result in a defined order and
//! leave the base bit for bit; the file reads back exactly, and a changed file or a form this build
//! does not seat is refused by name; preference cartridges are kept and switched apart; a library
//! written in the earlier form is still listed. Folders under `scratch/library-test/`.
#![allow(clippy::unwrap_used)]
use nmjson::Value;
use nmlearn::library::{CartridgeSpec, Filter, LibFault, Library, Point, Stack, apply_numbers, apply_settings, use_rank};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A test's folder, removed when the test ends (2026-11-26: tests leave no junk).
struct TestDir(PathBuf);
impl std::ops::Deref for TestDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn dir(name: &str) -> TestDir {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scratch/library-test");
    // 2026-11-26 (Luis Jake Gabriel III: "We should always delete test and junk stuff"): folders an
    // earlier run of these tests left (another process id) are removed first.
    let me = format!("-{}", std::process::id());
    if let Ok(rd) = std::fs::read_dir(&base) {
        for e in rd.flatten() {
            if !e.file_name().to_string_lossy().ends_with(&me) {
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
    }
    let d = base.join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    TestDir(d)
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn spec(id: &str, family: &str) -> CartridgeSpec {
    CartridgeSpec {
        id: id.into(),
        version: 1,
        family: family.into(),
        domain: String::new(),
        scope: "general".into(),
        model: "qwen3-reranker-0.6b".into(),
        weights: "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b".into(),
        layers: Some((27, 27)),
        name: format!("test {id}"),
        description: "A test filter.".into(),
        made_from: "numbers written for the test".into(),
        effect: String::new(),
    }
}
fn affine(gain: &[f32], bias: &[f32]) -> Filter {
    Filter::point("final_norm", gain.to_vec(), bias.to_vec())
}
fn stack(id: &str, use_for: &str, members: &[&str]) -> Stack {
    Stack { id: id.into(), version: 1, use_for: use_for.into(), piece: None, cartridges: members.iter().map(|b| (*b).to_owned()).collect(), description: String::new(), hash: String::new(), trial: false }
}

/// The base the filters read: never written.
const BASE: [f32; 4] = [0.5, -1.25, 3.0, 0.0];

#[test]
fn stacked_cartridges_give_a_defined_repeatable_result_in_order() {
    let (g1, b1, g2, b2) = ([2.0, 1.0, 0.5, 1.0], [0.0, 0.25, 0.0, -1.0], [1.0, -1.0, 1.0, 3.0], [1.0, 0.0, -0.5, 0.0]);
    let (one, two) = (affine(&g1, &b1), affine(&g2, &b2));
    let got = apply_numbers("final_norm", &BASE, &[&one, &two]).unwrap();
    let want: Vec<f32> = BASE.iter().enumerate().map(|(i, x)| x.mul_add(g1[i], b1[i]).mul_add(g2[i], b2[i])).collect();
    assert_eq!(bits(&got), bits(&want));
    assert_eq!(got, vec![2.0, 1.0, 1.0, -3.0]);
    assert_eq!(bits(&apply_numbers("final_norm", &BASE, &[&one, &two]).unwrap()), bits(&got));
    let other = apply_numbers("final_norm", &BASE, &[&two, &one]).unwrap();
    assert_eq!(other, vec![3.0, 1.5, 1.25, -1.0]);
    // Nothing plugged, or a filter at another socket: the base bit for bit; a wrong width refused.
    assert_eq!(bits(&apply_numbers("final_norm", &BASE, &[]).unwrap()), bits(&BASE));
    let elsewhere = Filter::point("L13.swigate", vec![9.0; 4], vec![9.0; 4]);
    assert_eq!(bits(&apply_numbers("final_norm", &BASE, &[&elsewhere]).unwrap()), bits(&BASE));
    assert!(matches!(apply_numbers("final_norm", &BASE, &[&affine(&[1.0], &[0.0])]), Err(LibFault::Shape(_))));
    let base: BTreeMap<String, String> = [("judge_window", "384"), ("fast_k", "4")].iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    let a = Filter::Settings([("judge_window".to_owned(), "256".to_owned())].into());
    let b = Filter::Settings([("judge_window".to_owned(), "512".to_owned())].into());
    assert_eq!(apply_settings(&base, &[&a, &b]).get("judge_window").map(String::as_str), Some("512"));
    assert_eq!(apply_settings(&base, &[&b, &a]).get("judge_window").map(String::as_str), Some("256"));
    assert_eq!(apply_settings(&base, &[&a, &b]).get("fast_k").map(String::as_str), Some("4"));
}

#[test]
fn a_cartridge_is_one_file_that_reads_back_exactly() {
    let home = dir("one-file");
    let mut lib = Library::open(&home).unwrap();
    let q = Filter::points_q8(vec![
        Point { target: "final_norm".into(), gain: vec![1.0, 1.25, 0.8, 1.003], bias: vec![0.0, 0.01, -0.02, 0.0] },
        Point { target: "L13.swigate".into(), gain: vec![0.9, 1.1], bias: vec![0.0, 0.0] },
    ]);
    let notes = nmjson::parse(br#"{"fitted_from": "test material"}"#).unwrap();
    lib.add_cartridge(&spec("rust", "fundamentals"), &q, notes.clone(), Vec::new(), true).unwrap();
    lib.add_cartridge(&spec("plain", ""), &affine(&[2.0; 4], &[0.0; 4]), Value::Null, Vec::new(), false).unwrap();
    lib.add_stack(stack("coding", "job:coding", &["rust", "plain"])).unwrap();
    lib.save().unwrap();
    // Kept by kind, one folder per cartridge, its version in its name.
    let files: Vec<String> = std::fs::read_dir(home.join("library/cartridges/learned")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(files.len(), 2, "{files:?}");
    assert!(home.join("library/cartridges/learned/rust/rust-v1.cartridge").is_file());
    // The shelf lists files and hashes; the label inside the file says the rest.
    let shelf = std::fs::read_to_string(home.join("library").join("cartridges").join("catalogue.json")).unwrap();
    assert!(!shelf.contains("final_norm") && !shelf.contains("fitted_from"), "the shelf holds the label: {shelf}");
    let lib = Library::open(&home).unwrap();
    let c = lib.cartridge("rust").unwrap();
    assert_eq!((c.family.as_str(), c.target.as_str(), c.notes.clone()), ("fundamentals", "final_norm,L13.swigate", notes));
    assert_eq!(c.weights, "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b");
    let back = lib.filter(c).unwrap();
    for (x, y) in back.points().iter().zip(&q.points()) {
        assert_eq!(bits(&x.gain), bits(&y.gain));
        assert_eq!(bits(&x.bias), bits(&y.bias));
    }
    assert_eq!(lib.load("coding").unwrap().filters.iter().map(|f| f.0.clone()).collect::<Vec<_>>(), vec!["rust@1".to_owned(), "plain@1".to_owned()]);
    // A changed byte is refused at load, by name, and the stack loads without it.
    let p = home.join("library/cartridges/learned/plain/plain-v1.cartridge");
    let mut b = std::fs::read(&p).unwrap();
    if let Some(x) = b.last_mut() {
        *x ^= 1;
    }
    std::fs::write(&p, &b).unwrap();
    let l = lib.load("coding").unwrap();
    assert_eq!(l.filters.len(), 1);
    assert!(l.refused.first().is_some_and(|(k, why)| k == "plain@1" && why.contains("(file-hash)")), "{:?}", l.refused);
    // The parameters are checked against the label too, and a part this build does not seat is
    // refused by name.
    assert!(matches!(nmlearn::library::read_cartridge(&b), Err(LibFault::ContentHash { .. })));
    // A cartridge without a label is refused.
    let mut unlabelled = spec("bare", "");
    unlabelled.name = String::new();
    let mut lib2 = Library::open(&home).unwrap();
    assert!(matches!(lib2.add_cartridge(&unlabelled, &affine(&[1.0; 4], &[0.0; 4]), Value::Null, Vec::new(), false), Err(LibFault::Schema(_))));
    assert!(lib.cartridge("rust").unwrap().label_text().contains("made from: numbers written for the test"));
    let label = nmjson::parse(br#"{"schema":"nm-cartridge/1","id":"t","version":1,"model":"qwen3-reranker-0.6b","parts":[{"socket":"L3.q","form":"tensor","shape":[2,2],"encoding":"f16","at":0,"bytes":8}],"parameters_sha256":"af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"}"#).unwrap();
    let file = nmlearn::library::cartridge_file(&label, &[0_u8; 8]);
    let (l, body) = nmlearn::library::read_cartridge(&file).unwrap();
    assert_eq!(body.len(), 8);
    assert_eq!(l.get("parts").unwrap().array().unwrap().len(), 1);
}

#[test]
fn preference_cartridges_are_kept_and_switched_apart() {
    let home = dir("preference");
    let mut lib = Library::open(&home).unwrap();
    lib.add_cartridge(&spec("my-ways", "preference"), &affine(&[1.0; 4], &[1.0; 4]), Value::Null, Vec::new(), false).unwrap();
    lib.add_cartridge(&spec("rust", "fundamentals"), &affine(&[2.0; 4], &[0.0; 4]), Value::Null, Vec::new(), false).unwrap();
    assert!(home.join("library/cartridges/preference/my-ways/my-ways-v1.cartridge").is_file());
    assert!(matches!(lib.add_stack(stack("mixed", "job:coding", &["rust", "my-ways"])), Err(LibFault::Conflict(_))));
    assert!(matches!(lib.add_stack(stack("wrong-use", "job:coding", &["my-ways"])), Err(LibFault::Conflict(_))));
    lib.add_stack(stack("mine", "preference", &["my-ways"])).unwrap();
    lib.add_stack(stack("coding", "job:coding", &["rust"])).unwrap();
    lib.save().unwrap();
    let lib = Library::open(&home).unwrap();
    assert_eq!(lib.load("mine").unwrap().filters.len(), 1);
    assert!(use_rank("preference") > use_rank("job:coding"), "preference uses apply after the others");
}

#[test]
fn a_library_written_in_the_earlier_form_still_reads() {
    let home = dir("earlier");
    let lib_dir = home.join("library");
    let folder = lib_dir.join("books").join("op-v1");
    std::fs::create_dir_all(&folder).unwrap();
    let numbers: Vec<u8> = [2.0_f32; 4].iter().chain(&[0.0_f32; 4]).flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(folder.join("numbers.f32"), &numbers).unwrap();
    let sha = nmhash::sha256_bytes(&numbers);
    let hash = nmlearn::library::content_hash(&[nmlearn::library::FileRec { name: "numbers.f32".into(), sha256: sha.clone(), bytes: 32 }]);
    let cat = format!(
        r#"{{"schema":"nm-library/1","books":[{{"id":"op","version":1,"domain":"hardware:gpu","model":"m","layers":null,"kind":"affine","target":"final_norm","description":"","files":[{{"name":"numbers.f32","sha256":"{sha}","bytes":32}}],"hash":"{hash}","evidence":[],"trial":false}}],"stacks":[],"active":{{}}}}"#
    );
    std::fs::write(lib_dir.join("catalogue.json"), cat).unwrap();
    let mut lib = Library::open(&home).unwrap();
    let c = lib.cartridge("op").unwrap().clone();
    // Listed and readable for writing again, but not loaded: it has no label of its own.
    assert_eq!(lib.earlier_filter(&c).unwrap().points().first().unwrap().gain, vec![2.0; 4]);
    lib.add_stack(stack("s", "job:x", &["op"])).unwrap();
    assert!(lib.load("s").unwrap().refused.first().is_some_and(|r| r.1.contains("label")));
}

#[test]
fn a_cartridge_carries_addressed_parameter_sets_and_says_where_each_lies_in_its_file() {
    use nmlearn::library::{Tensor, tensor_parts_of};
    let home = dir("tensors");
    let mut lib = Library::open(&home).unwrap();
    let a = Tensor { name: "model.layers.3.mlp.down_proj.weight".into(), layer: Some(3), block: "feed-forward".into(), dtype: "BF16".into(), shape: vec![2, 3], bytes: (0..12).collect() };
    let b = Tensor { name: "model.norm.weight".into(), layer: None, block: "final norm".into(), dtype: "F32".into(), shape: vec![2], bytes: vec![9; 8] };
    let q = Tensor { name: "model.layers.3.mlp.up_proj.weight".into(), layer: Some(3), block: "feed-forward".into(), dtype: "F16".into(), shape: vec![3, 2], bytes: vec![1; 12] };
    let f = Filter::Tensors(vec![a.clone(), b.clone(), q.clone()]);
    lib.add_cartridge(&spec("judge-base", ""), &f, Value::Null, Vec::new(), false).unwrap();
    let c = lib.cartridge("judge-base").unwrap().clone();
    assert_eq!(c.kind.as_str(), "tensors");
    // Read back whole, and as places in the file that hold exactly the sets' bytes, with their
    // addresses and content hashes.
    assert_eq!(lib.filter(&c).unwrap(), f);
    let parts = lib.tensor_parts(&c).unwrap();
    let file = std::fs::read(&parts[0].file).unwrap();
    for (p, t) in parts.iter().zip([&a, &b, &q]) {
        assert_eq!((p.name.as_str(), p.layer, p.block.as_str(), p.dtype.as_str(), &p.shape), (t.name.as_str(), t.layer, t.block.as_str(), t.dtype.as_str(), &t.shape));
        assert_eq!(&file[p.offset as usize..(p.offset + p.len) as usize], t.bytes.as_slice());
        assert_eq!(p.sha256, nmhash::sha256_bytes(&t.bytes));
    }
    // Bytes that do not fit the type and shape, a part carrying structure, and a file without a
    // label, are refused.
    let bad = Filter::Tensors(vec![Tensor { bytes: vec![0; 5], ..a.clone() }]);
    lib.add_cartridge(&spec("bad-shape", ""), &bad, Value::Null, Vec::new(), false).unwrap();
    assert!(lib.filter(lib.cartridge("bad-shape").unwrap()).is_err());
    let body = br#"{"name":"cut"}"#;
    let with_structure = nmjson::parse(format!(r#"{{"schema":"nm-cartridge/1","id":"s","version":1,"model":"m","name":"n","does":"d","made_from":"m","made":"2026-11-16","parts":[{{"form":"assembly","encoding":"json","at":0,"bytes":{}}}],"parameters_sha256":"{}"}}"#, body.len(), nmhash::sha256_bytes(body)).as_bytes()).unwrap();
    let p = home.join("structure.cartridge");
    std::fs::write(&p, nmlearn::library::cartridge_file(&with_structure, body)).unwrap();
    assert!(tensor_parts_of(&p).is_err_and(|e| e.to_string().contains("structure")));
    let label = nmjson::parse(br#"{"schema":"nm-cartridge/1","id":"t","version":1,"model":"m","parts":[],"parameters_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}"#).unwrap();
    let p = home.join("unlabelled.cartridge");
    std::fs::write(&p, nmlearn::library::cartridge_file(&label, &[])).unwrap();
    assert!(tensor_parts_of(&p).is_err_and(|e| e.to_string().contains("label")));
}

#[test]
fn adapters_have_their_own_shelf_and_section_and_are_refused_elsewhere() {
    let home = dir("adapters");
    let mut lib = Library::open(&home).unwrap();
    let key = lib.add_cartridge(&spec("judge-adapter", "adapter"), &affine(&[1.0; 4], &[0.0; 4]), Value::Null, Vec::new(), true).unwrap();
    lib.add_cartridge(&spec("plain", ""), &affine(&[1.0; 4], &[0.0; 4]), Value::Null, Vec::new(), true).unwrap();
    // A base cartridge (parameter sets) is never on the adapter shelf.
    let sets = Filter::Tensors(vec![nmlearn::library::Tensor { name: "model.norm.weight".into(), layer: None, block: "final norm".into(), dtype: "BF16".into(), shape: vec![2], bytes: vec![0; 4] }]);
    assert!(matches!(lib.add_cartridge(&spec("base-as-adapter", "adapter"), &sets, Value::Null, Vec::new(), true), Err(LibFault::Conflict(_))));
    lib.save().unwrap();
    // The file sits in `cartridges/adapter/`, the catalogue lists it in its own section, never in
    // `cartridges`.
    assert!(home.join("library/cartridges/adapter/judge-adapter/judge-adapter-v1.cartridge").is_file());
    assert!(!home.join("library/cartridges/judge-adapter-v1.cartridge").exists());
    let cat = nmjson::parse(&std::fs::read(home.join("library/cartridges/catalogue.json")).unwrap()).unwrap();
    let ids = |k: &str| cat.get(k).and_then(Value::array).unwrap().iter().map(|e| e.get("id").and_then(Value::string).unwrap().to_owned()).collect::<Vec<_>>();
    assert_eq!(ids("adapters"), vec!["judge-adapter".to_owned()]);
    assert_eq!(ids("cartridges"), vec!["plain".to_owned()]);
    // Read back, it loads; listed in the base section instead, it is refused by name.
    let back = Library::open(&home).unwrap();
    assert!(back.filter(back.cartridge(&key).unwrap()).is_ok());
    let mut text = String::from_utf8(std::fs::read(home.join("library/cartridges/catalogue.json")).unwrap()).unwrap();
    text = text.replacen("\"adapters\"", "\"adapters_moved\"", 1).replacen("\"cartridges\": [", "\"cartridges\": [{\"bytes\": 0, \"file\": \"adapter/judge-adapter/judge-adapter-v1.cartridge\", \"id\": \"judge-adapter\", \"sha256\": \"\", \"trial\": true, \"version\": 1},", 1);
    std::fs::write(home.join("library/cartridges/catalogue.json"), text).unwrap();
    let moved = Library::open(&home).unwrap();
    let c = moved.cartridge(&key).unwrap();
    assert!(moved.filter(c).is_err_and(|e| e.to_string().contains("cartridges section")));
    // And a base-family cartridge listed in the adapters section is refused the same way.
    let mut text = String::from_utf8(std::fs::read(home.join("library/cartridges/catalogue.json")).unwrap()).unwrap();
    text = text.replacen("\"adapters_moved\"", "\"adapters\"", 1);
    let plain = "{\"bytes\": 0, \"file\": \"plain-v1.cartridge\", \"id\": \"plain\", \"sha256\": \"\", \"trial\": true, \"version\": 1}";
    text = text.replacen("\"adapters\": [", &format!("\"adapters\": [{plain},"), 1);
    std::fs::write(home.join("library/cartridges/catalogue.json"), text).unwrap();
    let wrong = Library::open(&home).unwrap();
    assert!(wrong.cartridges.iter().filter(|c| c.id == "plain").any(|c| wrong.filter(c).is_err_and(|e| e.to_string().contains("adapters section"))));
}

#[test]
fn a_base_cartridge_is_held_as_what_differs_from_the_open_weights() {
    use nmlearn::library::{Keep, OpenSet, Tensor, tensor_parts_of};
    let home = dir("open-form");
    let mut lib = Library::open(&home).unwrap();
    let whole = OpenSet { name: "model.norm.weight".into(), layer: None, block: "final norm".into(), dtype: "BF16".into(), shape: vec![1024], sha256: "aa".into(), keep: None };
    let narrowed = OpenSet {
        name: "model.layers.3.mlp.down_proj.weight".into(),
        layer: Some(3),
        block: "feed-forward".into(),
        dtype: "BF16".into(),
        shape: vec![1024, 2048],
        sha256: "bb".into(),
        keep: Some(Keep { cols: true, ranges: vec![(0, 1024), (2048, 3072)] }),
    };
    let edited = Tensor { name: "model.layers.3.mlp.up_proj.weight".into(), layer: Some(3), block: "feed-forward".into(), dtype: "BF16".into(), shape: vec![2, 2], bytes: vec![7; 8] };
    let f = Filter::Open { sets: vec![whole.clone(), narrowed.clone()], edited: vec![edited.clone()] };
    lib.add_cartridge(&spec("judge-delta", ""), &f, Value::Null, Vec::new(), false).unwrap();
    let c = lib.cartridge("judge-delta").unwrap().clone();
    assert_eq!(c.kind.as_str(), "tensors");
    assert_eq!(lib.filter(&c).unwrap(), f);
    // The file holds a label and the edited values only: the sets taken from the open weights take
    // no bytes, and say what they keep.
    let file = home.join("library/cartridges/base/reranker/judge-delta-v1.cartridge");
    assert!(std::fs::metadata(&file).unwrap().len() < 4096);
    let parts = tensor_parts_of(&file).unwrap();
    assert_eq!(parts.len(), 3);
    assert!(parts[0].open && parts[0].keep.is_none() && parts[0].len == 2048);
    assert!(parts[1].open && parts[1].keep == narrowed.keep && parts[1].len == 1024 * 2048 * 2);
    assert!(!parts[2].open && parts[2].len == 8 && parts[2].sha256 == nmhash::sha256_bytes(&edited.bytes));
}
