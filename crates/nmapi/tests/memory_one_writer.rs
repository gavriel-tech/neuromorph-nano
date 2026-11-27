//! One writer per memory file, on a copy of a real memory folder left in the older mixed layout
//! (`scratch/memtime/home/memory/projects/repo-0d19e713`, studied repository `scratch/memtime/repo`):
//! the copy goes to `scratch/memfix/test-one-writer/`, the organism opens it with memory on, and
//! then: the tier overlap is 0, every id the old files listed is kept (journal, hierarchy or
//! temporary notes), the hierarchy's own items keep their counts, and the overlap stays 0 through 50
//! mixed calls (context pack, memory recall, feedback, temporary notes). Without the folder the
//! test says so and passes; the section check `nmmem.onewriter` proves the same on a folder it
//! builds itself.
use nmjson::Value;
use std::path::{Path, PathBuf};

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("folder");
    for e in std::fs::read_dir(from).expect("read").flatten() {
        let p = e.path();
        let name = e.file_name();
        if p.is_dir() {
            copy_dir(&p, &to.join(&name));
        } else if name != "LOCK" {
            std::fs::copy(&p, to.join(&name)).expect("copy");
        }
    }
}

fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::Object(pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect())
}
fn s(t: &str) -> Value {
    Value::String(t.to_owned())
}

#[test]
fn mixed_folder_gets_one_writer_per_file() {
    let home = nmtype::nano_root();
    let seed = home.join("scratch/memtime/home/memory/projects/repo-0d19e713");
    let repo = home.join("scratch/memtime/repo");
    if !seed.join("journal").is_dir() || !repo.is_dir() {
        eprintln!("skipped: no mixed memory folder at {}", seed.display());
        return;
    }
    let mem: PathBuf = home.join("scratch/memfix/test-one-writer");
    if mem.exists() {
        std::fs::remove_dir_all(&mem).expect("clear the earlier copy");
    }
    copy_dir(&seed, &mem);
    let before_overlap = nmmem::onewriter::tiers_overlap(&mem);
    let before_ids = nmmem::onewriter::listed_ids(&mem);
    // The hierarchy's own items in the old tier files, with their counts.
    let mut own: Vec<(String, u64)> = Vec::new();
    for (_, f) in nmmem::hierarchy::tier_files(&mem) {
        let v = nmjson::parse(&std::fs::read(&f).expect("tier file")).expect("json");
        for x in v.get("items").and_then(Value::array).unwrap_or_default() {
            let id = x.get("id").and_then(Value::string).unwrap_or("").to_owned();
            if nmmem::hierarchy::is_own_item(x) && !id.contains(':') {
                own.push((id, x.get("uses").and_then(Value::integer).unwrap_or(0)));
            }
        }
    }
    assert!(before_overlap > 0 && nmmem::onewriter::is_mixed(&mem), "the copy is not mixed (overlap {before_overlap})");

    let org = nmapi::forage::Organism::open_with(&home, &repo, &[], true, &mem).expect("open");
    assert_eq!(nmmem::onewriter::tiers_overlap(&mem), 0, "overlap after the migration");
    assert!(!nmmem::onewriter::is_mixed(&mem), "still mixed after the migration");
    {
        let m = org.memory.lock().expect("memory");
        let book = &m.mem2.as_ref().expect("journal open").book;
        let lost: Vec<&String> = before_ids
            .iter()
            .filter(|id| !book.records.contains_key(*id) && m.hierarchy.get(id).is_none() && !m.temp.list().iter().any(|n| &n.id == *id))
            .collect();
        assert!(lost.is_empty(), "ids lost: {lost:?}");
        for (id, uses) in &own {
            assert!(m.hierarchy.get(id).is_some_and(|i| i.uses >= *uses), "hierarchy item {id} lost its counts");
        }
        assert!(std::fs::read_dir(mem.join("migrated")).is_ok_and(|mut d| d.next().is_some()), "no copy of the mixed files");
    }
    let episodes: Vec<String> = std::fs::read_to_string(mem.join("episodes.jsonl"))
        .expect("episodes")
        .lines()
        .filter_map(|l| nmjson::parse(l.as_bytes()).ok())
        .filter_map(|v| v.get("id").and_then(Value::string).ok().map(str::to_owned))
        .collect();
    nmapi::tools::install(org).map_err(|e| e.message).expect("install");
    let api = nmapi::tools::api();
    let questions = [
        "Where is the function take_from_shelf defined?",
        "How does the shop refund an order?",
        "What does checkout do when stock runs out?",
        "Which crate keeps stock on a shelf?",
        "Where does a refund put stock back?",
        "Which function takes an item from the shelf?",
        "How is an order refunded in the shop?",
        "What does the checkout test check?",
        "Where is the shelf count kept?",
        "What happens to stock on a refund?",
    ];
    for i in 0..50_usize {
        let q = questions[(i * 7) % questions.len()];
        let (name, input) = match i % 5 {
            0 | 3 => ("nm_context_pack", obj(&[("question", s(q))])),
            1 => ("nm_memory", obj(&[("op", s("recall")), ("text", s(q))])),
            2 => ("nm_feedback", obj(&[("episode", s(&episodes[i % episodes.len()])), ("correct", Value::Bool(i % 3 != 0))])),
            _ => (
                "nm_memory",
                obj(&[
                    ("op", s("temp")),
                    ("text", s(&format!("note about {}", q.split(' ').next_back().unwrap_or("")))),
                    ("scope", s("brain")),
                ]),
            ),
        };
        api.dispatch(name, &input).unwrap_or_else(|e| panic!("call {i} {name}: {}", e.message));
        assert_eq!(nmmem::onewriter::tiers_overlap(&mem), 0, "overlap after call {i} ({name})");
        assert!(!nmmem::onewriter::is_mixed(&mem), "a tier file got a hierarchy item after call {i} ({name})");
    }
}
