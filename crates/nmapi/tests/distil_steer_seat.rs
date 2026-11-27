//! A distilled steer cartridge on the adapter shelf is seated on the Writer by the serving path,
//! and taken out again when it is unplugged.
//!
//! The steer's build proves its gate and seats the vector it measured (`nmlab::distil`); its file
//! format round-trips in `nmapi::distil`'s unit tests. This test covers the remaining link: a
//! serving process finding a plugged steer file on the shelf of a home and seating it on the
//! Writer it opened (`distil::seat_steer`), then clearing it when the cartridge is unplugged. The
//! steer file is a test fixture written into a throwaway home, never into the Nano's own library,
//! and is not a distilled result: no gate is passed or bypassed for any cartridge a user can plug.
//! Writes only `scratch/lane-steer-seat/`.
//! Run: `cargo test --release -p nmapi --test distil_steer_seat -- --ignored --nocapture`.

use nmjson::Value;

#[test]
#[ignore = "opens the Writer on the graphics card (about a minute)"]
fn a_plugged_steer_file_is_seated_on_the_writer_and_cleared_when_unplugged() {
    let real = nmtype::nano_root();
    let tmp = real.join("scratch/lane-steer-seat");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("throwaway home");

    let m = nmapi::neural::writer_model(&real, true).expect("the Writer opens");
    let width = m.pop.dims.hidden;
    let layer = m.pop.dims.layers - 1;
    let values = nmapi::neural::parameters_key(&m);
    let id = nmapi::distil::cartridge_id("writer", "steer-seat-test");
    let notes = Value::Object(
        [
            ("values".to_owned(), Value::String(values.clone())),
            ("layer".to_owned(), Value::Number(layer.to_string())),
            ("scale".to_owned(), Value::Number("0.25".into())),
        ]
        .into_iter()
        .collect(),
    );
    let label = nmapi::adapter_fit::LabelIn {
        id: &id,
        version: 0,
        model: &m.pop.card.id,
        weights: &m.pop.card.sha_main,
        kind: nmapi::distil::STEER_KIND,
        does: "test fixture: a small steering vector for the seat test",
        made_from: "test fixture",
        notes,
    };
    let v: Vec<f32> = (0..width).map(|i| if i % 2 == 0 { 1e-3 } else { -1e-3 }).collect();
    let (file, version) = nmapi::adapter_fit::write_next(&tmp, label, &[("v".to_owned(), vec![width], v)]).expect("steer file written");
    println!("written {} ({id}@{version})", file.display());

    // Plugged: the serving path seats it on the Writer.
    nmapi::plugs::change(&tmp, |s| {
        s.units_off.remove(&nmapi::plugs::key(nmapi::plugs::NANO, &id));
    })
    .expect("plug it in");
    let line = nmapi::distil::seat_steer(&tmp, &m, "writer");
    println!("plugged: {line}");
    assert!(line.contains(&format!("{id}@{version}")) && line.contains("steering the Writer") && line.contains(&format!("layer {layer}")), "seated: {line}");

    // Unplugged: the next seat takes it out and names nothing.
    nmapi::plugs::change(&tmp, |s| {
        s.units_off.insert(nmapi::plugs::key(nmapi::plugs::NANO, &id));
    })
    .expect("unplug it");
    let line2 = nmapi::distil::seat_steer(&tmp, &m, "writer");
    println!("unplugged: {line2:?}");
    assert!(!line2.contains("steering the Writer"), "cleared when unplugged: {line2}");

    // Made for other parameter values: refused by name, nothing seated.
    let notes_b = Value::Object(
        [
            ("values".to_owned(), Value::String(format!("{values}-other"))),
            ("layer".to_owned(), Value::Number(layer.to_string())),
            ("scale".to_owned(), Value::Number("0.25".into())),
        ]
        .into_iter()
        .collect(),
    );
    let id_b = nmapi::distil::cartridge_id("writer", "steer-seat-other");
    let label_b = nmapi::adapter_fit::LabelIn { id: &id_b, version: 0, model: &m.pop.card.id, weights: &m.pop.card.sha_main, kind: nmapi::distil::STEER_KIND, does: "test fixture", made_from: "test fixture", notes: notes_b };
    nmapi::adapter_fit::write_next(&tmp, label_b, &[("v".to_owned(), vec![width], vec![0.0; width])]).expect("second fixture");
    let line3 = nmapi::distil::seat_steer(&tmp, &m, "writer");
    println!("other values: {line3:?}");
    assert!(!line3.contains("steering the Writer") && line3.contains("no distilled steer seats"), "refused by name: {line3}");

    let _ = std::fs::remove_dir_all(&tmp);
}
