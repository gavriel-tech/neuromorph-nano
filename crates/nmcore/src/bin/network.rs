//! Local network access for tablets (setting `network`: `off`, `view`, `control`): secrets, the
//! six-digit pairing code, paired devices and the address rules. Standard library, nmhash and
//! nmjson only.
//!
//! Secrets: the standard library's hasher keys are drawn from the operating system's random source
//! once per thread; each secret mixes the keyed outputs of four fresh threads (512 bits of keys),
//! the clock and the process id through SHA-256.
//!
//! Pairing: a code is valid for `CODE_MS`, usable once, and stored only as a salted hash. Five
//! wrong codes lock pairing for `LOCK_MS`. A device that sends the right code gets its own random
//! token; only the token's SHA-256 is kept, with the device's name, when it was paired and last
//! seen, and when it was revoked. State lives in `memory/network/`.
//!
//! There is no encryption on the wire: on the local network a device's token and the traffic can
//! be read by anyone who can listen there.
use nmjson::Value;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// How long a pairing code is valid.
pub const CODE_MS: u64 = 5 * 60 * 1000;
/// Wrong codes in a row that lock pairing.
pub const MAX_WRONG: u64 = 5;
/// How long pairing stays locked.
pub const LOCK_MS: u64 = 10 * 60 * 1000;
/// A device's last-seen time is written at most this often.
const SEEN_EVERY_MS: u64 = 60_000;
/// Longest device name kept.
const NAME_CHARS: usize = 40;

static STATE: Mutex<()> = Mutex::new(());

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}
fn n(v: u64) -> Value {
    Value::Number(v.to_string())
}
fn field_u64(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(Value::integer).unwrap_or(0)
}
fn field_str<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::string).unwrap_or("")
}

/// A fresh 256-bit secret as 64 hex digits (see the module note for its sources).
#[must_use]
pub fn random_hex() -> String {
    use std::hash::{BuildHasher, Hasher};
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let draw = |t: u64| -> String {
        let mut out = String::new();
        for i in 0..2u64 {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(t);
            h.write_u64(i);
            out.push_str(&format!("{:016x}", h.finish()));
        }
        out
    };
    let mut seed = String::new();
    for t in 0..4u64 {
        // Each new thread takes its own keys from the operating system; if a thread cannot be
        // made, this thread's keys stand in.
        let part = std::thread::Builder::new().spawn(move || draw(t)).ok().and_then(|h| h.join().ok()).unwrap_or_else(|| draw(t + 100));
        seed.push_str(&part);
    }
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let c = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    seed.push_str(&format!("{nanos}:{}:{c}", std::process::id()));
    nmhash::sha256_bytes(seed.as_bytes())
}

/// A six-digit code from a fresh secret.
fn six_digits() -> String {
    let h = random_hex();
    let x = u64::from_str_radix(h.get(..16).unwrap_or("0"), 16).unwrap_or(0);
    format!("{:06}", x % 1_000_000)
}

/// Compares two byte strings in time that depends only on their lengths.
#[must_use]
pub fn same_secret(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn dir(home: &Path) -> PathBuf {
    home.join("memory").join("network")
}
fn read(p: &Path) -> Option<Value> {
    std::fs::read(p).ok().and_then(|b| nmjson::parse(&b).ok())
}
fn write(p: &Path, v: &Value) -> Result<(), String> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, nmjson::pretty::pretty(v)).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, p).map_err(|e| format!("{}: {e}", p.display()))
}
fn obj(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

/// Pairing state: the code's salted hash and expiry, whether it was used, wrong codes in a row and
/// the end of a lockout.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Pairing {
    salt: String,
    code_sha256: String,
    expires_ms: u64,
    used: bool,
    wrong: u64,
    locked_until_ms: u64,
}
impl Pairing {
    fn load(home: &Path) -> Self {
        read(&dir(home).join("pairing.json")).map_or_else(Self::default, |v| Self {
            salt: field_str(&v, "salt").to_owned(),
            code_sha256: field_str(&v, "code_sha256").to_owned(),
            expires_ms: field_u64(&v, "expires_ms"),
            used: matches!(v.get("used"), Ok(Value::Bool(true))),
            wrong: field_u64(&v, "wrong"),
            locked_until_ms: field_u64(&v, "locked_until_ms"),
        })
    }
    fn save(&self, home: &Path) -> Result<(), String> {
        write(
            &dir(home).join("pairing.json"),
            &obj(vec![
                ("note", s("Pairing state of the Neuromorph Nano's local network access: the code is kept only as a salted hash.")),
                ("salt", s(&self.salt)),
                ("code_sha256", s(&self.code_sha256)),
                ("expires_ms", n(self.expires_ms)),
                ("used", Value::Bool(self.used)),
                ("wrong", n(self.wrong)),
                ("locked_until_ms", n(self.locked_until_ms)),
            ]),
        )
    }
}

/// Why a pairing attempt failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairError {
    /// Pairing is locked until this time (milliseconds since 1970).
    Locked(u64),
    /// No code is open, it expired, it was used, or it is wrong; this many wrong in a row now.
    Wrong(u64),
    /// The state could not be read or written.
    Store(String),
}
impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Locked(t) => write!(f, "pairing is locked for {} more seconds after {MAX_WRONG} wrong codes", t.saturating_sub(nmtype::now_ms()) / 1000 + 1),
            Self::Wrong(k) => write!(f, "wrong, expired or used code ({k} of {MAX_WRONG} before pairing locks)"),
            Self::Store(e) => write!(f, "pairing state: {e}"),
        }
    }
}

/// Makes a new pairing code (the last one stops working). Refused while pairing is locked.
/// # Errors
/// Locked, or the state cannot be written.
pub fn new_code(home: &Path, by: &str) -> Result<(String, u64), PairError> {
    let _g = STATE.lock().map_err(|_| PairError::Store("lock poisoned".into()))?;
    let now = nmtype::now_ms();
    let mut p = Pairing::load(home);
    if p.locked_until_ms > now {
        return Err(PairError::Locked(p.locked_until_ms));
    }
    let code = six_digits();
    p.salt = random_hex();
    p.code_sha256 = nmhash::sha256_bytes(format!("{}:{code}", p.salt).as_bytes());
    p.expires_ms = now + CODE_MS;
    p.used = false;
    p.save(home).map_err(PairError::Store)?;
    let _ = crate::controls::audit(
        home,
        "pairing",
        vec![("label", s("Pairing code made (valid five minutes, once)")), ("action", s("code")), ("by", s(by)), ("expires_ms", n(p.expires_ms))],
    );
    Ok((code, p.expires_ms))
}

/// One paired device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub token_sha256: String,
    pub paired_ms: u64,
    pub last_seen_ms: u64,
    pub revoked_ms: u64,
}
impl Device {
    fn from_value(v: &Value) -> Self {
        Self {
            id: field_str(v, "id").to_owned(),
            name: field_str(v, "name").to_owned(),
            token_sha256: field_str(v, "token_sha256").to_owned(),
            paired_ms: field_u64(v, "paired_ms"),
            last_seen_ms: field_u64(v, "last_seen_ms"),
            revoked_ms: field_u64(v, "revoked_ms"),
        }
    }
    fn to_value(&self, with_hash: bool) -> Value {
        let mut f = vec![
            ("id", s(&self.id)),
            ("name", s(&self.name)),
            ("paired_ms", n(self.paired_ms)),
            ("last_seen_ms", n(self.last_seen_ms)),
            ("revoked", Value::Bool(self.revoked_ms > 0)),
            ("revoked_ms", n(self.revoked_ms)),
        ];
        if with_hash {
            f.push(("token_sha256", s(&self.token_sha256)));
        }
        obj(f)
    }
}
fn devices(home: &Path) -> Vec<Device> {
    read(&dir(home).join("devices.json"))
        .and_then(|v| v.get("devices").and_then(Value::array).ok().map(|a| a.iter().map(Device::from_value).collect()))
        .unwrap_or_default()
}
fn save_devices(home: &Path, list: &[Device]) -> Result<(), String> {
    write(
        &dir(home).join("devices.json"),
        &obj(vec![
            ("note", s("Devices paired with the Neuromorph Nano's interface. Tokens are kept only as SHA-256 hashes.")),
            ("devices", Value::Array(list.iter().map(|d| d.to_value(true)).collect())),
        ]),
    )
}

/// A device name as kept: printable characters, spacing collapsed, at most `NAME_CHARS`.
fn clean_name(name: &str) -> String {
    let one: String =
        name.chars().map(|c| if c.is_control() { ' ' } else { c }).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
    let one: String = one.chars().take(NAME_CHARS).collect();
    if one.is_empty() { "tablet".to_owned() } else { one }
}

/// Pairs a device with the open code: returns the new device and its token (shown once, never
/// kept). Wrong codes count toward the lockout, also when no code is open.
/// # Errors
/// Locked, wrong, or the state cannot be written.
pub fn pair(home: &Path, code: &str, name: &str, peer: &str) -> Result<(Device, String), PairError> {
    let _g = STATE.lock().map_err(|_| PairError::Store("lock poisoned".into()))?;
    let now = nmtype::now_ms();
    let mut p = Pairing::load(home);
    if p.locked_until_ms > now {
        return Err(PairError::Locked(p.locked_until_ms));
    }
    let given = nmhash::sha256_bytes(format!("{}:{}", p.salt, code.trim()).as_bytes());
    let open = !p.code_sha256.is_empty() && !p.used && now <= p.expires_ms;
    if !(open && same_secret(given.as_bytes(), p.code_sha256.as_bytes())) {
        p.wrong += 1;
        let wrong = p.wrong;
        let locked = wrong >= MAX_WRONG;
        if locked {
            p.locked_until_ms = now + LOCK_MS;
            p.wrong = 0;
        }
        p.save(home).map_err(PairError::Store)?;
        let _ = crate::controls::audit(
            home,
            "pairing",
            vec![
                ("label", s(if locked { "Pairing locked for ten minutes after five wrong codes" } else { "Pairing refused: wrong code" })),
                ("action", s(if locked { "locked" } else { "refused" })),
                ("peer", s(peer)),
                ("wrong", n(wrong)),
            ],
        );
        return Err(if locked { PairError::Locked(p.locked_until_ms) } else { PairError::Wrong(wrong) });
    }
    p.used = true;
    p.wrong = 0;
    p.save(home).map_err(PairError::Store)?;
    let token = random_hex();
    let d = Device {
        id: format!("d{}", random_hex().get(..12).unwrap_or("0")),
        name: clean_name(name),
        token_sha256: nmhash::sha256_bytes(token.as_bytes()),
        paired_ms: now,
        last_seen_ms: now,
        revoked_ms: 0,
    };
    let mut list = devices(home);
    list.push(d.clone());
    save_devices(home, &list).map_err(PairError::Store)?;
    let _ = crate::controls::audit(
        home,
        "pairing",
        vec![
            ("label", s(&format!("Device paired: {}", d.name))),
            ("action", s("paired")),
            ("device", s(&d.id)),
            ("device_name", s(&d.name)),
            ("peer", s(peer)),
        ],
    );
    Ok((d, token))
}

/// The paired, unrevoked device a token belongs to; its last-seen time is brought up to date.
#[must_use]
pub fn authenticate(home: &Path, token: &str) -> Option<Device> {
    if token.len() != 64 {
        return None;
    }
    let want = nmhash::sha256_bytes(token.as_bytes());
    let _g = STATE.lock().ok()?;
    let mut list = devices(home);
    let i = list.iter().position(|d| d.revoked_ms == 0 && same_secret(d.token_sha256.as_bytes(), want.as_bytes()))?;
    let now = nmtype::now_ms();
    let d = list.get_mut(i)?;
    if now.saturating_sub(d.last_seen_ms) >= SEEN_EVERY_MS {
        d.last_seen_ms = now;
        let out = d.clone();
        let _ = save_devices(home, &list);
        return Some(out);
    }
    Some(d.clone())
}

/// Revokes a device: its token is refused from then on.
/// # Errors
/// No such device, or the list cannot be written.
pub fn revoke(home: &Path, id: &str, by: &str) -> Result<(), String> {
    let _g = STATE.lock().map_err(|_| "lock poisoned".to_owned())?;
    let mut list = devices(home);
    let d = list.iter_mut().find(|d| d.id == id).ok_or_else(|| format!("no device {id}"))?;
    if d.revoked_ms > 0 {
        return Ok(());
    }
    d.revoked_ms = nmtype::now_ms();
    let name = d.name.clone();
    save_devices(home, &list)?;
    let _ = crate::controls::audit(
        home,
        "pairing",
        vec![("label", s(&format!("Device revoked: {name}"))), ("action", s("revoked")), ("device", s(id)), ("device_name", s(&name)), ("by", s(by))],
    );
    Ok(())
}

/// The devices and the pairing state for the interface (no hashes, no code).
#[must_use]
pub fn view(home: &Path, mode: &str, addrs: &[IpAddr], port: u16) -> Value {
    let p = Pairing::load(home);
    let now = nmtype::now_ms();
    obj(vec![
        ("network", s(mode)),
        ("addresses", Value::Array(addrs.iter().map(|a| s(&format!("http://{a}:{port}/"))).collect())),
        ("devices", Value::Array(devices(home).iter().map(|d| d.to_value(false)).collect())),
        ("code_open", Value::Bool(!p.code_sha256.is_empty() && !p.used && now <= p.expires_ms)),
        ("code_expires_ms", n(p.expires_ms)),
        ("locked_until_ms", n(if p.locked_until_ms > now { p.locked_until_ms } else { 0 })),
        ("encryption", s("none: on the local network the traffic, including a device's token, can be read by anyone listening there")),
    ])
}

/// Private, link-local or loopback: the only source addresses served in `view` and `control`.
#[must_use]
pub fn is_private(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v) => v.is_private() || v.is_link_local() || v.is_loopback(),
        IpAddr::V6(v) => {
            if let Some(m) = v.to_ipv4_mapped() {
                return m.is_private() || m.is_link_local() || m.is_loopback();
            }
            let first = v.segments().first().copied().unwrap_or(0);
            v.is_loopback() || (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80
        }
    }
}

/// This machine's private IPv4 addresses, found by asking the operating system which local
/// address it would use toward each private range (a UDP socket is connected, nothing is sent).
#[must_use]
pub fn local_addresses() -> Vec<IpAddr> {
    let mut out: Vec<IpAddr> = Vec::new();
    for target in ["10.255.255.254:9", "172.31.255.254:9", "192.168.255.254:9", "192.168.0.1:9", "192.168.1.1:9", "169.254.255.254:9"] {
        let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") else { continue };
        if sock.connect(target).is_err() {
            continue;
        }
        if let Ok(a) = sock.local_addr() {
            let ip = a.ip();
            if is_private(&ip) && !ip.is_loopback() && !ip.is_unspecified() && !out.contains(&ip) {
                out.push(ip);
            }
        }
    }
    out
}

/// The page an unpaired device gets: a form for the six-digit code and a name.
pub const PAIR_PAGE: &str = r#"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Pair this device</title>
<style>:root{color-scheme:light dark}body{font-family:system-ui,sans-serif;max-width:28rem;margin:3rem auto;padding:0 1rem;line-height:1.5}input,button{font:inherit;padding:.5rem;margin:.25rem 0;width:100%;box-sizing:border-box}p.note{font-size:.9rem;opacity:.8}</style></head>
<body><h1>Neuromorph Nano</h1><p>Pair this device: enter the six-digit code shown on the computer (in the interface, or by <code>neuromorph-nano pair</code>).</p>
<form id="f"><label>Code<input id="code" inputmode="numeric" autocomplete="one-time-code" maxlength="6" required></label>
<label>Name of this device<input id="name" maxlength="40" value="tablet"></label><button>Pair</button></form><p id="msg" role="status"></p>
<p class="note">No encryption on the wire: on this local network the traffic, including this device's token, can be read by anyone listening there.</p>
<script>document.getElementById('f').onsubmit=async function(e){e.preventDefault();var m=document.getElementById('msg');m.textContent='Pairing...';try{var r=await fetch('/api/pair',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:document.getElementById('code').value,name:document.getElementById('name').value})});var j=await r.json();if(r.ok){m.textContent='Paired as '+j.name+'.';location.href='/';}else{m.textContent=j.error||'Refused.';}}catch(x){m.textContent='No answer from the computer.';}};</script></body></html>"#;

/// Section check (temporary folder, no network): codes are six digits, used once, expire, and
/// five wrong codes lock pairing; a paired token authenticates until revoked; only private
/// addresses pass; secrets are 64 hex digits and fresh.
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let (a, b) = (random_hex(), random_hex());
    if a.len() != 64 || !a.chars().all(|c| c.is_ascii_hexdigit()) || a == b {
        return Err("network: secrets must be 64 hex digits and fresh".into());
    }
    let ip = |t: &str| t.parse::<IpAddr>().map_err(|e| e.to_string());
    for (addr, want) in [
        ("192.168.1.20", true),
        ("10.0.0.5", true),
        ("172.20.1.1", true),
        ("169.254.3.4", true),
        ("127.0.0.1", true),
        ("8.8.8.8", false),
        ("172.32.0.1", false),
        ("fe80::1", true),
        ("fd00::7", true),
        ("2001:db8::1", false),
        ("::ffff:192.168.0.9", true),
    ] {
        if is_private(&ip(addr)?) != want {
            return Err(format!("network: private address rule for {addr}"));
        }
    }
    let home = std::env::temp_dir().join(format!("nano-network-check-{}-{}", std::process::id(), nmtype::now_ms()));
    let r = check_in(&home);
    let _ = std::fs::remove_dir_all(&home);
    r
}
fn check_in(home: &Path) -> Result<(), String> {
    let (code, _) = new_code(home, "check").map_err(|e| e.to_string())?;
    if code.len() != 6 || !code.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("network: code {code} is not six digits"));
    }
    let other = if code == "000000" { "000001" } else { "000000" };
    if !matches!(pair(home, other, "x", "test"), Err(PairError::Wrong(1))) {
        return Err("network: a wrong code was not refused".into());
    }
    let (d, token) = pair(home, &code, "  kitchen\ttablet ", "test").map_err(|e| e.to_string())?;
    if d.name != "kitchen tablet" || authenticate(home, &token).map(|x| x.id) != Some(d.id.clone()) {
        return Err("network: the paired token does not authenticate".into());
    }
    if pair(home, &code, "again", "test").is_ok() {
        return Err("network: a code was used twice".into());
    }
    let stored = std::fs::read_to_string(dir(home).join("devices.json")).map_err(|e| e.to_string())?;
    if stored.contains(&token) {
        return Err("network: a token was stored in clear".into());
    }
    revoke(home, &d.id, "check")?;
    if authenticate(home, &token).is_some() {
        return Err("network: a revoked token authenticates".into());
    }
    // An expired code is refused.
    let (late, _) = new_code(home, "check").map_err(|e| e.to_string())?;
    let mut p = Pairing::load(home);
    p.expires_ms = 1;
    p.save(home)?;
    if pair(home, &late, "late", "test").is_ok() {
        return Err("network: an expired code was accepted".into());
    }
    // Wrong codes lock pairing: the right code is refused while locked.
    let (code2, _) = new_code(home, "check").map_err(|e| e.to_string())?;
    let bad = if code2 == "999999" { "999998" } else { "999999" };
    let mut last = Err(PairError::Wrong(0));
    for _ in 0..MAX_WRONG {
        last = pair(home, bad, "x", "test");
    }
    if !matches!(last, Err(PairError::Locked(_))) || !matches!(pair(home, &code2, "x", "test"), Err(PairError::Locked(_))) {
        return Err("network: five wrong codes did not lock pairing".into());
    }
    if new_code(home, "check").is_ok() {
        return Err("network: a code was made while pairing was locked".into());
    }
    Ok(())
}
