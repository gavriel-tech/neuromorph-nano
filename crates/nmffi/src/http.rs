//! HTTPS GET through Windows' own HTTP service (WinHTTP, `winhttp.dll`, part of every Windows),
//! loaded at run time like the card driver. 2026-11-26 (Luis Jake Gabriel III: "Any dependencies
//! should be rebuilt in rust"): the downloads and listings no longer need the `curl.exe` program.
//! TLS stays with Windows (its certificate store and its updates); this module only drives it.
//!
//! Redirects are followed here, not by WinHTTP, so the token goes only to the host it is meant
//! for: a download redirected from huggingface.co to its file servers carries no token there.
use crate::dynlib::Library;
use std::ffi::c_void;
use std::sync::OnceLock;

type Handle = *mut c_void;

/// What a GET reports, in order: the final status once (after redirects), then the body in pieces.
/// The handler returns whether to go on (false after the status: skip the body).
#[derive(Debug)]
pub enum Event<'a> {
    Status(u16),
    Data(&'a [u8]),
}

struct Api {
    open: unsafe extern "system" fn(*const u16, u32, *const u16, *const u16, u32) -> Handle,
    connect: unsafe extern "system" fn(Handle, *const u16, u16, u32) -> Handle,
    open_request: unsafe extern "system" fn(Handle, *const u16, *const u16, *const u16, *const u16, *const *const u16, u32) -> Handle,
    set_option: unsafe extern "system" fn(Handle, u32, *const c_void, u32) -> i32,
    set_timeouts: unsafe extern "system" fn(Handle, i32, i32, i32, i32) -> i32,
    send_request: unsafe extern "system" fn(Handle, *const u16, u32, *const c_void, u32, u32, usize) -> i32,
    receive_response: unsafe extern "system" fn(Handle, *mut c_void) -> i32,
    query_headers: unsafe extern "system" fn(Handle, u32, *const u16, *mut c_void, *mut u32, *mut u32) -> i32,
    query_available: unsafe extern "system" fn(Handle, *mut u32) -> i32,
    read_data: unsafe extern "system" fn(Handle, *mut c_void, u32, *mut u32) -> i32,
    close: unsafe extern "system" fn(Handle) -> i32,
}
// SAFETY: the function pointers come from winhttp.dll, which is never unloaded (dynlib keeps every
// library for the process lifetime); WinHTTP's functions are callable from any thread.
unsafe impl Send for Api {}
// SAFETY: as above; the table is read-only after it is built.
unsafe impl Sync for Api {}

const ACCESS_AUTOMATIC_PROXY: u32 = 4;
const ACCESS_DEFAULT_PROXY: u32 = 0;
const FLAG_SECURE: u32 = 0x0080_0000;
const OPTION_DISABLE_FEATURE: u32 = 63;
const DISABLE_REDIRECTS: u32 = 0x0000_0002;
const QUERY_STATUS_CODE: u32 = 19;
const QUERY_LOCATION: u32 = 33;
const QUERY_FLAG_NUMBER: u32 = 0x2000_0000;
/// `-1L` as a header length: the header string is NUL-terminated.
const HEADERS_NUL_TERMINATED: u32 = u32::MAX;

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = Library::open("winhttp.dll")?;
        let lib: &'static Library = Box::leak(Box::new(lib));
        macro_rules! sym {
            ($name:literal, $t:ty) => {{
                let p = lib.symbol($name)?;
                // SAFETY: `p` is the address of the named WinHTTP export, whose documented
                // signature is `$t` (x64, the system calling convention).
                unsafe { std::mem::transmute::<*mut c_void, $t>(p) }
            }};
        }
        Ok(Api {
            open: sym!("WinHttpOpen", unsafe extern "system" fn(*const u16, u32, *const u16, *const u16, u32) -> Handle),
            connect: sym!("WinHttpConnect", unsafe extern "system" fn(Handle, *const u16, u16, u32) -> Handle),
            open_request: sym!("WinHttpOpenRequest", unsafe extern "system" fn(Handle, *const u16, *const u16, *const u16, *const u16, *const *const u16, u32) -> Handle),
            set_option: sym!("WinHttpSetOption", unsafe extern "system" fn(Handle, u32, *const c_void, u32) -> i32),
            set_timeouts: sym!("WinHttpSetTimeouts", unsafe extern "system" fn(Handle, i32, i32, i32, i32) -> i32),
            send_request: sym!("WinHttpSendRequest", unsafe extern "system" fn(Handle, *const u16, u32, *const c_void, u32, u32, usize) -> i32),
            receive_response: sym!("WinHttpReceiveResponse", unsafe extern "system" fn(Handle, *mut c_void) -> i32),
            query_headers: sym!("WinHttpQueryHeaders", unsafe extern "system" fn(Handle, u32, *const u16, *mut c_void, *mut u32, *mut u32) -> i32),
            query_available: sym!("WinHttpQueryDataAvailable", unsafe extern "system" fn(Handle, *mut u32) -> i32),
            read_data: sym!("WinHttpReadData", unsafe extern "system" fn(Handle, *mut c_void, u32, *mut u32) -> i32),
            close: sym!("WinHttpCloseHandle", unsafe extern "system" fn(Handle) -> i32),
        })
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Whether Windows' HTTP service can be loaded on this machine.
#[must_use]
pub fn available() -> bool {
    api().is_ok()
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A WinHTTP handle closed when dropped.
struct H(Handle);
impl Drop for H {
    fn drop(&mut self) {
        if !self.0.is_null()
            && let Ok(a) = api()
        {
            // SAFETY: the handle came from WinHTTP and is closed once, here.
            unsafe {
                (a.close)(self.0);
            }
        }
    }
}

/// `https://host[:port]/path?query` into (host, port, path and query).
fn split(url: &str) -> Result<(String, u16, String), String> {
    let rest = url.strip_prefix("https://").ok_or_else(|| format!("only https addresses are fetched: {url}"))?;
    let (authority, path) = rest.find('/').map_or((rest, "/"), |i| (&rest[..i], &rest[i..]));
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty() => (h, p.parse::<u16>().map_err(|e| format!("port in {url}: {e}"))?),
        _ => (authority, 443),
    };
    if host.is_empty() {
        return Err(format!("no host in {url}"));
    }
    Ok((host.to_owned(), port, path.to_owned()))
}

fn status_of(a: &Api, req: Handle) -> Result<u16, String> {
    let mut code: u32 = 0;
    let mut len = u32::try_from(size_of::<u32>()).unwrap_or(4);
    // SAFETY: `req` is a request handle whose response was received; `code` is a u32 the size of
    // which is passed in `len`; the name and index pointers may be null.
    let ok = unsafe { (a.query_headers)(req, QUERY_STATUS_CODE | QUERY_FLAG_NUMBER, std::ptr::null(), (&raw mut code).cast::<c_void>(), &raw mut len, std::ptr::null_mut()) };
    if ok == 0 {
        return Err("no status code in the reply".into());
    }
    u16::try_from(code).map_err(|_| format!("status code {code} out of range"))
}

fn location_of(a: &Api, req: Handle) -> Option<String> {
    let mut len: u32 = 0;
    // SAFETY: a null buffer with length 0 asks for the size of the Location header in bytes.
    let _ = unsafe { (a.query_headers)(req, QUERY_LOCATION, std::ptr::null(), std::ptr::null_mut(), &raw mut len, std::ptr::null_mut()) };
    if len == 0 {
        return None;
    }
    let mut buf = vec![0_u16; (len as usize).div_ceil(2) + 1];
    let mut blen = u32::try_from(buf.len() * 2).ok()?;
    // SAFETY: `buf` holds `blen` bytes and outlives the call.
    let ok = unsafe { (a.query_headers)(req, QUERY_LOCATION, std::ptr::null(), buf.as_mut_ptr().cast::<c_void>(), &raw mut blen, std::ptr::null_mut()) };
    if ok == 0 {
        return None;
    }
    let n = (blen as usize) / 2;
    Some(String::from_utf16_lossy(buf.get(..n).unwrap_or(&[])).trim_end_matches('\0').to_owned())
}

/// One GET without following redirects: the status, the Location (for a redirect) and, for any
/// other status, the status and then the body handed to `on`.
fn get_once(url: &str, headers: &str, timeout_ms: i32, on: &mut dyn FnMut(Event<'_>) -> Result<bool, String>) -> Result<(u16, Option<String>), String> {
    let a = api()?;
    let (host, port, path) = split(url)?;
    let agent = wide("neuromorph-nano");
    // SAFETY: `agent` is NUL-terminated and outlives the call; null proxy strings ask for the
    // system's own proxy choice.
    let mut session = H(unsafe { (a.open)(agent.as_ptr(), ACCESS_AUTOMATIC_PROXY, std::ptr::null(), std::ptr::null(), 0) });
    if session.0.is_null() {
        // Windows before 8.1 has no automatic proxy access type; the default proxy is the next best.
        // SAFETY: as above.
        session = H(unsafe { (a.open)(agent.as_ptr(), ACCESS_DEFAULT_PROXY, std::ptr::null(), std::ptr::null(), 0) });
    }
    if session.0.is_null() {
        return Err("Windows' HTTP service did not open a session".into());
    }
    let whost = wide(&host);
    // SAFETY: the session handle is open; `whost` is NUL-terminated and outlives the call.
    let conn = H(unsafe { (a.connect)(session.0, whost.as_ptr(), port, 0) });
    if conn.0.is_null() {
        return Err(format!("could not reach {host}"));
    }
    let (verb, wpath) = (wide("GET"), wide(&path));
    // SAFETY: the connection handle is open; the verb and path are NUL-terminated and outlive the
    // call; null version, referrer and accept types take WinHTTP's defaults.
    let req = H(unsafe { (a.open_request)(conn.0, verb.as_ptr(), wpath.as_ptr(), std::ptr::null(), std::ptr::null(), std::ptr::null(), FLAG_SECURE) });
    if req.0.is_null() {
        return Err(format!("could not open a request to {host}"));
    }
    let off = DISABLE_REDIRECTS;
    // SAFETY: the request handle is open; the option value is a u32 whose size is passed.
    unsafe {
        (a.set_option)(req.0, OPTION_DISABLE_FEATURE, (&raw const off).cast::<c_void>(), 4);
    }
    // SAFETY: the request handle is open; the timeouts are plain milliseconds.
    unsafe {
        (a.set_timeouts)(req.0, 30_000, 30_000, 30_000, timeout_ms);
    }
    let wheaders = wide(headers);
    let (hp, hl) = if headers.is_empty() { (std::ptr::null(), 0) } else { (wheaders.as_ptr(), HEADERS_NUL_TERMINATED) };
    // SAFETY: the request handle is open; the headers are NUL-terminated (or null) and outlive the
    // call; there is no request body.
    let sent = unsafe { (a.send_request)(req.0, hp, hl, std::ptr::null(), 0, 0, 0) };
    if sent == 0 {
        return Err(format!("the request to {host} was not sent (network, proxy or certificate)"));
    }
    // SAFETY: the request was sent; the reserved pointer must be null.
    if unsafe { (a.receive_response)(req.0, std::ptr::null_mut()) } == 0 {
        return Err(format!("no reply from {host}"));
    }
    let status = status_of(a, req.0)?;
    if matches!(status, 301 | 302 | 303 | 307 | 308) {
        return Ok((status, location_of(a, req.0)));
    }
    if !on(Event::Status(status))? {
        return Ok((status, None));
    }
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let mut avail: u32 = 0;
        // SAFETY: the response was received; `avail` is a u32 WinHTTP writes.
        if unsafe { (a.query_available)(req.0, &raw mut avail) } == 0 {
            return Err(format!("the reply from {host} broke off"));
        }
        if avail == 0 {
            break;
        }
        buf.resize(avail as usize, 0);
        let mut read: u32 = 0;
        // SAFETY: `buf` holds `avail` bytes and outlives the call; `read` is written by WinHTTP.
        if unsafe { (a.read_data)(req.0, buf.as_mut_ptr().cast::<c_void>(), avail, &raw mut read) } == 0 {
            return Err(format!("the reply from {host} broke off"));
        }
        if read == 0 {
            break;
        }
        if !on(Event::Data(buf.get(..read as usize).unwrap_or(&[])))? {
            break;
        }
    }
    Ok((status, None))
}

/// GET `url`, following up to eight redirects, and hand the final status and then its body to `on`. `token` is sent
/// as `Authorization: Bearer` only to hosts equal to or ending in `token_host` (never to the
/// servers a redirect leads to elsewhere). `range_from` above zero asks for the bytes from there on
/// (a resumed download). Returns the final status (200, 206, or the error status with its body
/// handed over too).
/// # Errors
/// Windows' HTTP service is missing, an address is not https, or the network fails.
pub fn get(url: &str, token: Option<&str>, token_host: &str, range_from: u64, timeout_ms: u32, on: &mut dyn FnMut(Event<'_>) -> Result<bool, String>) -> Result<u16, String> {
    let mut at = url.to_owned();
    let timeout = i32::try_from(timeout_ms).unwrap_or(i32::MAX);
    for _ in 0..8 {
        let (host, _, _) = split(&at)?;
        let mut headers = String::new();
        if let Some(t) = token
            && !t.is_empty()
            && (host == token_host || host.ends_with(&format!(".{token_host}")))
        {
            headers.push_str(&format!("Authorization: Bearer {t}\r\n"));
        }
        if range_from > 0 {
            headers.push_str(&format!("Range: bytes={range_from}-\r\n"));
        }
        let (status, location) = get_once(&at, &headers, timeout, on)?;
        match location {
            Some(next) => {
                at = if next.starts_with("https://") {
                    next
                } else if next.starts_with('/') {
                    format!("https://{host}{next}")
                } else {
                    return Err(format!("redirect to an address that is not https: {next}"));
                };
            }
            None => return Ok(status),
        }
    }
    Err(format!("too many redirects from {url}"))
}

/// Self-check: the address splitter, and that Windows' HTTP service loads (no request is made).
/// # Errors
/// One fault line.
pub fn check() -> Result<(), String> {
    let (h, p, path) = split("https://huggingface.co/api/models/x/tree/main?recursive=true")?;
    if (h.as_str(), p, path.as_str()) != ("huggingface.co", 443, "/api/models/x/tree/main?recursive=true") {
        return Err(format!("nmffi.http: split gave {h} {p} {path}"));
    }
    if split("http://example.com/").is_ok() {
        return Err("nmffi.http: a plain http address was accepted".into());
    }
    let (h2, p2, path2) = split("https://cdn.example.com:8443")?;
    if (h2.as_str(), p2, path2.as_str()) != ("cdn.example.com", 8443, "/") {
        return Err(format!("nmffi.http: split gave {h2} {p2} {path2}"));
    }
    api().map(|_| ())
}
