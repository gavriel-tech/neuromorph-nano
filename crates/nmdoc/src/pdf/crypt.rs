//! Encryption of PDF files with the standard security handler, for files that open with an empty
//! user password (permissions only): MD5 (RFC 1321), RC4, AES-128 and AES-256 in CBC mode (FIPS
//! 197), SHA-256, SHA-384 and SHA-512 (FIPS 180-4), and the key derivations of revisions 2 to 6
//! of the handler. A file that needs a real password is refused; no password is ever asked for.
use super::lex::{Dict, Obj};

// ---------------------------------------------------------------- MD5

const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
    16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// The MD5 digest of `data`.
#[must_use]
pub fn md5(data: &[u8]) -> [u8; 16] {
    // The sine-derived constants of RFC 1321: floor(abs(sin(i + 1)) * 2^32).
    let k: Vec<u32> = (0..64u32).map(|i| (f64::from(i + 1).sin().abs() * 4_294_967_296.0) as u32).collect();
    let mut h: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_le_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        let [mut a, mut b, mut c, mut d] = h;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let t = d;
            d = c;
            c = b;
            b = b.wrapping_add(a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g]).rotate_left(MD5_S[i]));
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
    }
    let mut out = [0u8; 16];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------- RC4

/// RC4 with `key` over `data` (the same call encrypts and decrypts).
#[must_use]
pub fn rc4(key: &[u8], data: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return data.to_vec();
    }
    let mut s: [u8; 256] = [0; 256];
    for (i, v) in s.iter_mut().enumerate() {
        *v = i as u8;
    }
    let mut j: u8 = 0;
    for i in 0..256 {
        j = j.wrapping_add(s[i]).wrapping_add(key[i % key.len()]);
        s.swap(i, usize::from(j));
    }
    let (mut i, mut j) = (0u8, 0u8);
    data.iter()
        .map(|b| {
            i = i.wrapping_add(1);
            j = j.wrapping_add(s[usize::from(i)]);
            s.swap(usize::from(i), usize::from(j));
            b ^ s[usize::from(s[usize::from(i)].wrapping_add(s[usize::from(j)]))]
        })
        .collect()
}

// ---------------------------------------------------------------- AES

/// The AES S-box and its inverse, built from the field inverse and the affine map of FIPS 197.
fn sboxes() -> ([u8; 256], [u8; 256]) {
    let mut sbox = [0u8; 256];
    let mut inv = [0u8; 256];
    let (mut p, mut q): (u8, u8) = (1, 1);
    loop {
        // p runs over the field by multiplying by 3; q by dividing by 3, so q is p's inverse.
        p = p ^ (p << 1) ^ if p & 0x80 != 0 { 0x1B } else { 0 };
        q ^= q << 1;
        q ^= q << 2;
        q ^= q << 4;
        if q & 0x80 != 0 {
            q ^= 0x09;
        }
        let x = q ^ q.rotate_left(1) ^ q.rotate_left(2) ^ q.rotate_left(3) ^ q.rotate_left(4) ^ 0x63;
        sbox[usize::from(p)] = x;
        inv[usize::from(x)] = p;
        if p == 1 {
            break;
        }
    }
    sbox[0] = 0x63;
    inv[0x63] = 0;
    (sbox, inv)
}

fn xtime(a: u8) -> u8 {
    (a << 1) ^ if a & 0x80 != 0 { 0x1B } else { 0 }
}

fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    p
}

/// An expanded AES key (128 or 256 bits).
pub struct Aes {
    rounds: usize,
    w: Vec<[u8; 4]>,
    sbox: [u8; 256],
    inv: [u8; 256],
}

impl Aes {
    /// The key schedule; `None` unless the key is 16 or 32 bytes.
    #[must_use]
    pub fn new(key: &[u8]) -> Option<Self> {
        let nk = match key.len() {
            16 => 4,
            32 => 8,
            _ => return None,
        };
        let (sbox, inv) = sboxes();
        let rounds = nk + 6;
        let mut w: Vec<[u8; 4]> = key.as_chunks::<4>().0.to_vec();
        let mut rcon: u8 = 1;
        for i in nk..4 * (rounds + 1) {
            let mut t = w[i - 1];
            if i % nk == 0 {
                t = [sbox[usize::from(t[1])] ^ rcon, sbox[usize::from(t[2])], sbox[usize::from(t[3])], sbox[usize::from(t[0])]];
                rcon = xtime(rcon);
            } else if nk > 6 && i % nk == 4 {
                t = t.map(|b| sbox[usize::from(b)]);
            }
            let p = w[i - nk];
            w.push([p[0] ^ t[0], p[1] ^ t[1], p[2] ^ t[2], p[3] ^ t[3]]);
        }
        Some(Self { rounds, w, sbox, inv })
    }

    fn add(&self, s: &mut [u8; 16], r: usize) {
        for c in 0..4 {
            for k in 0..4 {
                s[4 * c + k] ^= self.w[4 * r + c][k];
            }
        }
    }

    /// Encrypt one block.
    #[must_use]
    pub fn encrypt(&self, block: [u8; 16]) -> [u8; 16] {
        let mut s = block;
        self.add(&mut s, 0);
        for r in 1..=self.rounds {
            for b in &mut s {
                *b = self.sbox[usize::from(*b)];
            }
            let t = s;
            for c in 0..4 {
                for k in 0..4 {
                    s[4 * c + k] = t[4 * ((c + k) % 4) + k];
                }
            }
            if r != self.rounds {
                for c in 0..4 {
                    let a = [s[4 * c], s[4 * c + 1], s[4 * c + 2], s[4 * c + 3]];
                    let x = a[0] ^ a[1] ^ a[2] ^ a[3];
                    for k in 0..4 {
                        s[4 * c + k] = a[k] ^ x ^ xtime(a[k] ^ a[(k + 1) % 4]);
                    }
                }
            }
            self.add(&mut s, r);
        }
        s
    }

    /// Decrypt one block.
    #[must_use]
    pub fn decrypt(&self, block: [u8; 16]) -> [u8; 16] {
        let mut s = block;
        self.add(&mut s, self.rounds);
        for r in (0..self.rounds).rev() {
            let t = s;
            for c in 0..4 {
                for k in 0..4 {
                    s[4 * ((c + k) % 4) + k] = t[4 * c + k];
                }
            }
            for b in &mut s {
                *b = self.inv[usize::from(*b)];
            }
            self.add(&mut s, r);
            if r != 0 {
                for c in 0..4 {
                    let a = [s[4 * c], s[4 * c + 1], s[4 * c + 2], s[4 * c + 3]];
                    for k in 0..4 {
                        s[4 * c + k] = gmul(a[k], 14) ^ gmul(a[(k + 1) % 4], 11) ^ gmul(a[(k + 2) % 4], 13) ^ gmul(a[(k + 3) % 4], 9);
                    }
                }
            }
        }
        s
    }

    /// CBC decryption; trailing bytes short of a block are dropped.
    #[must_use]
    pub fn cbc_decrypt(&self, iv: [u8; 16], data: &[u8]) -> Vec<u8> {
        let mut prev = iv;
        let mut out = Vec::with_capacity(data.len());
        for c in data.as_chunks::<16>().0 {
            let b = *c;
            let p = self.decrypt(b);
            out.extend(p.iter().zip(prev.iter()).map(|(x, y)| x ^ y));
            prev = b;
        }
        out
    }

    /// CBC encryption of whole blocks (no padding).
    #[must_use]
    pub fn cbc_encrypt(&self, iv: [u8; 16], data: &[u8]) -> Vec<u8> {
        let mut prev = iv;
        let mut out = Vec::with_capacity(data.len());
        for c in data.as_chunks::<16>().0 {
            let mut b = [0u8; 16];
            for (i, x) in b.iter_mut().enumerate() {
                *x = c[i] ^ prev[i];
            }
            prev = self.encrypt(b);
            out.extend_from_slice(&prev);
        }
        out
    }
}

// ---------------------------------------------------------------- SHA-2

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// The SHA-256 digest of `data`.
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K256[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

const K512: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b,
    0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f, 0x80deb1fe3b1696b1,
    0x9bdc06a725c71235, 0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275,
    0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df, 0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
    0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8,
    0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec, 0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915,
    0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6,
    0x113f9804bef90dae, 0x1b710b35131c471b, 0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c, 0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

fn sha512_core(data: &[u8], mut h: [u64; 8], out_len: usize) -> Vec<u8> {
    let mut msg = data.to_vec();
    let bits = (data.len() as u128).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 128 != 112 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.as_chunks::<128>().0 {
        let mut w = [0u64; 80];
        for i in 0..16 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&chunk[8 * i..8 * i + 8]);
            w[i] = u64::from_be_bytes(b);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..80 {
            let s1 = v[4].rotate_right(14) ^ v[4].rotate_right(18) ^ v[4].rotate_right(41);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K512[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(28) ^ v[0].rotate_right(34) ^ v[0].rotate_right(39);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    h.iter().flat_map(|x| x.to_be_bytes()).take(out_len).collect()
}

/// The SHA-384 digest of `data`.
#[must_use]
pub fn sha384(data: &[u8]) -> Vec<u8> {
    sha512_core(
        data,
        [
            0xcbbb9d5dc1059ed8,
            0x629a292a367cd507,
            0x9159015a3070dd17,
            0x152fecd8f70e5939,
            0x67332667ffc00b31,
            0x8eb44a8768581511,
            0xdb0c2e0d64f98fa7,
            0x47b5481dbefa4fa4,
        ],
        48,
    )
}

/// The SHA-512 digest of `data`.
#[must_use]
pub fn sha512(data: &[u8]) -> Vec<u8> {
    sha512_core(
        data,
        [
            0x6a09e667f3bcc908,
            0xbb67ae8584caa73b,
            0x3c6ef372fe94f82b,
            0xa54ff53a5f1d36f1,
            0x510e527fade682d1,
            0x9b05688c2b3e6c1f,
            0x1f83d9abfb41bd6b,
            0x5be0cd19137e2179,
        ],
        64,
    )
}

// ---------------------------------------------------------------- the standard security handler

/// The padding string of the standard security handler.
const PAD: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08, 0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80, 0x2F,
    0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

/// How strings or streams are encrypted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    None,
    Rc4,
    Aes128,
    Aes256,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Rc4 => "RC4",
            Self::Aes128 => "AES-128",
            Self::Aes256 => "AES-256",
        }
    }
}

/// The decryption state of one file: its file key and the methods for streams and strings.
#[derive(Clone, Debug)]
pub struct Crypt {
    key: Vec<u8>,
    pub streams: Method,
    pub strings: Method,
    pub revision: i64,
}

/// Why an encrypted file is not read.
pub const NEEDS_PASSWORD: &str = "encrypted: opening it needs a password";
pub const UNSUPPORTED: &str = "encrypted with a security handler other than the standard one: not read";

fn bytes_of(d: &Dict, k: &str) -> Vec<u8> {
    d.get(k).and_then(Obj::as_str).map(<[u8]>::to_vec).unwrap_or_default()
}

/// The method a crypt filter name stands for.
fn filter_method(d: &Dict, name: Option<&str>) -> Method {
    match name {
        None | Some("Identity") => Method::None,
        Some(n) => {
            let cfm = d.get("CF").and_then(Obj::as_dict).and_then(|cf| cf.get(n)).and_then(Obj::as_dict).and_then(|f| f.name("CFM"));
            match cfm {
                Some("V2") => Method::Rc4,
                Some("AESV2") => Method::Aes128,
                Some("AESV3") => Method::Aes256,
                _ => Method::None,
            }
        }
    }
}

impl Crypt {
    /// Open the encryption dictionary `enc` of a file whose first identifier is `id`, with the
    /// empty user password.
    /// # Errors
    /// `NEEDS_PASSWORD` when the empty password does not open it, `UNSUPPORTED` for another
    /// handler or revision.
    pub fn open(enc: &Dict, id: &[u8]) -> Result<Self, &'static str> {
        if enc.name("Filter") != Some("Standard") {
            return Err(UNSUPPORTED);
        }
        let v = enc.int("V").unwrap_or(0);
        let r = enc.int("R").unwrap_or(0);
        let o = bytes_of(enc, "O");
        let u = bytes_of(enc, "U");
        let p = enc.int("P").unwrap_or(0);
        let (streams, strings) = match v {
            1 | 2 => (Method::Rc4, Method::Rc4),
            4 | 5 => (filter_method(enc, enc.name("StmF")), filter_method(enc, enc.name("StrF"))),
            _ => return Err(UNSUPPORTED),
        };
        if r >= 5 {
            let key = Self::key_r6(enc, &u, r).ok_or(NEEDS_PASSWORD)?;
            return Ok(Self { key, streams, strings, revision: r });
        }
        if !(2..=4).contains(&r) {
            return Err(UNSUPPORTED);
        }
        let len = if r == 2 { 5 } else { usize::try_from(enc.int("Length").unwrap_or(40) / 8).unwrap_or(5).clamp(5, 16) };
        let meta = !matches!(enc.get("EncryptMetadata"), Some(Obj::Bool(false)));
        // Algorithm 2: the file key from the padded (empty) password.
        let mut m = PAD.to_vec();
        m.extend_from_slice(&o);
        m.extend_from_slice(&(p as i32).to_le_bytes());
        m.extend_from_slice(id);
        if r >= 4 && !meta {
            m.extend_from_slice(&[0xFF; 4]);
        }
        let mut h = md5(&m);
        if r >= 3 {
            for _ in 0..50 {
                h = md5(h.get(..len).unwrap_or(&h));
            }
        }
        let key = h.get(..len).unwrap_or(&h).to_vec();
        // Algorithms 4 and 5: check the key against /U.
        let ok = if r == 2 {
            rc4(&key, &PAD) == u.get(..32).unwrap_or(&u)
        } else {
            let mut m = PAD.to_vec();
            m.extend_from_slice(id);
            let mut x = rc4(&key, &md5(&m));
            for i in 1..=19u8 {
                let k: Vec<u8> = key.iter().map(|b| b ^ i).collect();
                x = rc4(&k, &x);
            }
            u.len() >= 16 && x.get(..16) == u.get(..16)
        };
        if !ok {
            return Err(NEEDS_PASSWORD);
        }
        Ok(Self { key, streams, strings, revision: r })
    }

    /// Revisions 5 and 6: check the empty password against /U and take the file key from /UE.
    fn key_r6(enc: &Dict, u: &[u8], r: i64) -> Option<Vec<u8>> {
        let ue = bytes_of(enc, "UE");
        if u.len() < 48 || ue.len() < 32 {
            return None;
        }
        let hash = |salt: &[u8]| if r == 5 { sha256(salt).to_vec() } else { hash_2b(b"", salt, b"") };
        if hash(u.get(32..40)?).get(..32) != u.get(..32) {
            return None;
        }
        let k = hash(u.get(40..48)?);
        let aes = Aes::new(k.get(..32)?)?;
        Some(aes.cbc_decrypt([0; 16], ue.get(..32)?))
    }

    /// A short description for the document's notes.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("encrypted with an empty user password (standard handler revision {}, {}): read as permitted", self.revision, self.streams.name())
    }

    /// The key for object `num gen`.
    fn object_key(&self, m: Method, num: u32, gen_: u16) -> Vec<u8> {
        if m == Method::Aes256 {
            return self.key.clone();
        }
        let mut k = self.key.clone();
        k.extend_from_slice(&num.to_le_bytes()[..3]);
        k.extend_from_slice(&gen_.to_le_bytes());
        if m == Method::Aes128 {
            k.extend_from_slice(b"sAlT");
        }
        let h = md5(&k);
        h.get(..(self.key.len() + 5).min(16)).unwrap_or(&h).to_vec()
    }

    fn apply(&self, m: Method, num: u32, gen_: u16, data: &[u8]) -> Vec<u8> {
        match m {
            Method::None => data.to_vec(),
            Method::Rc4 => rc4(&self.object_key(m, num, gen_), data),
            Method::Aes128 | Method::Aes256 => {
                let k = self.object_key(m, num, gen_);
                let (Some(iv), Some(aes)) = (data.get(..16), Aes::new(&k)) else { return Vec::new() };
                let mut ivb = [0u8; 16];
                ivb.copy_from_slice(iv);
                let mut out = aes.cbc_decrypt(ivb, data.get(16..).unwrap_or(&[]));
                // Remove the padding when it is well formed.
                if let Some(&n) = out.last()
                    && (1..=16).contains(&n)
                    && out.len() >= usize::from(n)
                    && out.iter().rev().take(usize::from(n)).all(|b| *b == n)
                {
                    out.truncate(out.len() - usize::from(n));
                }
                out
            }
        }
    }

    /// Decrypt a stream's bytes.
    #[must_use]
    pub fn stream(&self, num: u32, gen_: u16, data: &[u8]) -> Vec<u8> {
        self.apply(self.streams, num, gen_, data)
    }

    /// Decrypt every string inside an object, in place.
    pub fn strings_in(&self, num: u32, gen_: u16, o: &mut Obj) {
        match o {
            Obj::Str(s) => *s = self.apply(self.strings, num, gen_, s),
            Obj::Array(a) => a.iter_mut().for_each(|x| self.strings_in(num, gen_, x)),
            Obj::Dict(d) => d.0.iter_mut().for_each(|(_, x)| self.strings_in(num, gen_, x)),
            Obj::Stream(s) => s.dict.0.iter_mut().for_each(|(_, x)| self.strings_in(num, gen_, x)),
            _ => {}
        }
    }
}

/// The revision 6 password hash (ISO 32000-2, algorithm 2.B).
#[must_use]
pub fn hash_2b(pw: &[u8], salt: &[u8], udata: &[u8]) -> Vec<u8> {
    let mut first = pw.to_vec();
    first.extend_from_slice(salt);
    first.extend_from_slice(udata);
    let mut k = sha256(&first).to_vec();
    let mut round = 0usize;
    loop {
        let mut one = pw.to_vec();
        one.extend_from_slice(&k);
        one.extend_from_slice(udata);
        let k1: Vec<u8> = one.iter().copied().cycle().take(one.len() * 64).collect();
        let (Some(key), Some(iv)) = (k.get(..16), k.get(16..32)) else { break };
        let Some(aes) = Aes::new(key) else { break };
        let mut ivb = [0u8; 16];
        ivb.copy_from_slice(iv);
        let e = aes.cbc_encrypt(ivb, &k1);
        let r = e.iter().take(16).map(|b| u32::from(*b)).sum::<u32>() % 3;
        k = match r {
            0 => sha256(&e).to_vec(),
            1 => sha384(&e),
            _ => sha512(&e),
        };
        round += 1;
        let last = usize::from(e.last().copied().unwrap_or(0));
        if round >= 64 && last + 32 <= round {
            break;
        }
        if round > 2048 {
            break;
        }
    }
    k.truncate(32);
    k
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub(crate) fn check() -> Result<(), String> {
    let cases: [(&str, String, &str); 7] = [
        ("md5 empty", hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e"),
        ("md5 abc", hex(&md5(b"The quick brown fox jumps over the lazy dog")), "9e107d9d372bb6826bd81d3542a419d6"),
        ("sha256 abc", hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ("sha384 abc", hex(&sha384(b"abc")), "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7"),
        (
            "sha512 abc",
            hex(&sha512(b"abc")),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        ),
        ("rc4 key 0102030405", hex(&rc4(&[1, 2, 3, 4, 5], &[0; 16])), "b2396305f03dc027ccc3524a0a1118a8"),
        ("md5 long", hex(&md5(&[b'a'; 1000])), "cabe45dcc9ae5b66ba86600cca6b8ba8"),
    ];
    for (name, got, want) in cases {
        if got != want {
            return Err(format!("nmdoc.pdf.crypt: {name}: {got}"));
        }
    }
    // FIPS 197 appendix C: AES-128 and AES-256 one block both ways.
    let pt: [u8; 16] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
    let k128: Vec<u8> = (0..16).collect();
    let k256: Vec<u8> = (0..32).collect();
    for (k, want) in [(k128, "69c4e0d86a7b0430d8cdb78070b4c55a"), (k256, "8ea2b7ca516745bfeafc49904b496089")] {
        let aes = Aes::new(&k).ok_or("nmdoc.pdf.crypt: key schedule")?;
        let c = aes.encrypt(pt);
        if hex(&c) != want || aes.decrypt(c) != pt {
            return Err(format!("nmdoc.pdf.crypt: aes-{} gave {}", k.len() * 8, hex(&c)));
        }
        let iv = [7u8; 16];
        let data: Vec<u8> = (0..48).collect();
        if aes.cbc_decrypt(iv, &aes.cbc_encrypt(iv, &data)) != data {
            return Err("nmdoc.pdf.crypt: cbc round trip".into());
        }
    }
    Ok(())
}
