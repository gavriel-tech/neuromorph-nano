//! Bounded streaming SHA-256, reused from our loader parity implementation.
use std::io::Read;
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];
fn block(state: &mut [u32; 8], raw: &[u8; 64]) {
    let mut schedule = [0_u32; 64];
    for (index, chunk) in raw.as_chunks::<4>().0.iter().enumerate() {
        schedule[index] = u32::from_be_bytes(*chunk);
    }
    for index in 16..64 {
        let first = schedule[index - 15];
        let second = schedule[index - 2];
        schedule[index] = schedule[index - 16]
            .wrapping_add(first.rotate_right(7) ^ first.rotate_right(18) ^ (first >> 3))
            .wrapping_add(schedule[index - 7])
            .wrapping_add(second.rotate_right(17) ^ second.rotate_right(19) ^ (second >> 10));
    }
    let [mut first, mut second, mut third, mut fourth, mut fifth, mut sixth, mut seventh, mut eighth] = *state;
    for index in 0..64 {
        let temp1 = eighth
            .wrapping_add(fifth.rotate_right(6) ^ fifth.rotate_right(11) ^ fifth.rotate_right(25))
            .wrapping_add((fifth & sixth) ^ (!fifth & seventh))
            .wrapping_add(K[index])
            .wrapping_add(schedule[index]);
        let temp2 = (first.rotate_right(2) ^ first.rotate_right(13) ^ first.rotate_right(22))
            .wrapping_add((first & second) ^ (first & third) ^ (second & third));
        eighth = seventh;
        seventh = sixth;
        sixth = fifth;
        fifth = fourth.wrapping_add(temp1);
        fourth = third;
        third = second;
        second = first;
        first = temp1.wrapping_add(temp2);
    }
    for (word, addend) in state.iter_mut().zip([first, second, third, fourth, fifth, sixth, seventh, eighth]) {
        *word = word.wrapping_add(addend);
    }
}
/// Hash a byte stream without caching its contents.
/// # Errors
/// Returns the underlying stream read error or an unrepresentable byte count.
pub fn sha256(reader: impl Read) -> std::io::Result<String> {
    digest(std::io::BufReader::with_capacity(1024 * 1024, reader))
}
fn digest(mut reader: impl Read) -> std::io::Result<String> {
    let mut state = [0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a, 0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19];
    let mut total = 0_u64;
    let mut raw = [0; 64];
    let mut used = 0;
    // Read 1 MiB at a time and hash its whole blocks in one pass (2026-11-05: reading 64
    // bytes per call made model loads hash at a fraction of the core's speed).
    let mut chunk = vec![0_u8; 1 << 20];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        total += u64::try_from(n).map_err(std::io::Error::other)?;
        let mut data = &chunk[..n];
        if used > 0 {
            let take = (64 - used).min(data.len());
            raw[used..used + take].copy_from_slice(&data[..take]);
            used += take;
            data = &data[take..];
            if used == 64 {
                block(&mut state, &raw);
                used = 0;
            }
        }
        let (blocks, rest) = data.as_chunks::<64>();
        for b in blocks {
            block(&mut state, b);
        }
        raw[..rest.len()].copy_from_slice(rest);
        used += rest.len();
    }
    raw[used] = 0x80;
    raw[used + 1..].fill(0);
    if used >= 56 {
        block(&mut state, &raw);
        raw.fill(0);
    }
    raw[56..].copy_from_slice(&(total * 8).to_be_bytes());
    block(&mut state, &raw);
    let mut text = String::new();
    for word in state {
        std::fmt::Write::write_fmt(&mut text, format_args!("{word:08x}")).map_err(std::io::Error::other)?;
    }
    Ok(text)
}
#[test]
fn known_hashes() -> std::io::Result<()> {
    for (raw, expected) in [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    ] {
        assert_eq!(sha256(raw.as_bytes())?, expected, "known SHA-256");
    }
    assert_eq!(
        digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".as_bytes())?,
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "NIST multi-block padding"
    );
    assert_eq!(
        digest(vec![b'a'; 1_000_000].as_slice())?,
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "NIST million a"
    );
    Ok(())
}
#[test]
fn chunked_reads_give_the_same_digest() -> std::io::Result<()> {
    // A reader that returns odd-sized pieces exercises the carried partial block.
    struct Pieces<'a>(&'a [u8]);
    impl Read for Pieces<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = self.0.len().min(out.len()).min(100_003);
            out[..n].copy_from_slice(&self.0[..n]);
            self.0 = &self.0[n..];
            Ok(n)
        }
    }
    let data = vec![b'a'; 1_000_000];
    let expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    assert_eq!(sha256(data.as_slice())?, expected, "NIST million a, one read");
    assert_eq!(digest(Pieces(&data))?, expected, "NIST million a, odd pieces");
    Ok(())
}
