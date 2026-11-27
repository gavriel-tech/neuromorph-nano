// Adler-32 checksum, read through the language-neutral entry (Zig has no table entry).
const std = @import("std");

const MOD_ADLER: u32 = 65521;

pub fn adler32(data: []const u8) u32 {
    var a: u32 = 1;
    var b: u32 = 0;
    for (data) |byte| {
        a = (a + byte) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    return (b << 16) | a;
}

fn hexDigit(n: u4) u8 {
    return "0123456789abcdef"[n];
}
