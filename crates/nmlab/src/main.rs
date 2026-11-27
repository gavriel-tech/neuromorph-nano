//! `nano-lab`: the lab commands as their own program (the same commands run inside the product as
//! `neuromorph-nano lab <command>`).
use std::process::ExitCode;

fn main() -> ExitCode {
    nmlab::run(std::env::args().skip(1).collect())
}
