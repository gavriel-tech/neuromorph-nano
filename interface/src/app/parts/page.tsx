"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

/** The earlier address of the anatomy page: forwards to it. */
export default function PartsAddress() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/anatomy/");
  }, [router]);
  return (
    <div className="mono px-6 py-10 text-[12px] text-faint">
      This page is now <Link href="/anatomy/" className="link">Anatomy</Link>.
    </div>
  );
}
