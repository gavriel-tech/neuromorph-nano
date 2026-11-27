package app;

import java.util.ArrayList;
import java.util.List;

/** Entries of one account; the balance never goes below the overdraft limit. */
public class Ledger {
    private static final long OVERDRAFT_LIMIT = -500;
    private final List<Long> entries = new ArrayList<>();

    public Ledger() {
        entries.add(0L);
    }

    public long balance() {
        return entries.stream().mapToLong(Long::longValue).sum();
    }

    public boolean withdraw(long amount) {
        if (balance() - amount < OVERDRAFT_LIMIT) {
            return false;
        }
        entries.add(-amount);
        return true;
    }
}
