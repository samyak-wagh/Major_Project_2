"""
kb_advanced.py — Applied, multi-concept, exam-synthesis OS knowledge base.
=============================================================================
kb_conceptual.py covers direct textbook recall ("what is X", "compare X and
Y"). Real exams build questions OVER concepts — a scenario that requires
combining two or three ideas and reasoning through a mechanism, the way a
strong general LLM (Claude/GPT) answers when asked cold. This file targets
exactly that style: each answer is a full reasoning chain (phenomenon ->
underlying mechanism -> resolution), not a one-line fact.

Same CHAPTERS structure as kb_conceptual.py so build_conceptual.py can merge
both without any special-casing. Topics here deliberately go beyond a
first-pass undergrad syllabus into material a rigorous OS course/exam
legitimately draws on: hardware memory models, applied synchronization
scenarios, real-time scheduling, and applied virtual memory / security.
"""

CHAPTERS = [
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Hardware Memory Models and Kernel Synchronization Internals",
        "topics": [
            {
                "id": "memory_barriers_reordering",
                "context": (
                    "On multi-core processors, each core often has a private "
                    "store buffer that lets writes complete locally before "
                    "they become visible to other cores, and a relaxed memory "
                    "model may allow the CPU to reorder a store followed by an "
                    "unrelated load unless an explicit memory barrier forbids "
                    "it. This StoreLoad reordering is the one reordering even "
                    "relatively strong models like x86-TSO still permit."
                ),
                "qas": [
                    ("Two threads share global variables X and Y, both initialized to 0, with no synchronization. "
                     "Thread A executes 'X = 1; r1 = Y;' and Thread B executes 'Y = 1; r2 = X;'. On a processor "
                     "with a relaxed memory model, is it possible for both r1 == 0 and r2 == 0 after both threads "
                     "finish? Explain using CPU store buffers.",
                     "Yes, this outcome is possible on a relaxed memory model, even though it looks impossible if "
                     "you reason about the code as if each thread executed in strict program order relative to the "
                     "other. The reason is the CPU store buffer: when Thread A executes 'X = 1', the write does not "
                     "go directly to shared memory — it is first placed in Thread A's private store buffer, and the "
                     "core is free to continue executing later instructions (like the read of Y) before that "
                     "buffered write has actually become visible to other cores. So Thread A can execute 'r1 = Y' "
                     "and see the pre-update value of Y (0) before its own write to X has drained out and become "
                     "globally visible. Symmetrically, Thread B can read the old value of X (0) before its write to "
                     "Y is visible. From each thread's own local point of view, its store still 'happened before' "
                     "its load — program order is preserved locally — but from a global point of view both loads "
                     "can observe pre-update values because neither store had propagated out of its store buffer "
                     "yet. This is StoreLoad reordering, and it means r1 == 0 && r2 == 0 is a real, reachable "
                     "outcome without synchronization."),
                    ("If you need to guarantee that r1 == 0 && r2 == 0 is absolutely impossible in the scenario above, "
                     "what specific primitive must be inserted, and exactly where?",
                     "A full memory barrier (fence) — e.g. x86's MFENCE instruction, C11's "
                     "atomic_thread_fence(memory_order_seq_cst), or the Linux kernel's smp_mb() — must be inserted "
                     "between the store and the load in EACH thread: Thread A becomes 'X = 1; memory_barrier(); "
                     "r1 = Y;' and Thread B becomes 'Y = 1; memory_barrier(); r2 = X;'. A full fence forces the "
                     "core to drain its store buffer (making the preceding store globally visible) before any "
                     "subsequent load is allowed to execute, restoring StoreLoad ordering. With the barrier present "
                     "in both threads, whichever store becomes globally visible first is guaranteed to be observed "
                     "by the other thread's subsequent load, so at least one of r1, r2 must read 1 — making "
                     "'r1==0 && r2==0' provably impossible. Critically, the barrier must be in BOTH threads: a "
                     "barrier in only one thread does not prevent the anomaly, since the other thread's load can "
                     "still race ahead of its own unbuffered store."),
                    ("How does the OS kernel ensure hardware memory reordering doesn't break internal primitives "
                     "like spinlocks or mutexes?",
                     "Kernel lock implementations bake explicit memory-barrier semantics into acquire and release, "
                     "so ordinary code using the lock never has to reason about hardware reordering itself. A lock "
                     "ACQUIRE is implemented with 'acquire' semantics — a barrier that prevents any memory "
                     "operation AFTER the lock in program order from being reordered to execute BEFORE the lock is "
                     "actually held, so the critical section's reads/writes can never leak out before the lock is "
                     "taken. A lock RELEASE is implemented with 'release' semantics — a barrier that prevents any "
                     "memory operation BEFORE the unlock in program order from being reordered to execute AFTER "
                     "the unlock, so critical-section writes cannot become visible to other cores only after the "
                     "lock appears free. In Linux this is done with explicit primitives (smp_mb__before_atomic(), "
                     "smp_mb__after_atomic()) or architecture-specific acquire/release instructions (ARM's "
                     "LDAR/STLR, or x86's LOCK-prefixed atomic instructions for the compare-and-swap that "
                     "implements the lock itself). The correctness burden is pushed into the lock primitive, so "
                     "kernel code that simply calls lock()/unlock() sees a consistent view of shared data "
                     "regardless of how aggressively the underlying hardware would otherwise reorder memory "
                     "operations."),
                    ("What is the difference between a store buffer and an invalidation queue, and how does each "
                     "contribute to memory reordering?",
                     "A store buffer is a small, per-core hardware structure that holds a core's own pending "
                     "writes before they are committed to the shared cache/memory hierarchy, letting the core keep "
                     "executing without stalling on every write; it causes StoreLoad reordering because a core's "
                     "own later load can execute before an earlier store from the same core has drained out and "
                     "become visible elsewhere. An invalidation queue sits on the receiving side of a "
                     "cache-coherence protocol: when another core's write invalidates a cache line this core "
                     "holds, the invalidation can be queued and acknowledged immediately rather than applied "
                     "right away, letting this core continue reading a stale (not-yet-invalidated) cached value "
                     "for a short window even though another core has already written a new value. Both effects "
                     "combine to produce the reordering/staleness that a memory consistency model must define "
                     "rules for, and that memory barriers exist to control."),
                ],
            },
            {
                "id": "tlb_shootdown",
                "context": (
                    "Each core has its own private TLB caching virtual-to-physical translations, and TLBs "
                    "are not automatically kept coherent with page-table memory the way data caches are "
                    "coherent via MESI. When the kernel changes a page table entry that other cores may have "
                    "cached (e.g. during page migration, compaction, or swap-out), it must perform a TLB "
                    "shootdown: an Inter-Processor Interrupt (IPI) to every core that might hold a stale "
                    "translation, forcing each to flush it, before the physical frame can safely be reused."
                ),
                "qas": [
                    ("Core 1 wants to reclaim a physical frame mapped into the address space of a multi-threaded "
                     "process. Why can't it simply update the page table entry in memory and flush only its own "
                     "local TLB? Describe the exact catastrophic race that occurs on Core 2, which is running "
                     "another thread of the same process, if a global TLB shootdown is not initiated.",
                     "Each core has its own private TLB, and TLBs are not automatically kept coherent with "
                     "page-table memory the way data caches are coherent with each other via MESI — there is no "
                     "hardware snooping protocol watching page-table writes and invalidating other cores' TLB "
                     "entries. So if Core 1 only updates the PTE in memory and flushes its own local TLB, Core 2 "
                     "still has the OLD virtual-to-physical mapping cached. The catastrophe: Core 1 is typically "
                     "reclaiming the physical frame because it is about to reuse it for something else entirely — "
                     "hand it to a different process, a DMA buffer, or a kernel data structure. Core 2, unaware "
                     "anything changed, accesses the same virtual address; its stale TLB entry lets that access "
                     "go straight to the physical frame WITHOUT ever walking the (now-updated) page table — "
                     "bypassing the page table is precisely the point of a TLB hit. Core 2 therefore silently "
                     "reads or writes physical memory that has been reassigned to something unrelated: this can "
                     "corrupt another process's data, leak sensitive data across a security boundary, or corrupt "
                     "kernel structures, and because it depends on timing it may not manifest immediately or "
                     "deterministically, making it extremely hard to debug."),
                    ("Walk through the step-by-step architectural mechanism the kernel uses to orchestrate a TLB "
                     "shootdown across multiple cores. What hardware-level communication protocol interrupts the "
                     "other cores?",
                     "(1) Core 1 updates (or clears) the page table entry in memory first, so any future TLB miss "
                     "on any core picks up the correct, fresh translation. (2) Core 1 determines which cores could "
                     "plausibly have this mapping cached — not a broadcast to every core in the system, but the "
                     "process's tracked cpumask (Linux calls this mm_cpumask: the set of cores the process has "
                     "actually run on recently), as a targeting optimization. (3) Core 1 sends an Inter-Processor "
                     "Interrupt (IPI) to exactly those cores — the hardware communication protocol: on x86, Core "
                     "1 writes to its Local APIC's Interrupt Command Register (ICR) specifying the destination "
                     "core(s) and interrupt vector, and the APIC hardware delivers a genuine hardware interrupt "
                     "to each target core almost immediately, even preempting user-mode execution (ARM's "
                     "equivalent is an SGI, Software Generated Interrupt, via the GIC). (4) Each target core's "
                     "registered IPI handler runs, executing a local TLB invalidation — either a single-address "
                     "INVLPG, or a full flush (reloading CR3) if many entries are affected. (5) Each target core "
                     "signals completion back to Core 1, typically via a shared atomic counter or bitmap. (6) "
                     "Core 1 BLOCKS until every target core has acknowledged completion before proceeding — it "
                     "must not let the reclaimed physical frame be reused for anything else until it is certain "
                     "no core can still reach it via a stale TLB entry."),
                    ("TLB shootdowns are notoriously expensive and scale poorly as core count increases. Identify "
                     "the two main reasons for this, and name one optimization technique modern OS kernels use to "
                     "avoid or minimize full shootdowns.",
                     "(1) IPI fan-out cost: sending and handling an interrupt on N cores is not free, and the "
                     "cost scales with the number of cores that must be interrupted; every interrupted core "
                     "suffers a 'stop the world' disruption to whatever it was doing, and on high-core-count "
                     "systems under heavy memory-management churn this can become an 'IPI storm'. (2) "
                     "Synchronous blocking: the initiating core must wait for EVERY targeted core to acknowledge "
                     "before it is safe to proceed (for correctness, per the race condition above) — if even one "
                     "target core is momentarily unavailable (e.g. running with interrupts disabled), the whole "
                     "operation stalls, and this delay sits directly on the critical path of memory reclaim, "
                     "compaction, or migration. Standard mitigation: batched/deferred shootdowns — e.g. Linux's "
                     "mmu_gather / tlb_gather_mmu machinery accumulates many PTE changes (such as during a large "
                     "munmap or a reclaim pass) and issues a SINGLE shootdown round covering all of them, instead "
                     "of one IPI round per individual PTE change, amortizing the fixed IPI cost across many "
                     "changes. (Restricting the IPI target set to the process's actual cpumask rather than "
                     "broadcasting to all cores, as in the mechanism above, is the other standard mitigation.)"),
                ],
            },
            {
                "id": "cache_coherence_mesi",
                "context": (
                    "On a multicore system, each core has its own private cache; without coordination, two "
                    "cores could each cache a different copy of the same memory location. Cache coherence "
                    "protocols like MESI (Modified, Exclusive, Shared, Invalid) keep caches consistent by "
                    "tagging each cache line with a state and broadcasting invalidations on writes."
                ),
                "qas": [
                    ("What problem does a cache coherence protocol like MESI solve, and what do its four states mean?",
                     "Without coordination, two cores could each cache their own (possibly different) copy of the "
                     "same memory location, and a write by one core would not automatically be visible to another "
                     "core still reading its own stale cached copy — this is the cache coherence problem. MESI "
                     "tags each cache line with one of four states: Modified — this cache holds the only copy, "
                     "and it has been written (dirty, differs from memory; must be written back before eviction "
                     "or sharing). Exclusive — this cache holds the only copy, and it is clean (matches memory; "
                     "can be silently upgraded to Modified on a write with no notification needed). Shared — this "
                     "line may also be cached (clean) by other cores simultaneously; a core must broadcast an "
                     "invalidation to other sharers before writing. Invalid — this line holds no valid data and "
                     "must be fetched before use. When a core wants to write to a Shared or Invalid line, it "
                     "broadcasts an invalidate message so all other cores drop their copies, guaranteeing only "
                     "one core can have write access at a time and all cores agree on the most recent value."),
                ],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Applied Synchronization and Deadlock Scenarios",
        "topics": [
            {
                "id": "priority_inversion_scenario",
                "context": (
                    "Priority inversion occurs when a lower-priority task holds a resource a higher-priority "
                    "task needs. Without a fix, an unrelated medium-priority task can indefinitely preempt the "
                    "low-priority holder, delaying the high-priority task by an unbounded amount — this is the "
                    "scenario the Priority Inheritance Protocol is designed to bound."
                ),
                "qas": [
                    ("Three tasks run on a single-core real-time system: Low (L, lowest priority) locks a shared "
                     "mutex M to access a resource; Medium (a CPU-bound task that never touches M); and High (H, "
                     "highest priority), which needs to lock M. Trace what happens if H becomes ready while L "
                     "holds M and Medium is also ready, under a naive locking scheme with no priority "
                     "inheritance, and explain why this is called 'unbounded priority inversion'.",
                     "(1) L acquires mutex M and begins its critical section. (2) H becomes ready and, being "
                     "highest priority, preempts L; H then tries to lock M, but M is held by L, so H blocks and "
                     "waits. (3) L is not immediately allowed to resume either — Medium is also ready and has "
                     "higher priority than L, so the scheduler picks Medium to run instead of L. (4) Medium now "
                     "runs for as long as it likes, never touching M and never itself blocked, so it can keep "
                     "preempting L indefinitely. The net effect: H, the highest-priority task, is delayed not "
                     "directly by L (a lower-priority task — an expected, bounded delay) but INDIRECTLY by "
                     "Medium, a task with priority strictly between L and H that has no logical relationship to "
                     "the resource H is waiting for. Since Medium's execution time is not bounded by the length "
                     "of L's critical section, H's wait time has no upper bound — this is 'unbounded priority "
                     "inversion'. The standard fix is the Priority Inheritance Protocol: while L holds a resource "
                     "a higher-priority task is blocked on, L's priority is temporarily boosted to match the "
                     "highest priority of any task blocked on it (here, to H's priority), so the scheduler runs L "
                     "instead of letting Medium preempt it — bounding H's wait to, at most, the length of L's "
                     "critical section."),
                ],
            },
            {
                "id": "race_condition_code_spotting",
                "context": (
                    "The statement 'counter++' is not atomic at the machine-instruction level: it typically "
                    "compiles to a separate load, add, and store, and a race between two threads executing "
                    "these instructions concurrently can silently lose an update."
                ),
                "qas": [
                    ("Two threads increment a shared global counter using 'counter++;' with no locking, where "
                     "this compiles to three machine instructions: LOAD counter into a register, ADD 1 to the "
                     "register, STORE the register back to counter. Starting from counter == 5, give a specific "
                     "interleaving of these six instructions that causes the final value to be 6 instead of the "
                     "expected 7, and explain why.",
                     "One race interleaving: Thread A: LOAD counter (reg_A = 5). Thread B: LOAD counter "
                     "(reg_B = 5) — this happens before Thread A has written anything back, so B also reads the "
                     "old value 5. Thread A: ADD 1 (reg_A = 6). Thread A: STORE reg_A to counter (counter "
                     "becomes 6). Thread B: ADD 1 (reg_B = 6, using its own stale copy read earlier). Thread B: "
                     "STORE reg_B to counter (counter becomes 6 again, silently overwriting Thread A's update). "
                     "Final value: counter == 6, not 7 — one increment is lost. This happens because 'counter++' "
                     "is not atomic: both threads read the same starting value into their own private registers "
                     "before either has written a result back, so their increments are each computed independently "
                     "from the same stale base value, and whichever thread stores last simply overwrites the "
                     "other's result instead of building on it. The fix is to make the read-modify-write sequence "
                     "atomic with respect to other threads — e.g. wrapping it in a mutex, or using a hardware "
                     "atomic instruction such as x86's LOCK XADD or C11's atomic_fetch_add."),
                ],
            },
            {
                "id": "deadlock_scenario_analysis",
                "context": (
                    "A cycle in the resource-allocation graph is necessary but not always sufficient for "
                    "deadlock: it is both necessary and sufficient only when every resource type in the cycle "
                    "has a single instance. With multiple instances, a cycle may not force deadlock if a "
                    "request can still be satisfied from an available instance."
                ),
                "qas": [
                    ("Process P1 holds Resource R1 and requests R2. Process P2 holds R2 and requests R3. "
                     "Process P3 holds R3 and requests R1. Each resource type has exactly one instance. "
                     "(a) Is this system deadlocked? (b) If R1 instead had two instances, one held by P1 and the "
                     "other currently available, would P3's request for R1 still contribute to a deadlock?",
                     "(a) The resource-allocation graph has assignment edges R1->P1, R2->P2, R3->P3, and request "
                     "edges P1->R2, P2->R3, P3->R1. Following the edges: P1 -> R2 -> P2 -> R3 -> P3 -> R1 -> P1 "
                     "closes a cycle. Since every resource type in this cycle has only a single instance, a cycle "
                     "here is both necessary and sufficient for deadlock — so yes, the system is deadlocked; none "
                     "of P1, P2, P3 can ever proceed. (b) If R1 has two instances and one is available, then when "
                     "P3 requests R1, that request can be granted immediately from the available instance — P3 "
                     "does not need to wait, so its request edge to R1 never actually blocks (it is effectively "
                     "converted straight to an assignment edge). With multiple instances of a resource, a cycle in "
                     "the graph is necessary but not sufficient for deadlock — you must also check whether the "
                     "requests along the cycle can be satisfied from available instances, not just whether a "
                     "cycle exists textually."),
                ],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Real-Time Scheduling",
        "topics": [
            {
                "id": "realtime_scheduling",
                "context": (
                    "Real-time scheduling algorithms like Rate Monotonic Scheduling (RMS) and Earliest "
                    "Deadline First (EDF) schedule periodic tasks to guarantee that each task's deadline "
                    "(usually equal to its period) is met. RMS is a fixed-priority algorithm; EDF is a "
                    "dynamic-priority algorithm and is provably optimal among such algorithms."
                ),
                "qas": [
                    ("What is Rate Monotonic Scheduling (RMS), and what is its utilization-bound schedulability test?",
                     "RMS is a fixed-priority preemptive scheduling algorithm for periodic real-time tasks: each "
                     "task is assigned a static priority inversely proportional to its period — the shorter a "
                     "task's period, the higher its priority, since it must run more frequently. A sufficient "
                     "(but not necessary) schedulability test is the Liu & Layland utilization bound: for n "
                     "periodic tasks with worst-case execution time C_i and period T_i, if total utilization "
                     "U = sum(C_i / T_i) is less than or equal to n(2^(1/n) - 1), all deadlines are guaranteed to "
                     "be met under RMS. As n -> infinity this bound approaches ln(2) ≈ 0.693, meaning RMS can "
                     "only guarantee schedulability up to roughly 69% CPU utilization, even though total demand "
                     "could technically be less than 100% — RMS can still miss deadlines above this bound because "
                     "of how fixed priorities interact."),
                    ("What is Earliest Deadline First (EDF) scheduling, and how does its schedulability bound "
                     "compare to RMS's?",
                     "EDF is a dynamic-priority preemptive scheduling algorithm: at every scheduling decision, "
                     "the CPU is given to whichever ready task has the nearest absolute deadline, so a task's "
                     "effective priority changes over time as its deadline approaches. Unlike RMS's fixed ~69% "
                     "(ln 2) utilization bound, EDF is provably optimal among all dynamic-priority scheduling "
                     "algorithms and guarantees all deadlines are met for any task set as long as total "
                     "utilization U = sum(C_i/T_i) <= 1 — i.e. up to 100% CPU utilization, strictly better "
                     "schedulability than RMS — at the cost of higher run-time scheduling overhead, since "
                     "priorities must be recomputed dynamically rather than fixed once."),
                    ("Given two periodic tasks — Task 1 with C1=1, T1=3, and Task 2 with C2=1, T2=5 — is this "
                     "task set schedulable under RMS according to the Liu & Layland bound?",
                     "Total utilization U = C1/T1 + C2/T2 = 1/3 + 1/5 = 5/15 + 3/15 = 8/15 ≈ 0.533. The Liu & "
                     "Layland bound for n=2 tasks is n(2^(1/n) - 1) = 2(2^0.5 - 1) = 2(1.41421 - 1) ≈ "
                     "2 x 0.41421 ≈ 0.828. Since U ≈ 0.533 <= 0.828, the sufficient condition is satisfied, so "
                     "yes — this task set is guaranteed schedulable under RMS."),
                ],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Applied Virtual Memory and Security",
        "topics": [
            {
                "id": "cow_fork",
                "context": (
                    "When a process calls fork(), copying the entire parent address space immediately would "
                    "be wasteful in the common case where the child soon calls exec() and discards it. "
                    "Copy-on-Write (COW) defers the actual copy until a write actually occurs."
                ),
                "qas": [
                    ("Why is Copy-on-Write (COW) used for fork() instead of immediately duplicating the parent's "
                     "entire address space, and how does it work at the page-table level?",
                     "Immediately copying the parent's full address space would be wasteful in the very common "
                     "case where the child calls exec() right afterward and discards the copy entirely (the "
                     "typical fork()+exec() pattern for launching a new program), or where parent and child "
                     "mostly just read shared data. Copy-on-Write avoids this: instead of physically duplicating "
                     "pages at fork() time, the child's page table is set up to point to the SAME physical "
                     "frames as the parent, and both processes' page table entries for these shared pages are "
                     "marked read-only, even for pages that were originally writable. If either process later "
                     "writes to one of these shared pages, the write triggers a protection-fault trap. The "
                     "kernel's fault handler recognizes this is a COW page, allocates a fresh physical frame, "
                     "copies the page's contents into it, updates the faulting process's page table entry to "
                     "point to this new private frame (now marked writable), and lets the write proceed. Actual "
                     "physical copying only happens for pages that are actually modified, and only at the moment "
                     "they're modified — read-only or never-modified pages are never duplicated, drastically "
                     "reducing the cost of fork() in the common case."),
                ],
            },
            {
                "id": "buffer_overflow_security",
                "context": (
                    "A function's stack frame holds local variables followed by the saved return address. "
                    "Writing past an unbounded local buffer can overwrite that return address, letting an "
                    "attacker hijack control flow when the function returns. ASLR and the NX bit are standard "
                    "OS/hardware defenses."
                ),
                "qas": [
                    ("Explain, at a mechanical level, how a classic stack-based buffer overflow can hijack a "
                     "program's control flow, and name two OS/hardware-level defenses against it.",
                     "A function's stack frame typically contains local variables — including, potentially, a "
                     "fixed-size buffer such as char buf[64] — followed by saved registers and the return "
                     "address the CPU jumps back to when the function returns. If the program writes into that "
                     "buffer without bounds-checking (e.g. using an unsafe function like strcpy on "
                     "attacker-controlled input), an attacker can supply input longer than the buffer, and the "
                     "excess bytes overflow into adjacent stack memory — including the saved return address. By "
                     "crafting the overflow so the bytes landing exactly where the return address is stored spell "
                     "out an address of the attacker's choosing (e.g. injected shellcode), the function's return "
                     "instruction jumps to that address instead of the legitimate caller, hijacking control flow. "
                     "Two standard defenses: (1) Address Space Layout Randomization (ASLR) — the OS randomizes "
                     "the base addresses of the stack, heap, and shared libraries on each run, so an attacker "
                     "cannot reliably predict the address to jump to. (2) The NX bit (Data Execution Prevention) "
                     "— a hardware page-table permission bit marking stack (and other data) memory as "
                     "non-executable, so even if execution is redirected into injected shellcode on the stack, "
                     "the CPU refuses to execute instructions fetched from that page and faults instead. (Stack "
                     "canaries — a random value placed between the buffer and the return address, checked before "
                     "the function returns — are a common compiler-level third defense.)"),
                ],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Scalable Multicore Synchronization",
        "topics": [
            {
                "id": "rcu",
                "context": (
                    "RCU (Read-Copy-Update) is a synchronization technique that lets readers access a "
                    "shared data structure with no locking at all — reads are essentially free and scale "
                    "perfectly across cores — by having writers publish updates atomically via a single "
                    "pointer swap and deferring reclamation of the old version until a 'grace period' "
                    "guarantees no reader can still be using it."
                ),
                "qas": [
                    ("What is RCU (Read-Copy-Update), and why do readers need no lock at all?",
                     "RCU lets readers traverse a shared data structure by simply dereferencing pointers, "
                     "bracketed only by rcu_read_lock()/rcu_read_unlock() — which, in the common kernel "
                     "implementation, compile down to nearly nothing (e.g. just disabling preemption) rather "
                     "than acquiring any actual lock or performing any atomic read-modify-write instruction. "
                     "This works because writers never mutate a structure readers might be traversing IN "
                     "PLACE: a writer builds an entire new (or copied-and-modified) version of the data off to "
                     "the side, then publishes it with a single atomic pointer write that redirects future "
                     "readers to the new version. Since any in-flight reader either read the pointer before "
                     "the swap (and sees the complete old version) or after (and sees the complete new "
                     "version), there is never a half-updated state a reader could observe, so readers need no "
                     "synchronization with the writer at all — this is what makes RCU reads scale with zero "
                     "cache-line contention across any number of cores."),
                    ("How does a writer safely update an RCU-protected data structure, and what is a 'grace period'?",
                     "The writer: (1) allocates/builds a new version of the data (or a modified copy of the "
                     "node being changed) off to the side, without touching the live structure readers can "
                     "see; (2) publishes the change with a single atomic pointer update that redirects future "
                     "readers to the new version — this is the moment the update becomes visible; (3) does NOT "
                     "immediately free the old version, because readers that began traversing before the "
                     "publish may still hold a reference to it and could still be using it. The writer must "
                     "wait for a grace period — an interval during which every core that could possibly have "
                     "been inside a pre-existing RCU read-side critical section is guaranteed to have exited "
                     "it (typically detected by observing that every core has passed through at least one "
                     "'quiescent state', such as a context switch, since the update). Only after the grace "
                     "period elapses is it safe to actually free the old version — the kernel API "
                     "synchronize_rcu() blocks until this happens, while call_rcu() schedules the free "
                     "asynchronously for after the grace period."),
                    ("Why is RCU well suited to read-mostly data structures (e.g. routing tables, syscall "
                     "dispatch tables) but poorly suited to write-heavy ones?",
                     "RCU makes the read side essentially free, but it shifts real cost onto the write side: "
                     "a writer must wait out a grace period before reclaiming old versions, potentially "
                     "keeping multiple stale versions of the structure alive in memory simultaneously, and "
                     "concurrent writers still need their own separate mutual exclusion between each other "
                     "(e.g. an ordinary lock) since RCU only protects readers against a writer, not writers "
                     "against each other. For a read-mostly workload, eliminating almost all reader overhead "
                     "is an enormous win that dwarfs the occasional write's extra cost. For a write-heavy "
                     "workload, the benefit to readers is small relative to how often it occurs, while every "
                     "single write now pays the grace-period-wait and versioning overhead — so RCU brings "
                     "little benefit and real extra cost."),
                ],
            },
            {
                "id": "spinlock_implementations",
                "context": (
                    "Naive spinlocks based on a single atomic test-and-set instruction scale poorly under "
                    "contention because every retry attempt generates cache-coherence traffic. Progressively "
                    "better designs — test-and-test-and-set, ticket locks, and MCS (queue-based) locks — "
                    "reduce this coherence traffic, with MCS locks giving each unlock O(1) coherence cost "
                    "regardless of how many threads are waiting."
                ),
                "qas": [
                    ("Why does a naive test-and-set spinlock scale poorly on a multicore system, and how does "
                     "test-and-test-and-set (TTAS) improve on it?",
                     "A test-and-set spinlock repeatedly executes an atomic test-and-set (or CAS) instruction "
                     "trying to flip the lock from free to held; if it's already held, the thread loops and "
                     "retries immediately. The problem is that the atomic instruction itself requires "
                     "exclusive (Modified-state) ownership of the lock's cache line, so EVERY retry by EVERY "
                     "waiting core — even the ones that will fail — forces that cache line to bounce between "
                     "cores, generating heavy coherence traffic and bus contention, and this constant "
                     "cache-line stealing can even slow down the core that actually HOLDS the lock, delaying "
                     "its release. Test-and-test-and-set improves this by first spinning on a plain, "
                     "non-atomic READ of the lock in a loop — this can be served entirely from each waiter's "
                     "local cache in the Shared coherence state, generating no bus traffic at all while the "
                     "lock is held — and only attempting the actual atomic test-and-set once that read "
                     "observes the lock appears free. This eliminates most of the wasted coherence traffic "
                     "during the waiting phase, though a burst of contention still occurs at the instant the "
                     "lock is released, since all waiters see it become free at roughly the same time and race "
                     "to grab it."),
                    ("What is a ticket lock, what problem does it solve compared to test-and-set, and what "
                     "scalability problem does it still have?",
                     "A ticket lock has each thread wanting the lock atomically fetch-and-increment a shared "
                     "'next ticket' counter, obtaining a unique ticket number, and then spin waiting for a "
                     "separate 'now serving' counter to reach its ticket number; unlocking simply increments "
                     "'now serving'. This guarantees strict FIFO fairness — every thread is served in the "
                     "exact order it arrived, with no possibility of starvation — unlike test-and-set/TTAS, "
                     "where an unlucky thread can in principle keep losing the race to newly-arriving threads "
                     "indefinitely. However, a ticket lock still has a scalability problem: every waiting "
                     "thread spins on the SAME shared 'now serving' cache line, so every single unlock (which "
                     "modifies that shared variable) invalidates it in every waiter's cache, forcing all of "
                     "them to re-fetch it — even though only one waiter's ticket actually matches and can "
                     "proceed. This means unlock cost grows with the number of waiters (O(n) coherence "
                     "traffic per unlock)."),
                    ("How does an MCS (queue-based) lock solve the ticket lock's remaining scalability problem?",
                     "Instead of having every waiter spin on one shared variable, each thread that wants an "
                     "MCS lock constructs its own local queue node and links it onto the end of a queue (via "
                     "an atomic CAS on a tail pointer), then spins on a flag INSIDE ITS OWN NODE — not on any "
                     "shared global variable. When the lock holder releases, it does not touch a shared "
                     "variable that everyone is watching; instead it writes directly into the NEXT waiting "
                     "thread's own node, setting that one specific thread's flag. Only that single node's "
                     "cache line is invalidated — every other waiting thread continues spinning purely "
                     "locally, generating zero coherence traffic from a release that isn't theirs. This makes "
                     "unlock cost O(1) regardless of how many threads are waiting, instead of the O(n) cost of "
                     "a ticket lock, which is why the Linux kernel's default spinlock implementation (the "
                     "'qspinlock') is essentially an MCS-lock-derived design, adopted specifically to fix this "
                     "exact scalability problem on many-core systems."),
                ],
            },
            {
                "id": "false_sharing",
                "context": (
                    "Cache coherence protocols operate at the granularity of a whole cache line (typically "
                    "64 bytes), not individual variables. False sharing occurs when two logically unrelated "
                    "variables accessed by different cores happen to land on the same cache line, causing "
                    "coherence traffic between cores that never actually touch each other's data."
                ),
                "qas": [
                    ("What is false sharing, and why does it hurt performance even though the two threads "
                     "involved never touch the same logical variable?",
                     "False sharing happens when two independent variables — say, variable X written "
                     "frequently by Core A and variable Y written frequently by Core B — happen to be laid "
                     "out by the compiler/allocator on the SAME physical cache line. Cache coherence "
                     "protocols like MESI track ownership at cache-LINE granularity, not per-byte or "
                     "per-variable granularity, because that's the unit hardware actually moves between "
                     "caches. So every time Core A writes X, it must invalidate that entire cache line in "
                     "Core B's cache — including the copy of Y that Core B actually cares about — forcing "
                     "Core B to re-fetch the line on its next access even though Y itself never changed. The "
                     "same happens in reverse when Core B writes Y. The cache line ends up ping-ponging "
                     "between the two cores' caches continuously, generating real coherence traffic and stall "
                     "time, purely as an artifact of memory layout — there is no actual logical dependency or "
                     "race between X and Y at all."),
                    ("How is false sharing typically fixed?",
                     "By padding or aligning independently-accessed 'hot' variables so each one lands on its "
                     "own separate cache line, even if the variable itself is much smaller than a cache line "
                     "— e.g. padding each entry of a per-CPU counters array out to 64 bytes even though each "
                     "individual counter is only 4 or 8 bytes, or using a compiler alignment attribute (such "
                     "as C++11's alignas(64)) on frequently-contended fields, so that cores updating "
                     "different logical variables never contend over the same physical cache line."),
                ],
            },
            {
                "id": "futexes",
                "context": (
                    "A futex ('fast userspace mutex') is the Linux primitive underlying most userspace lock "
                    "implementations. It separates the fast, uncontended path — handled entirely in "
                    "userspace with a single atomic instruction — from the slow, contended path, which is "
                    "the only case that requires an actual syscall to block or wake a thread in the kernel."
                ),
                "qas": [
                    ("What problem does a futex solve, and why would making a syscall for every single "
                     "lock/unlock operation be wasteful?",
                     "Most locks, most of the time, are uncontended — a thread acquires them, does a short "
                     "critical section, and releases them with no other thread ever trying to acquire the "
                     "same lock at the same instant. A syscall costs real time (a context switch into the "
                     "kernel and back, on the order of a couple hundred nanoseconds or more), and if every "
                     "single lock/unlock always made a syscall, that fixed cost would be paid constantly even "
                     "though blocking in the kernel is almost never actually necessary. A futex solves this by "
                     "letting the fast (uncontended) path run entirely in userspace using nothing but an "
                     "atomic CPU instruction, and only falling into the kernel — via an actual futex() syscall "
                     "— in the rare case a thread genuinely needs to block waiting for another thread, or "
                     "needs to wake one that's blocked."),
                    ("Walk through the fast path (uncontended) and slow path (contended) of a futex-based "
                     "mutex lock and unlock.",
                     "The lock is just a plain integer in userspace shared memory (e.g. 0 = unlocked, 1 = "
                     "locked with no waiters, 2 = locked with waiters). Lock, fast path: the thread attempts "
                     "an atomic compare-and-swap of the integer from 0 to 1 entirely in userspace; if it "
                     "succeeds, the thread now holds the lock and no syscall was ever made. Lock, slow path: "
                     "if the CAS fails because the value wasn't 0 (someone else holds it), the thread marks "
                     "the value as 'locked with waiters' and makes a futex(FUTEX_WAIT, ...) syscall, which "
                     "puts it to sleep in the kernel, associated with that specific memory address, until "
                     "woken. Unlock, fast path: if the lock's value indicates there were no waiters, unlocking "
                     "is just an atomic write of 0 back to the integer in userspace — again, no syscall. "
                     "Unlock, slow path: if the value indicates waiters exist, the unlocking thread must make "
                     "a futex(FUTEX_WAKE, ...) syscall to explicitly wake (at least) one sleeping waiter in "
                     "the kernel. The net effect: the overwhelmingly common uncontended case costs a few "
                     "nanoseconds for one atomic instruction, and the syscall/blocking cost is paid only in "
                     "the genuinely rare case where a thread must actually wait."),
                ],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "NUMA and Memory Locality",
        "topics": [
            {
                "id": "numa_effects",
                "context": (
                    "On multi-socket systems, physical memory is divided into NUMA nodes, each directly "
                    "attached to one socket. Any core can access any node's memory, but a remote access "
                    "(crossing to another socket's node over an inter-socket interconnect) has higher "
                    "latency and lower bandwidth than a local access, which shapes how a NUMA-aware OS "
                    "schedules threads and allocates memory."
                ),
                "qas": [
                    ("What is NUMA, and why is a 'remote' memory access slower than a 'local' one?",
                     "NUMA (Non-Uniform Memory Access) describes multi-socket systems where physical memory "
                     "is partitioned into nodes, each directly attached to one socket's memory controller. "
                     "Any core in the system can address any node's memory — the address space is unified — "
                     "but a core accessing its OWN node's (local) memory goes directly through its local "
                     "memory controller, while accessing another socket's (remote) node requires the request "
                     "to travel across an inter-socket interconnect (e.g. Intel's UPI or AMD's Infinity "
                     "Fabric) to reach the remote controller and then travel back with the result. This extra "
                     "hop adds real latency and consumes shared interconnect bandwidth, so remote accesses are "
                     "measurably slower — often by a significant margin — than local ones, even though both "
                     "are architecturally 'just memory access' from the program's point of view."),
                    ("How does NUMA affect scheduling and memory-allocation decisions in a NUMA-aware OS?",
                     "A NUMA-aware scheduler prefers to keep a thread running on cores within the same NUMA "
                     "node where its memory already lives, rather than freely migrating it to any idle core "
                     "system-wide — migrating it to a remote node's cores would leave its working data "
                     "'behind' on the far node, making every subsequent memory access pay the remote-access "
                     "latency penalty. Symmetrically, the memory allocator tries to place a thread's memory on "
                     "the same node it is currently running on. When load-balancing genuinely requires moving "
                     "a thread to a different node's cores, the OS faces a real trade-off between the "
                     "load-balancing benefit and the locality cost of leaving its memory far away (or paying "
                     "to actually migrate the physical pages, which is itself expensive and requires a TLB "
                     "shootdown). Modern kernels (e.g. Linux's NUMA balancing / AutoNUMA) periodically sample "
                     "where a task's memory actually resides versus where it's running, and either migrate "
                     "the task back toward its memory or migrate the memory toward the task, converging "
                     "toward NUMA-local execution over time."),
                    ("What is the 'first-touch' memory allocation policy, and why is it used on NUMA systems?",
                     "Under first-touch, a virtual page is not actually backed by a physical frame until the "
                     "first time it is accessed ('touched'); at that moment, the kernel allocates the "
                     "physical frame from the NUMA node local to whichever core performed that first access. "
                     "This is used because it's a simple, effective heuristic for locality: the thread that "
                     "first uses a page is very often the thread that will keep using it, so backing the page "
                     "with local memory on that thread's node maximizes the odds that its future accesses "
                     "stay local rather than remote, without the OS needing to predict access patterns in "
                     "advance."),
                ],
            },
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════
# Extra glossary terms surfaced by this advanced tier.
# ══════════════════════════════════════════════════════════════════════
GLOSSARY = [
    ("Memory barrier (fence)", "A CPU instruction (e.g. MFENCE, smp_mb()) that forces preceding memory operations to become globally visible before any subsequent memory operation is allowed to execute, preventing hardware/compiler reordering across it."),
    ("Store buffer", "A small per-core hardware structure that holds a core's own pending writes before they are committed to shared memory, allowing StoreLoad reordering where a core's later load can execute before its own earlier store becomes globally visible."),
    ("Priority Inheritance Protocol", "A real-time scheduling technique where a low-priority task holding a resource temporarily inherits the priority of the highest-priority task blocked waiting for that resource, bounding the delay caused by priority inversion."),
    ("Copy-on-Write (COW)", "A memory-management optimization where two processes (e.g. after fork()) share the same physical pages read-only until one attempts to write, at which point a private copy of just that page is made."),
    ("Address Space Layout Randomization (ASLR)", "An OS security defense that randomizes the base addresses of a process's stack, heap, and shared libraries on each execution, making it harder for an attacker to predict addresses for exploits like buffer overflows."),
    ("Earliest Deadline First (EDF)", "A dynamic-priority real-time scheduling algorithm that always runs the ready task with the nearest absolute deadline; it is provably optimal and can guarantee schedulability up to 100% CPU utilization."),
    ("TLB shootdown", "The process of sending an Inter-Processor Interrupt to every core that might have a stale cached translation for a changed page table entry, forcing each to flush it, before the underlying physical frame can safely be reused."),
    ("Inter-Processor Interrupt (IPI)", "A hardware interrupt one CPU core sends directly to another (via the Local APIC on x86, or an SGI via the GIC on ARM) to force it to run a specific handler immediately, used e.g. to coordinate TLB shootdowns across cores."),
    ("RCU (Read-Copy-Update)", "A synchronization technique letting readers access shared data with no locking at all, by having writers publish updates via an atomic pointer swap and deferring reclamation of old versions until a grace period guarantees no reader can still be using them."),
    ("MCS lock (queued spinlock)", "A queue-based spinlock where each waiting thread spins on a flag in its own local node rather than a shared variable, so an unlock only invalidates one specific waiter's cache line (O(1) cost) instead of broadcasting to every waiter."),
    ("False sharing", "A performance problem where two logically unrelated variables accessed by different cores happen to share the same cache line, causing coherence invalidation traffic between the cores even though neither actually touches the other's data."),
    ("NUMA (Non-Uniform Memory Access)", "A multi-socket memory architecture where each socket has its own directly-attached memory node, and accessing another socket's (remote) node over the inter-socket interconnect is slower than accessing the local node."),
    ("Futex (fast userspace mutex)", "A Linux kernel primitive that lets an uncontended lock/unlock complete entirely in userspace with a single atomic instruction, only making a syscall to block or wake a thread in the rare case of actual contention."),
]
