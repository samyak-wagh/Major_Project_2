"""
numerical_gen.py — Correctness-guaranteed numerical OS problem generator.
===========================================================================
Every problem instance's answer is computed by actually simulating the
algorithm (scheduling, page replacement, disk scheduling, Banker's safety
check, address translation) — never guessed or hand-typed — so correctness
is guaranteed by construction rather than by manual verification at scale.

Five categories, matching the standard GATE/college-exam OS syllabus:
  1. CPU scheduling      — FCFS, SJF, SRTF, Round Robin, Priority
  2. Page replacement    — FIFO, LRU, Optimal
  3. Disk scheduling     — FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK
  4. Banker's algorithm  — safety check + resource-request evaluation
  5. Memory addressing   — paging translation + contiguous-allocation fit

Each gen_* function returns (context, question, answer) — the same shape
consumed by build_dataset.py to wrap into the chat-messages schema.
"""

import random
from collections import deque

PIDS = list("123456789")


# ══════════════════════════════════════════════════════════════════════
# 1. CPU SCHEDULING
# ══════════════════════════════════════════════════════════════════════

CPU_SCHED_CONTEXT = {
    "FCFS": (
        "First-Come, First-Served (FCFS) CPU scheduling allocates the CPU to "
        "processes strictly in order of arrival time, using a FIFO ready "
        "queue. It is non-preemptive: once a process starts running, it runs "
        "to completion. Ties in arrival time are broken by process ID order. "
        "Completion Time (CT) is when a process finishes; Turnaround Time "
        "(TAT) = CT − Arrival Time; Waiting Time (WT) = TAT − Burst Time."
    ),
    "SJF": (
        "Shortest Job First (SJF, non-preemptive) scheduling selects, among "
        "all processes that have arrived and are ready, the one with the "
        "smallest CPU burst time to run next; once started it runs to "
        "completion. Ties are broken by earliest arrival time, then by "
        "process ID. Completion Time (CT) is when a process finishes; "
        "Turnaround Time (TAT) = CT − Arrival Time; Waiting Time (WT) = TAT "
        "− Burst Time."
    ),
    "SRTF": (
        "Shortest Remaining Time First (SRTF) is the preemptive version of "
        "SJF: at every instant, the CPU runs whichever ready process has the "
        "smallest remaining burst time, preempting the current process if a "
        "process with a shorter remaining time becomes ready. Ties are "
        "broken by earliest arrival time, then process ID. Completion Time "
        "(CT) is when a process finishes; Turnaround Time (TAT) = CT − "
        "Arrival Time; Waiting Time (WT) = TAT − Burst Time."
    ),
    "RR": (
        "Round Robin (RR) scheduling gives each ready process a fixed time "
        "quantum to run. If a process does not finish within its quantum, "
        "it is preempted and placed at the back of the (circular) ready "
        "queue. Newly-arriving processes are added to the back of the queue "
        "before the just-preempted process is re-added. Completion Time "
        "(CT) is when a process finishes; Turnaround Time (TAT) = CT − "
        "Arrival Time; Waiting Time (WT) = TAT − Burst Time."
    ),
    "Priority": (
        "Priority scheduling (non-preemptive) allocates the CPU to the ready "
        "process with the highest priority, where a SMALLER priority number "
        "means HIGHER priority. Once started, a process runs to completion. "
        "Ties are broken by earliest arrival time, then process ID. "
        "Completion Time (CT) is when a process finishes; Turnaround Time "
        "(TAT) = CT − Arrival Time; Waiting Time (WT) = TAT − Burst Time."
    ),
}


def _fmt(x):
    """Format a number, dropping a trailing .0 for integers."""
    if isinstance(x, float) and x == int(x):
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def _gantt_str(timeline):
    parts = []
    for pid, start, end in timeline:
        label = "Idle" if pid == "IDLE" else f"P{pid}"
        parts.append(f"{label}[{start}-{end}]")
    return " -> ".join(parts)


def sim_fcfs(procs):
    order = sorted(procs, key=lambda p: (p["arrival"], p["pid"]))
    time = 0
    timeline = []
    completion = {}
    for p in order:
        start = max(time, p["arrival"])
        if start > time:
            timeline.append(("IDLE", time, start))
        end = start + p["burst"]
        timeline.append((p["pid"], start, end))
        time = end
        completion[p["pid"]] = end
    return timeline, completion


def sim_sjf_np(procs):
    remaining = {p["pid"]: dict(p) for p in procs}
    pending = set(remaining.keys())
    time = 0
    timeline = []
    completion = {}
    while pending:
        avail = [pid for pid in pending if remaining[pid]["arrival"] <= time]
        if not avail:
            nxt = min(remaining[pid]["arrival"] for pid in pending)
            timeline.append(("IDLE", time, nxt))
            time = nxt
            continue
        pid = min(avail, key=lambda x: (remaining[x]["burst"], remaining[x]["arrival"], x))
        start = time
        end = start + remaining[pid]["burst"]
        timeline.append((pid, start, end))
        completion[pid] = end
        time = end
        pending.remove(pid)
    return timeline, completion


def sim_priority_np(procs):
    remaining = {p["pid"]: dict(p) for p in procs}
    pending = set(remaining.keys())
    time = 0
    timeline = []
    completion = {}
    while pending:
        avail = [pid for pid in pending if remaining[pid]["arrival"] <= time]
        if not avail:
            nxt = min(remaining[pid]["arrival"] for pid in pending)
            timeline.append(("IDLE", time, nxt))
            time = nxt
            continue
        pid = min(avail, key=lambda x: (remaining[x]["priority"], remaining[x]["arrival"], x))
        start = time
        end = start + remaining[pid]["burst"]
        timeline.append((pid, start, end))
        completion[pid] = end
        time = end
        pending.remove(pid)
    return timeline, completion


def sim_srtf(procs):
    state = {p["pid"]: {"arrival": p["arrival"], "burst": p["burst"], "remaining": p["burst"]} for p in procs}
    n = len(state)
    completed = 0
    completion = {}
    timeline = []
    current = None
    seg_start = 0
    time = 0
    max_time = sum(p["burst"] for p in procs) + max(p["arrival"] for p in procs) + 5
    while completed < n and time <= max_time:
        avail = [pid for pid, s in state.items() if s["arrival"] <= time and s["remaining"] > 0]
        pid = min(avail, key=lambda x: (state[x]["remaining"], state[x]["arrival"], x)) if avail else "IDLE"
        if pid != current:
            if current is not None:
                timeline.append((current, seg_start, time))
            current = pid
            seg_start = time
        if pid != "IDLE":
            state[pid]["remaining"] -= 1
            if state[pid]["remaining"] == 0:
                completion[pid] = time + 1
                completed += 1
        time += 1
    if current is not None:
        timeline.append((current, seg_start, time))
    # merge consecutive identical segments (can happen at boundaries)
    merged = []
    for seg in timeline:
        if merged and merged[-1][0] == seg[0] and merged[-1][2] == seg[1]:
            merged[-1] = (merged[-1][0], merged[-1][1], seg[2])
        else:
            merged.append(seg)
    return merged, completion


def sim_rr(procs, quantum):
    state = {p["pid"]: {"arrival": p["arrival"], "burst": p["burst"], "remaining": p["burst"]} for p in procs}
    order_by_arrival = sorted(state.keys(), key=lambda x: (state[x]["arrival"], x))
    n = len(state)
    time = 0
    completed = 0
    completion = {}
    timeline = []
    q = deque()
    idx = 0

    def admit_up_to(t):
        nonlocal idx
        while idx < len(order_by_arrival) and state[order_by_arrival[idx]]["arrival"] <= t:
            q.append(order_by_arrival[idx])
            idx += 1

    admit_up_to(time)
    if not q:
        time = state[order_by_arrival[0]]["arrival"]
        admit_up_to(time)

    while completed < n:
        if not q:
            nxt = state[order_by_arrival[idx]]["arrival"]
            timeline.append(("IDLE", time, nxt))
            time = nxt
            admit_up_to(time)
            continue
        pid = q.popleft()
        run = min(quantum, state[pid]["remaining"])
        start, end = time, time + run
        timeline.append((pid, start, end))
        state[pid]["remaining"] -= run
        time = end
        admit_up_to(time)
        if state[pid]["remaining"] > 0:
            q.append(pid)
        else:
            completion[pid] = time
            completed += 1
    return timeline, completion


def _metrics_table(procs, completion):
    rows = []
    for p in sorted(procs, key=lambda x: x["pid"]):
        ct = completion[p["pid"]]
        tat = ct - p["arrival"]
        wt = tat - p["burst"]
        rows.append({"pid": p["pid"], "arrival": p["arrival"], "burst": p["burst"], "ct": ct, "tat": tat, "wt": wt})
    return rows


def gen_cpu_scheduling(rng: random.Random):
    algo = rng.choice(["FCFS", "SJF", "SRTF", "RR", "Priority"])
    n = rng.randint(3, 5)
    pids = PIDS[:n]
    procs = []
    for pid in pids:
        procs.append({
            "pid": pid,
            "arrival": rng.randint(0, 2 * n),
            "burst": rng.randint(2, 12),
            "priority": rng.randint(1, 9),
        })
    # avoid the degenerate case where every arrival time is identical AND
    # every burst is identical (uninteresting problem)
    if len({p["arrival"] for p in procs}) == 1 and len({p["burst"] for p in procs}) == 1:
        procs[0]["burst"] += 3

    quantum = rng.choice([2, 3, 4])

    if algo == "FCFS":
        timeline, completion = sim_fcfs(procs)
    elif algo == "SJF":
        timeline, completion = sim_sjf_np(procs)
    elif algo == "SRTF":
        timeline, completion = sim_srtf(procs)
    elif algo == "RR":
        timeline, completion = sim_rr(procs, quantum)
    else:
        timeline, completion = sim_priority_np(procs)

    rows = _metrics_table(procs, completion)
    avg_tat = sum(r["tat"] for r in rows) / n
    avg_wt = sum(r["wt"] for r in rows) / n

    # ── Question ──────────────────────────────────────────────────────
    header = "Process | Arrival Time | Burst Time" + (" | Priority" if algo == "Priority" else "")
    lines = [header, "-" * len(header)]
    for p in procs:
        line = f"P{p['pid']}      | {p['arrival']}            | {p['burst']}"
        if algo == "Priority":
            line += f"          | {p['priority']}"
        lines.append(line)
    table_text = "\n".join(lines)

    algo_label = {
        "FCFS": "First-Come, First-Served (FCFS)",
        "SJF": "Shortest Job First (SJF, non-preemptive)",
        "SRTF": "Shortest Remaining Time First (SRTF, preemptive)",
        "RR": f"Round Robin (time quantum = {quantum})",
        "Priority": "Priority (non-preemptive, smaller number = higher priority)",
    }[algo]

    question = (
        f"Consider the following processes with their arrival and burst times "
        f"(all times in milliseconds):\n\n{table_text}\n\n"
        f"Using {algo_label} scheduling, draw the Gantt chart and compute the "
        f"Completion Time, Turnaround Time, and Waiting Time for each process. "
        f"Also compute the average waiting time and average turnaround time."
    )

    # ── Answer ───────────────────────────────────────────────────────
    ans_lines = [f"Gantt Chart:\n{_gantt_str(timeline)}", ""]
    ans_lines.append("Process | CT | TAT (=CT-AT) | WT (=TAT-BT)")
    for r in rows:
        ans_lines.append(f"P{r['pid']}      | {r['ct']}  | {r['tat']}            | {r['wt']}")
    ans_lines.append("")
    ans_lines.append(f"Average Turnaround Time = ({' + '.join(_fmt(r['tat']) for r in rows)}) / {n} = {_fmt(avg_tat)} ms")
    ans_lines.append(f"Average Waiting Time = ({' + '.join(_fmt(r['wt']) for r in rows)}) / {n} = {_fmt(avg_wt)} ms")
    answer = "\n".join(ans_lines)

    return CPU_SCHED_CONTEXT[algo], question, answer


# ══════════════════════════════════════════════════════════════════════
# 2. PAGE REPLACEMENT
# ══════════════════════════════════════════════════════════════════════

PAGE_REPL_CONTEXT = {
    "FIFO": (
        "FIFO page replacement evicts the page that has been resident in "
        "memory the longest, regardless of how recently or often it has "
        "been used, when a page fault occurs and all frames are full."
    ),
    "LRU": (
        "LRU (Least Recently Used) page replacement evicts the page that "
        "has not been referenced for the longest time, when a page fault "
        "occurs and all frames are full — it uses the recent past as an "
        "approximation of the near future."
    ),
    "Optimal": (
        "Optimal (MIN/Belady's) page replacement evicts the page that will "
        "not be used again for the longest time in the future (or never "
        "used again), when a page fault occurs and all frames are full. It "
        "gives the theoretical minimum number of page faults but requires "
        "advance knowledge of the full reference string."
    ),
}


def sim_fifo(ref, frames):
    mem = []
    faults = 0
    trace = []
    for page in ref:
        if page in mem:
            trace.append((page, list(mem), False))
        else:
            faults += 1
            if len(mem) < frames:
                mem.append(page)
            else:
                mem.pop(0)
                mem.append(page)
            trace.append((page, list(mem), True))
    return trace, faults


def sim_lru(ref, frames):
    mem = []
    faults = 0
    trace = []
    for page in ref:
        if page in mem:
            mem.remove(page)
            mem.append(page)
            trace.append((page, list(mem), False))
        else:
            faults += 1
            if len(mem) < frames:
                mem.append(page)
            else:
                mem.pop(0)
                mem.append(page)
            trace.append((page, list(mem), True))
    return trace, faults


def sim_optimal(ref, frames):
    mem = []
    faults = 0
    trace = []
    n = len(ref)
    for i, page in enumerate(ref):
        if page in mem:
            trace.append((page, list(mem), False))
            continue
        faults += 1
        if len(mem) < frames:
            mem.append(page)
        else:
            future = ref[i + 1:]
            victim = None
            farthest = -1
            for m in mem:
                if m not in future:
                    victim = m
                    break
                idx = future.index(m)
                if idx > farthest:
                    farthest = idx
                    victim = m
            mem.remove(victim)
            mem.append(page)
        trace.append((page, list(mem), True))
    return trace, faults


def gen_page_replacement(rng: random.Random):
    algo = rng.choice(["FIFO", "LRU", "Optimal"])
    frames = rng.randint(3, 4)
    length = rng.randint(10, 14)
    ref = [rng.randint(0, 5) for _ in range(length)]

    if algo == "FIFO":
        trace, faults = sim_fifo(ref, frames)
    elif algo == "LRU":
        trace, faults = sim_lru(ref, frames)
    else:
        trace, faults = sim_optimal(ref, frames)

    hits = length - faults
    ref_str = ", ".join(str(x) for x in ref)

    question = (
        f"A system uses {frames} page frames and the {algo} page "
        f"replacement algorithm. Given the page reference string:\n"
        f"{ref_str}\n\n"
        f"Determine, for each reference, whether it is a page hit or a page "
        f"fault, show the resulting frame contents, and compute the total "
        f"number of page faults and page hits."
    )

    lines = ["Ref | Frame contents after this reference | Hit/Fault"]
    for page, mem_state, is_fault in trace:
        mem_str = ", ".join(str(x) for x in mem_state)
        lines.append(f"{page}   | [{mem_str}]" + " " * max(1, 20 - len(mem_str)) + ("| Fault" if is_fault else "| Hit"))
    trace_text = "\n".join(lines)

    answer = (
        f"{trace_text}\n\n"
        f"Total references = {length}\n"
        f"Total page faults = {faults}\n"
        f"Total page hits = {hits}\n"
        f"Page fault rate = {faults}/{length} = {faults/length:.2f}"
    )

    return PAGE_REPL_CONTEXT[algo], question, answer


# ══════════════════════════════════════════════════════════════════════
# 3. DISK SCHEDULING
# ══════════════════════════════════════════════════════════════════════

DISK_SCHED_CONTEXT = {
    "FCFS": (
        "FCFS disk scheduling services pending I/O requests strictly in the "
        "order they arrived, regardless of cylinder position. Total head "
        "movement is the sum of the absolute differences between "
        "consecutively serviced cylinders, starting from the initial head "
        "position."
    ),
    "SSTF": (
        "SSTF (Shortest Seek Time First) disk scheduling always services "
        "the pending request whose cylinder is closest to the disk head's "
        "current position next, minimizing each individual seek."
    ),
    "SCAN": (
        "SCAN disk scheduling moves the head in one direction, servicing "
        "every pending request it passes, continuing all the way to that "
        "end of the disk (cylinder 0 or the maximum cylinder) even if no "
        "request lies exactly there, then reverses direction and services "
        "the remaining requests on the other side, stopping once the last "
        "one is serviced."
    ),
    "CSCAN": (
        "C-SCAN (Circular SCAN) disk scheduling moves the head in one "
        "direction, servicing every pending request it passes, all the way "
        "to that end of the disk; it then jumps immediately to the opposite "
        "end of the disk (this jump counts toward total head movement) and "
        "continues scanning in the SAME original direction, servicing the "
        "remaining requests, ensuring more uniform wait times than SCAN."
    ),
    "LOOK": (
        "LOOK disk scheduling behaves like SCAN, but instead of travelling "
        "all the way to the physical end of the disk, the head reverses "
        "direction as soon as it services the last pending request in the "
        "current direction, avoiding unnecessary movement."
    ),
    "CLOOK": (
        "C-LOOK disk scheduling behaves like C-SCAN, but instead of jumping "
        "to the physical end of the disk, the head jumps directly to the "
        "closest pending request on the other side, avoiding unnecessary "
        "movement to an empty boundary."
    ),
}


def disk_scan_family(requests, head, disk_size, direction, variant):
    """direction: 'up' (increasing cylinder) or 'down' (decreasing cylinder)."""
    reqs = sorted(set(requests))
    order = []
    cur = head
    total = 0

    def go(target, record=True):
        nonlocal cur, total
        total += abs(target - cur)
        cur = target
        if record:
            order.append(target)

    if direction == "up":
        at_or_above = sorted(r for r in reqs if r >= head)
        below = sorted((r for r in reqs if r < head), reverse=True)
        for r in at_or_above:
            go(r)
        if variant == "SCAN":
            go(disk_size - 1, record=False)
            for r in below:
                go(r)
        elif variant == "LOOK":
            for r in below:
                go(r)
        elif variant == "CSCAN":
            go(disk_size - 1, record=False)
            go(0, record=False)
            for r in sorted(r for r in reqs if r < head):
                go(r)
        elif variant == "CLOOK":
            if below:
                go(min(below), record=False)
                for r in sorted(r for r in reqs if r < head):
                    go(r)
    else:  # direction == "down"
        at_or_below = sorted((r for r in reqs if r <= head), reverse=True)
        above = sorted(r for r in reqs if r > head)
        for r in at_or_below:
            go(r)
        if variant == "SCAN":
            go(0, record=False)
            for r in above:
                go(r)
        elif variant == "LOOK":
            for r in above:
                go(r)
        elif variant == "CSCAN":
            go(0, record=False)
            go(disk_size - 1, record=False)
            for r in sorted((r for r in reqs if r > head), reverse=True):
                go(r)
        elif variant == "CLOOK":
            if above:
                go(max(above), record=False)
                for r in sorted((r for r in reqs if r > head), reverse=True):
                    go(r)

    return order, total


def disk_fcfs(requests, head):
    order = list(requests)
    total = 0
    cur = head
    for r in order:
        total += abs(r - cur)
        cur = r
    return order, total


def disk_sstf(requests, head):
    remaining = list(requests)
    order = []
    cur = head
    total = 0
    while remaining:
        nxt = min(remaining, key=lambda r: abs(r - cur))
        total += abs(nxt - cur)
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order, total


def gen_disk_scheduling(rng: random.Random):
    algo = rng.choice(["FCFS", "SSTF", "SCAN", "CSCAN", "LOOK", "CLOOK"])
    disk_size = 200  # cylinders numbered 0..199
    head = rng.randint(20, 180)
    n_req = rng.randint(6, 9)
    requests = rng.sample(range(0, disk_size), n_req)
    direction = rng.choice(["up", "down"])

    if algo == "FCFS":
        order, total = disk_fcfs(requests, head)
    elif algo == "SSTF":
        order, total = disk_sstf(requests, head)
    else:
        order, total = disk_scan_family(requests, head, disk_size, direction, algo)

    algo_label = {
        "FCFS": "FCFS", "SSTF": "SSTF",
        "SCAN": f"SCAN (initial direction: {'increasing' if direction=='up' else 'decreasing'} cylinder numbers)",
        "CSCAN": f"C-SCAN (initial direction: {'increasing' if direction=='up' else 'decreasing'} cylinder numbers)",
        "LOOK": f"LOOK (initial direction: {'increasing' if direction=='up' else 'decreasing'} cylinder numbers)",
        "CLOOK": f"C-LOOK (initial direction: {'increasing' if direction=='up' else 'decreasing'} cylinder numbers)",
    }[algo]

    req_str = ", ".join(str(r) for r in requests)
    question = (
        f"A disk has {disk_size} cylinders (numbered 0 to {disk_size - 1}). "
        f"The disk head is currently at cylinder {head}. The pending "
        f"requests, in the order they arrived, are: {req_str}.\n\n"
        f"Using the {algo_label} disk scheduling algorithm, determine the "
        f"order in which the requests are serviced and compute the total "
        f"head movement (in number of cylinders)."
    )

    order_str = " -> ".join(str(x) for x in order)
    seq_str = f"{head} -> " + order_str
    seq_diffs = []
    cur = head
    for r in order:
        seq_diffs.append(f"|{r}-{cur}|={abs(r-cur)}")
        cur = r
    answer = (
        f"Service order: {seq_str}\n\n"
        f"Head movement per step: {' + '.join(seq_diffs)}\n\n"
        f"Total head movement = {total} cylinders"
    )

    return DISK_SCHED_CONTEXT[algo], question, answer


# ══════════════════════════════════════════════════════════════════════
# 4. BANKER'S ALGORITHM
# ══════════════════════════════════════════════════════════════════════

BANKERS_CONTEXT = (
    "The Banker's algorithm avoids deadlock by only granting a resource "
    "request if the resulting state is 'safe'. A state is safe if there is "
    "some ordering (safe sequence) of all processes such that each "
    "process's remaining Need can be satisfied by the currently Available "
    "resources plus what is freed by processes earlier in the sequence. "
    "Need[i] = Max[i] - Allocation[i]. The safety algorithm repeatedly finds "
    "any unfinished process whose Need is <= Work (initially Available), "
    "'runs' it (adding its Allocation back to Work), and marks it finished, "
    "until no more processes can be added (safe if all finish) or no more "
    "can be found (unsafe if any remain unfinished)."
)


def safety_algorithm(n, m, alloc, need, available):
    work = list(available)
    finish = [False] * n
    sequence = []
    steps = []
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not finish[i] and all(need[i][j] <= work[j] for j in range(m)):
                steps.append((i, list(work), list(alloc[i])))
                work = [work[j] + alloc[i][j] for j in range(m)]
                finish[i] = True
                sequence.append(i)
                changed = True
    safe = all(finish)
    return safe, sequence, steps, work


def gen_bankers(rng: random.Random):
    n = rng.randint(3, 5)
    m = rng.randint(2, 3)
    res_names = ["A", "B", "C"][:m]

    max_matrix = [[rng.randint(3, 9) for _ in range(m)] for _ in range(n)]
    alloc_matrix = [[rng.randint(0, max_matrix[i][j]) for j in range(m)] for i in range(n)]
    need_matrix = [[max_matrix[i][j] - alloc_matrix[i][j] for j in range(m)] for i in range(n)]
    available = [rng.randint(2, 8) for _ in range(m)]

    def fmt_matrix(mat):
        lines = ["Process | " + "  ".join(res_names)]
        for i, row in enumerate(mat):
            lines.append(f"P{i}      | " + "  ".join(str(x) for x in row))
        return "\n".join(lines)

    ask_request = rng.random() < 0.5

    if not ask_request:
        safe, sequence, steps, _ = safety_algorithm(n, m, alloc_matrix, need_matrix, available)
        question = (
            f"A system has {n} processes (P0..P{n-1}) and {m} resource types "
            f"({', '.join(res_names)}). The current Allocation and Max "
            f"matrices, and the Available vector, are:\n\n"
            f"Allocation:\n{fmt_matrix(alloc_matrix)}\n\n"
            f"Max:\n{fmt_matrix(max_matrix)}\n\n"
            f"Available: {', '.join(f'{res_names[j]}={available[j]}' for j in range(m))}\n\n"
            f"Compute the Need matrix, and using the Banker's algorithm, "
            f"determine whether the system is currently in a safe state. If "
            f"it is, give a valid safe sequence."
        )
        ans_lines = [f"Need = Max - Allocation:\n{fmt_matrix(need_matrix)}", ""]
        if safe:
            work = list(available)
            for i, w_before, a_i in steps:
                ans_lines.append(
                    f"Need[P{i}] = {need_matrix[i]} <= Work {w_before} -> "
                    f"P{i} can run; Work becomes {[w_before[j] + a_i[j] for j in range(m)]}"
                )
            seq_str = " -> ".join(f"P{i}" for i in sequence)
            ans_lines.append(f"\nAll processes finish. The system IS in a safe state.")
            ans_lines.append(f"Safe sequence: {seq_str}")
        else:
            ans_lines.append(
                "No unfinished process has Need <= Work at any point, so no "
                "process beyond the ones above can be scheduled. The system "
                "is NOT in a safe state (it may deadlock)."
            )
        answer = "\n".join(ans_lines)
        return BANKERS_CONTEXT, question, answer

    else:
        i = rng.randrange(n)
        if all(v == 0 for v in need_matrix[i]):
            need_matrix[i][0] = 1
            max_matrix[i][0] += 1
        request = [rng.randint(0, need_matrix[i][j]) for j in range(m)]
        if rng.random() < 0.3:
            request[rng.randrange(m)] += rng.randint(1, 3)  # sometimes exceed Need, to test rejection

        req_str = ", ".join(f"{res_names[j]}={request[j]}" for j in range(m))
        question = (
            f"A system has {n} processes (P0..P{n-1}) and {m} resource types "
            f"({', '.join(res_names)}). The current Allocation and Max "
            f"matrices, and the Available vector, are:\n\n"
            f"Allocation:\n{fmt_matrix(alloc_matrix)}\n\n"
            f"Max:\n{fmt_matrix(max_matrix)}\n\n"
            f"Available: {', '.join(f'{res_names[j]}={available[j]}' for j in range(m))}\n\n"
            f"Process P{i} now requests additional resources: {req_str}. "
            f"Using the Banker's algorithm, determine whether this request "
            f"can be granted immediately."
        )

        exceeds_need = any(request[j] > need_matrix[i][j] for j in range(m))
        exceeds_avail = any(request[j] > available[j] for j in range(m))
        ans_lines = [f"Need = Max - Allocation:\n{fmt_matrix(need_matrix)}", ""]
        ans_lines.append(f"Step 1: Is Request <= Need[P{i}]? Request={request}, Need[P{i}]={need_matrix[i]}")
        if exceeds_need:
            ans_lines.append(
                f"-> No. The request exceeds P{i}'s declared maximum need, "
                f"which is an error condition. The request cannot be granted."
            )
        else:
            ans_lines.append("-> Yes.")
            ans_lines.append(f"\nStep 2: Is Request <= Available? Request={request}, Available={available}")
            if exceeds_avail:
                ans_lines.append(
                    f"-> No. Available cannot satisfy the request, so P{i} "
                    f"must wait until enough resources are free."
                )
            else:
                ans_lines.append("-> Yes. Pretend to allocate and check safety of the resulting state.")
                new_alloc = [row[:] for row in alloc_matrix]
                new_need = [row[:] for row in need_matrix]
                new_avail = available[:]
                for j in range(m):
                    new_avail[j] -= request[j]
                    new_alloc[i][j] += request[j]
                    new_need[i][j] -= request[j]
                safe, sequence, _, _ = safety_algorithm(n, m, new_alloc, new_need, new_avail)
                ans_lines.append(f"\nTentative new Available: {new_avail}")
                if safe:
                    seq_str = " -> ".join(f"P{x}" for x in sequence)
                    ans_lines.append(
                        f"The resulting state IS safe (safe sequence: "
                        f"{seq_str}). The request is GRANTED immediately."
                    )
                else:
                    ans_lines.append(
                        "The resulting state is NOT safe (no safe sequence "
                        f"exists). The request is therefore DENIED for now — "
                        f"P{i} must wait, and the tentative allocation is "
                        "rolled back."
                    )
        answer = "\n".join(ans_lines)
        return BANKERS_CONTEXT, question, answer


# ══════════════════════════════════════════════════════════════════════
# 5. MEMORY ADDRESSING
# ══════════════════════════════════════════════════════════════════════

PAGING_CONTEXT = (
    "In a paged memory system, a logical address is split into a page "
    "number and an offset: page number = logical address // page size; "
    "offset = logical address % page size. The page table maps each page "
    "number to a physical frame number. The physical address is computed "
    "as: physical address = (frame number x page size) + offset."
)

FRAGMENTATION_CONTEXT = (
    "In contiguous memory allocation, free memory is organized as a list "
    "of holes (blocks). First-fit allocates a process into the first hole "
    "large enough to hold it (scanning from the start). Best-fit allocates "
    "into the smallest hole that is still large enough. Worst-fit "
    "allocates into the largest available hole. After allocation, the "
    "unused remainder of the chosen hole remains as a smaller free hole."
)


def gen_paging_translation(rng: random.Random):
    page_size = rng.choice([256, 512, 1024])
    num_pages = rng.randint(4, 7)
    logical_space = page_size * num_pages
    total_frames = rng.randint(num_pages + 2, num_pages + 8)
    frame_pool = rng.sample(range(total_frames), num_pages)
    page_table = {p: frame_pool[p] for p in range(num_pages)}
    logical_address = rng.randint(0, logical_space - 1)

    page_number = logical_address // page_size
    offset = logical_address % page_size
    frame_number = page_table[page_number]
    physical_address = frame_number * page_size + offset

    pt_str = ", ".join(f"page {p} -> frame {f}" for p, f in page_table.items())
    question = (
        f"A system uses paging with a page size of {page_size} bytes. A "
        f"process has {num_pages} pages, with the following page table: "
        f"{pt_str}.\n\n"
        f"For the logical address {logical_address}, determine: (a) the "
        f"page number, (b) the offset within the page, (c) the "
        f"corresponding frame number, and (d) the resulting physical "
        f"address."
    )
    answer = (
        f"(a) Page number = logical address // page size = {logical_address} // {page_size} = {page_number}\n"
        f"(b) Offset = logical address % page size = {logical_address} % {page_size} = {offset}\n"
        f"(c) From the page table, page {page_number} -> frame {frame_number}\n"
        f"(d) Physical address = (frame number x page size) + offset = "
        f"({frame_number} x {page_size}) + {offset} = {physical_address}"
    )
    return PAGING_CONTEXT, question, answer


def gen_fragmentation(rng: random.Random):
    algo = rng.choice(["first-fit", "best-fit", "worst-fit"])
    num_blocks = rng.randint(4, 6)
    blocks = [rng.randint(50, 400) for _ in range(num_blocks)]
    num_procs = rng.randint(3, 5)
    proc_sizes = [rng.randint(30, 300) for _ in range(num_procs)]

    remaining = list(blocks)
    steps = []
    for idx, size in enumerate(proc_sizes):
        candidates = [(i, remaining[i]) for i in range(len(remaining)) if remaining[i] >= size]
        if not candidates:
            steps.append((idx, size, None, None))
            continue
        if algo == "first-fit":
            chosen = candidates[0]
        elif algo == "best-fit":
            chosen = min(candidates, key=lambda c: c[1])
        else:
            chosen = max(candidates, key=lambda c: c[1])
        block_idx, block_size_before = chosen
        remaining[block_idx] = block_size_before - size
        steps.append((idx, size, block_idx, block_size_before))

    blocks_str = ", ".join(f"B{i}={blocks[i]}KB" for i in range(num_blocks))
    procs_str = ", ".join(f"P{i}={proc_sizes[i]}KB" for i in range(num_procs))
    question = (
        f"Free memory consists of the following holes, in this order: "
        f"{blocks_str}. The following processes arrive in this order and "
        f"must each be placed into a single contiguous free hole: "
        f"{procs_str}.\n\n"
        f"Using the {algo} allocation strategy, determine which memory "
        f"block (if any) each process is placed into, and the sizes of the "
        f"remaining free blocks after all allocations."
    )

    lines = []
    for idx, size, block_idx, block_size_before in steps:
        if block_idx is None:
            lines.append(f"P{idx} ({size}KB): no free block is large enough -> CANNOT be allocated")
        else:
            leftover = block_size_before - size
            lines.append(
                f"P{idx} ({size}KB): placed in B{block_idx} (was {block_size_before}KB) "
                f"-> B{block_idx} now has {leftover}KB free"
            )
    final_str = ", ".join(f"B{i}={remaining[i]}KB" for i in range(num_blocks))
    answer = "\n".join(lines) + f"\n\nFinal remaining free block sizes: {final_str}"

    return FRAGMENTATION_CONTEXT, question, answer


def gen_memory_addressing(rng: random.Random):
    if rng.random() < 0.5:
        return gen_paging_translation(rng)
    return gen_fragmentation(rng)


# ══════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════

GENERATORS = {
    "cpu_scheduling": gen_cpu_scheduling,
    "page_replacement": gen_page_replacement,
    "disk_scheduling": gen_disk_scheduling,
    "bankers": gen_bankers,
    "memory_addressing": gen_memory_addressing,
}
