"""
kb_conceptual.py — Curated Operating Systems knowledge base.
==============================================================
This is the single source of truth for factual content used to build the
fine-tuning dataset's conceptual (non-numerical) examples.

Every (question, answer) pair here is hand-written to be a direct, correct,
textbook-accurate answer — unlike the old dataset.json, which trained the
model to deflect with a generic Socratic question instead of answering.

Structure
---------
CHAPTERS: list of chapters, each:
    {
        "chapter": "<chapter name>",
        "topics": [
            {
                "id": "<slug>",
                "context": "<short textbook-style passage — stands in for what
                             the Qdrant retriever would actually return for this
                             topic at inference time>",
                "qas": [ ("question", "answer"), ... ],
            },
            ...
        ],
    }

GLOSSARY: list of (term, one-line accurate definition) pairs, used to generate
extra short-definition examples with multiple question phrasings.
"""

CHAPTERS = [
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Introduction to Operating Systems",
        "topics": [
            {
                "id": "os_overview",
                "context": (
                    "An operating system (OS) is system software that acts as an "
                    "intermediary between computer hardware and the user, managing "
                    "hardware resources (CPU, memory, I/O devices, storage) and "
                    "providing a convenient environment for executing programs. Its "
                    "two primary goals are convenience for the user and efficient "
                    "utilization of hardware resources."
                ),
                "qas": [
                    ("What is an operating system?",
                     "An operating system is system software that manages computer "
                     "hardware and software resources and provides common services "
                     "for computer programs. It acts as an intermediary between the "
                     "user/application programs and the hardware, handling tasks "
                     "such as process management, memory management, file "
                     "management, and I/O control."),
                    ("What are the two main goals of an operating system?",
                     "(1) Convenience — provide a user-friendly, easy-to-use "
                     "environment for executing programs, and (2) Efficiency — "
                     "ensure the computer's hardware resources (CPU, memory, I/O "
                     "devices) are used effectively and fairly among competing "
                     "processes."),
                    ("List the main functions of an operating system.",
                     "Process management, memory management, file system "
                     "management, I/O device management, secondary storage "
                     "management, security and protection, networking, and "
                     "providing a user interface (CLI/GUI) and system calls for "
                     "application programs."),
                    ("What is the difference between a kernel and an operating system?",
                     "The kernel is the core component of the OS that runs in "
                     "privileged (kernel) mode and directly manages hardware — CPU "
                     "scheduling, memory, and device drivers. The operating system "
                     "is the broader software package that includes the kernel plus "
                     "utilities, system libraries, and user interface components "
                     "built on top of it."),
                ],
                "applied": [{
                            "question": (
                                "A new operating system is being designed for a low-power IoT sensor with "
                                "extremely limited CPU and memory, and no user ever interacts with it "
                                "directly. Between the OS's two core goals of convenience and efficiency, "
                                "which should the design prioritize, and why?"
                            ),
                            "answer": (
                                "Efficiency should be prioritized. The OS's two core goals are convenience "
                                "(a user-friendly, easy-to-use environment for executing programs) and "
                                "efficiency (ensuring hardware resources are used effectively). Since this "
                                "IoT sensor has no direct user interaction at all, the 'convenience' goal "
                                "has essentially no audience to serve — there's no one who benefits from a "
                                "friendlier interface. Meanwhile, the sensor's constrained CPU and memory "
                                "make efficient resource use critical just to keep the device functional. "
                                "So the design should lean almost entirely toward the efficiency goal, "
                                "minimizing overhead from any convenience-oriented features that would "
                                "consume scarce resources without benefiting anyone."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "os_types",
                "context": (
                    "Operating systems can be classified by how they schedule and "
                    "share the CPU among users and processes: batch systems, "
                    "multiprogrammed systems, time-sharing systems, real-time "
                    "systems, and distributed systems, each suited to different "
                    "workloads."
                ),
                "qas": [
                    ("What is a batch operating system?",
                     "A batch operating system groups similar jobs together and "
                     "executes them one after another without user interaction "
                     "during execution. Jobs are collected, batched, and processed "
                     "sequentially by an operator or job scheduler, which is "
                     "efficient for repetitive, non-interactive tasks but offers no "
                     "interactivity."),
                    ("What is multiprogramming and why is it used?",
                     "Multiprogramming is the technique of keeping multiple jobs in "
                     "main memory simultaneously so that the CPU always has a job "
                     "to execute. When one job needs to wait for I/O, the CPU "
                     "switches to another job instead of sitting idle, which "
                     "increases CPU utilization and system throughput."),
                    ("What is a time-sharing (multitasking) operating system?",
                     "A time-sharing OS extends multiprogramming by rapidly "
                     "switching the CPU among multiple users' processes (using "
                     "time slices), giving each user the illusion of having a "
                     "dedicated, responsive system even though the CPU is actually "
                     "shared, e.g. Windows, Linux, UNIX."),
                    ("What distinguishes a hard real-time system from a soft real-time system?",
                     "A hard real-time system must meet its deadlines absolutely — "
                     "missing a deadline is considered a total system failure (e.g. "
                     "flight control, pacemakers). A soft real-time system prefers "
                     "to meet deadlines but tolerates occasional misses with "
                     "degraded (not catastrophic) results, e.g. multimedia "
                     "streaming."),
                    ("What is a distributed operating system?",
                     "A distributed OS manages a group of independent, networked "
                     "computers and makes them appear to users as a single "
                     "coherent system, providing resource sharing, transparency, "
                     "and coordinated computation across multiple physical "
                     "machines."),
                ],
                "applied": [{
                            "question": (
                                "A hospital's patient-monitoring system must guarantee that a heart-rate "
                                "alarm is raised within a strict, guaranteed time window every time — a "
                                "missed alarm is unacceptable. Which type of operating system does this "
                                "need, and why would a standard time-sharing OS be unsuitable?"
                            ),
                            "answer": (
                                "This system needs a hard real-time operating system, since a hard "
                                "real-time system must meet its deadlines absolutely — missing a deadline "
                                "is considered a total system failure, which matches the requirement that "
                                "a missed alarm is unacceptable. A standard time-sharing OS is unsuitable "
                                "because it is designed to rapidly switch the CPU among multiple users' "
                                "processes to give each the illusion of a dedicated system, with no "
                                "guarantee about exactly when any particular process will next get the "
                                "CPU — it optimizes for overall fairness and responsiveness across many "
                                "processes, not for guaranteeing a specific process meets a strict "
                                "deadline every single time."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "os_structure",
                "context": (
                    "Operating systems can be structured in several ways, including "
                    "the simple/monolithic structure, layered approach, microkernel "
                    "architecture, and modular approach, each trading off "
                    "performance against maintainability and reliability."
                ),
                "qas": [
                    ("What is a monolithic kernel?",
                     "A monolithic kernel is an OS design where the entire "
                     "operating system — process management, memory management, "
                     "file systems, device drivers — runs as a single large "
                     "program in kernel mode. It offers high performance due to "
                     "direct function calls between components but is harder to "
                     "maintain, and a bug in any part can crash the whole system."),
                    ("What is a microkernel and what is its main advantage?",
                     "A microkernel keeps only the most essential functions (basic "
                     "IPC, minimal process/memory management) in kernel mode, "
                     "moving services like file systems and device drivers to "
                     "user-mode servers. Its main advantage is improved reliability "
                     "and security — a failure in a user-mode service does not "
                     "crash the kernel — at the cost of extra message-passing "
                     "overhead between components."),
                    ("What is the layered approach to OS design?",
                     "In the layered approach, the OS is divided into a hierarchy "
                     "of layers, each built on top of lower layers and using only "
                     "their services. Layer 0 is the hardware and the topmost layer "
                     "is the user interface. This improves modularity and debugging "
                     "but adds performance overhead since a request may have to "
                     "pass through many layers."),
                    ("What is the difference between a system call and a system program?",
                     "A system call is a low-level, direct request from a program "
                     "to the kernel for a service (e.g. read(), fork()), typically "
                     "invoked via a trap/software interrupt. A system program (or "
                     "system utility) is a higher-level, user-facing program "
                     "shipped with the OS (e.g. a file manager, compiler, or text "
                     "editor) that itself may use several system calls internally."),
                ],
                "negative": [{
                            "question": (
                                "Does choosing a microkernel architecture over a monolithic kernel always "
                                "make every individual operation slower?"
                            ),
                            "answer": (
                                "Not necessarily as an absolute rule. The stated trade-off is that a "
                                "microkernel's main advantage is improved reliability and security — a "
                                "failure in a user-mode service doesn't crash the kernel — 'at the cost of "
                                "extra message-passing overhead between components.' That overhead "
                                "specifically applies to operations that must cross between the kernel and "
                                "a user-mode service (e.g. a file system or driver call), not to every "
                                "operation an OS performs in general. So while microkernel designs do "
                                "introduce overhead for those specific cross-component interactions, the "
                                "passage doesn't state that a microkernel makes literally every operation "
                                "slower than a monolithic kernel — only that this particular class of "
                                "interaction pays a message-passing cost."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "system_calls",
                "context": (
                    "System calls provide the interface between a running program "
                    "and the operating system, allowing user programs to request "
                    "kernel services such as file operations, process control, and "
                    "communication."
                ),
                "qas": [
                    ("What is a system call?",
                     "A system call is the programming interface through which a "
                     "user-mode process requests a service from the operating "
                     "system kernel, such as creating a process, opening a file, or "
                     "allocating memory. It causes a controlled switch from user "
                     "mode to kernel mode via a trap."),
                    ("What are the major categories of system calls?",
                     "Process control (fork, exec, exit, wait), file management "
                     "(open, read, write, close), device management (request, "
                     "release, read, write), information maintenance (getpid, "
                     "alarm, time), and communication (pipe, send, receive, shared "
                     "memory)."),
                    ("How does a system call switch from user mode to kernel mode?",
                     "The user program executes a special trap instruction (a "
                     "software interrupt) with a system call number and arguments "
                     "placed in registers or on the stack. This transfers control "
                     "to a fixed kernel entry point, the processor mode bit "
                     "switches to kernel mode, the kernel dispatches to the correct "
                     "system call handler and executes it with full hardware "
                     "access, then returns control and switches back to user "
                     "mode."),
                    ("What is the difference between user mode and kernel mode?",
                     "User mode is a restricted CPU execution mode in which "
                     "application code runs; it cannot directly execute privileged "
                     "instructions or access hardware. Kernel mode (supervisor "
                     "mode) is an unrestricted mode in which the OS runs, with full "
                     "access to all instructions and hardware, entered only via "
                     "system calls, interrupts, or exceptions."),
                ],
                "analytical": [{
                            "question": (
                                "Why must a system call switch the CPU into kernel mode via a trap "
                                "instead of the user program simply calling the kernel's function "
                                "directly, the way it would call any other function in its own code?"
                            ),
                            "answer": (
                                "System calls require the transition specifically so the switch to kernel "
                                "mode is controlled and verified at a single, fixed entry point rather "
                                "than the user program being able to jump into arbitrary kernel code. The "
                                "trap instruction transfers control to a fixed kernel entry point, at "
                                "which point the processor mode bit switches to kernel mode and the kernel "
                                "itself dispatches to the correct handler — the user program never gets to "
                                "decide directly what kernel code runs or with what privileges. If a user "
                                "program could simply 'call' kernel functions the way it calls its own "
                                "functions, user mode and kernel mode would offer no real protection "
                                "boundary at all, since user code could invoke privileged operations "
                                "however it liked; the trap mechanism is what makes user mode's "
                                "restriction (no direct execution of privileged instructions) actually "
                                "enforceable."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Process Management",
        "topics": [
            {
                "id": "process_basics",
                "context": (
                    "A process is a program in execution. It is the fundamental "
                    "unit of work in an operating system and includes the program "
                    "code, current activity (program counter, registers), and "
                    "associated resources such as memory and open files."
                ),
                "qas": [
                    ("What is the difference between a process and a program?",
                     "A program is a passive entity — static executable code "
                     "stored on disk. A process is an active entity — a program in "
                     "execution, with its own program counter, registers, memory "
                     "space, and execution state. One program can be launched "
                     "multiple times, creating multiple separate processes."),
                    ("What is a Process Control Block (PCB) and what does it contain?",
                     "A PCB is a kernel data structure that stores all the "
                     "information needed to manage a specific process: process ID, "
                     "process state, program counter, CPU registers, CPU scheduling "
                     "information (priority), memory management information "
                     "(base/limit registers, page tables), accounting information, "
                     "and I/O status information (open files, allocated devices)."),
                    ("List the typical states of a process and briefly describe each.",
                     "New (the process is being created), Ready (waiting to be "
                     "assigned to the CPU), Running (instructions are being "
                     "executed by the CPU), Waiting/Blocked (waiting for some event "
                     "such as I/O completion or a signal), and Terminated (finished "
                     "execution)."),
                    ("What is context switching?",
                     "Context switching is the act of saving the state (PCB: "
                     "registers, program counter, memory pointers) of the "
                     "currently running process and loading the saved state of "
                     "another process so the CPU can switch to executing it. It "
                     "enables multitasking but is pure overhead — no useful work is "
                     "done during the switch itself."),
                    ("What is the difference between a process's logical address space and physical address space?",
                     "The logical (virtual) address space is the set of addresses "
                     "generated by the CPU as a process runs, starting typically "
                     "from 0. The physical address space is the actual set of "
                     "addresses in main memory (RAM) that these logical addresses "
                     "are mapped to by the Memory Management Unit (MMU) at run "
                     "time."),
                ],
                "applied": [{
                            "question": (
                                "A running process on a single-CPU system issues a read() call to a slow "
                                "disk device and must wait for the data. Walk through which process state "
                                "it moves out of, which state it moves into, and what happens to the CPU "
                                "it was using."
                            ),
                            "answer": (
                                "The process moves out of the Running state (its instructions were being "
                                "executed by the CPU) and into the Waiting/Blocked state, since it is now "
                                "waiting for an event — the I/O completion of its disk read — before it "
                                "can continue. Because a single CPU can only run one process's "
                                "instructions at a time and this process can make no progress until its "
                                "I/O finishes, the CPU it was using becomes free for the scheduler to "
                                "allocate to a different process that is in the Ready state (waiting to "
                                "be assigned to the CPU), rather than sitting idle while this process "
                                "waits."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "process_scheduling_queues",
                "context": (
                    "As processes enter the system, they are placed in a job "
                    "queue. Processes in main memory waiting to run are in the "
                    "ready queue, and processes waiting for a particular I/O "
                    "device are placed in a device queue."
                ),
                "qas": [
                    ("What is the difference between a job queue, a ready queue, and a device (I/O) queue?",
                     "The job queue holds all processes in the system (including "
                     "those not yet in memory). The ready queue holds processes "
                     "that are in main memory and ready to execute, waiting for CPU "
                     "allocation. A device queue holds processes waiting for a "
                     "specific I/O device to become available."),
                    ("What is the difference between a long-term scheduler and a short-term scheduler?",
                     "The long-term scheduler (job scheduler) selects which "
                     "processes are admitted from secondary storage into the ready "
                     "queue in memory, controlling the degree of multiprogramming "
                     "and running infrequently. The short-term scheduler (CPU "
                     "scheduler) selects which ready process gets the CPU next, "
                     "running very frequently (every few milliseconds)."),
                    ("What is a medium-term scheduler and why is it used?",
                     "A medium-term scheduler temporarily removes (swaps out) a "
                     "process from main memory to secondary storage to reduce the "
                     "degree of multiprogramming, and later swaps it back in. It is "
                     "used to improve the process mix and free up memory when the "
                     "system is overloaded."),
                ],
                "analytical": [{
                            "question": (
                                "Why does the long-term scheduler run far less frequently than the "
                                "short-term (CPU) scheduler?"
                            ),
                            "answer": (
                                "The long-term scheduler controls the degree of multiprogramming by "
                                "deciding which processes are admitted from secondary storage into the "
                                "ready queue in memory — this is a comparatively rare, coarse-grained "
                                "decision, since a process typically stays resident in memory for its "
                                "entire lifetime once admitted, so there's no need to re-run this decision "
                                "often. The short-term scheduler, by contrast, decides which "
                                "already-resident, ready process gets the CPU next — a decision that must "
                                "be made constantly, every time the currently running process is "
                                "preempted, blocks, or finishes, which happens on the order of "
                                "milliseconds. Because CPU allocation must be re-decided vastly more often "
                                "than memory admission, the short-term scheduler necessarily runs far more "
                                "frequently than the long-term one."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "threads",
                "context": (
                    "A thread is the basic unit of CPU utilization, comprising a "
                    "thread ID, program counter, register set, and stack. Multiple "
                    "threads within the same process share the process's code "
                    "section, data section, and other OS resources like open "
                    "files."
                ),
                "qas": [
                    ("What is a thread and how does it differ from a process?",
                     "A thread is a lightweight unit of execution within a "
                     "process, with its own program counter, register set, and "
                     "stack, but sharing the code, data, and resources (open "
                     "files, memory) of its parent process. A process is a "
                     "heavier, independent unit with its own separate address "
                     "space; threads within one process can communicate more "
                     "efficiently since they share memory directly."),
                    ("What are the benefits of multithreading?",
                     "Responsiveness (a program can continue running even if part "
                     "of it is blocked), resource sharing (threads share memory "
                     "and resources of the process, cheaper than separate "
                     "processes), economy (creating and context-switching threads "
                     "is cheaper than processes), and scalability (threads can run "
                     "in parallel on multicore systems)."),
                    ("Differentiate between user-level threads and kernel-level threads.",
                     "User-level threads are managed by a thread library in user "
                     "space without kernel awareness — fast to create/switch, but "
                     "the whole process blocks if one thread makes a blocking "
                     "system call, and they cannot run in parallel on multiple "
                     "cores. Kernel-level threads are managed and scheduled "
                     "directly by the OS kernel — slower to create/switch (needs a "
                     "kernel call), but a blocked thread doesn't block others, and "
                     "they can run in true parallel on multicore systems."),
                    ("Explain the many-to-one, one-to-one, and many-to-many multithreading models.",
                     "Many-to-one maps many user threads to a single kernel thread "
                     "— efficient, but one blocking call blocks all threads and "
                     "there is no true parallelism. One-to-one maps each user "
                     "thread to its own kernel thread — allows parallelism and "
                     "blocking calls don't block other threads, but thread "
                     "creation overhead is higher (e.g. Windows, Linux). "
                     "Many-to-many multiplexes many user threads onto a smaller or "
                     "equal number of kernel threads, giving developers "
                     "flexibility while letting the OS create enough kernel "
                     "threads for parallelism."),
                ],
                "applied": [{
                            "question": (
                                "A web server needs to handle thousands of simultaneous client "
                                "connections on a multicore machine, where any single client's blocking "
                                "I/O call must never stall the handling of other clients, and true "
                                "parallel execution across cores is required. Which multithreading model "
                                "(many-to-one, one-to-one, or many-to-many) best fits this requirement, "
                                "and why do the other two fall short?"
                            ),
                            "answer": (
                                "One-to-one fits best here, since it maps each user thread to its own "
                                "kernel thread, meaning a blocking call from one thread's connection "
                                "doesn't block any other thread, and multiple threads can run in true "
                                "parallel across cores — both hard requirements in this scenario. "
                                "Many-to-one falls short because it maps many user threads onto a single "
                                "kernel thread: one blocking call blocks all threads, and there is no "
                                "true parallelism at all, since the kernel only ever sees one schedulable "
                                "thread. Many-to-many could also satisfy both requirements in principle, "
                                "since it multiplexes user threads onto multiple kernel threads, but it "
                                "adds implementation complexity for the flexibility it offers; for a "
                                "requirement stated as simply as this one, one-to-one is the most direct, "
                                "guaranteed fit."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "process_creation_ipc",
                "context": (
                    "Processes are created dynamically using system calls such as "
                    "fork() and exec() on Unix-like systems, and cooperating "
                    "processes exchange data using inter-process communication "
                    "(IPC) mechanisms such as shared memory and message passing."
                ),
                "qas": [
                    ("What is the difference between fork() and exec() in process creation?",
                     "fork() creates a new process (the child) by duplicating the "
                     "calling (parent) process — the child gets a copy of the "
                     "parent's memory, and after fork() both continue executing "
                     "from the same point, distinguished by the return value (0 in "
                     "the child, the child's PID in the parent). exec() replaces "
                     "the calling process's own memory image (code, data, stack) "
                     "with a new program, without creating a new process — "
                     "commonly, a child calls fork() then immediately calls exec() "
                     "to run a completely different program in place of its copy "
                     "of the parent."),
                    ("What is a zombie process, and what is an orphan process?",
                     "A zombie process is a process that has finished execution "
                     "(called exit()) but whose entry still remains in the process "
                     "table because its parent hasn't yet called wait() to read "
                     "its exit status — it holds no resources except its slot in "
                     "the table. An orphan process is one whose parent terminated "
                     "before it did; orphans are typically re-parented to a "
                     "special system process (like init/PID 1 on Unix), which "
                     "periodically calls wait() on them to clean them up."),
                    ("What are the main IPC (inter-process communication) models, and how do they differ?",
                     "Shared memory, where processes establish a region of memory "
                     "they both can read/write directly — fast, but requires the "
                     "processes themselves to synchronize access (e.g. using "
                     "semaphores) to avoid race conditions. Message passing, where "
                     "processes exchange data via send()/receive() primitives "
                     "(directly, or through the kernel via pipes, message queues, "
                     "sockets) — easier to use correctly and works across machines "
                     "in a distributed system, but has more overhead per "
                     "communication since the kernel is typically involved."),
                    ("What is the difference between a pipe and shared memory for IPC?",
                     "A pipe is a unidirectional communication channel managed by "
                     "the kernel, where one process writes bytes in and another "
                     "reads them out in FIFO order — simple and safe, but every "
                     "byte transferred goes through a kernel-mediated buffer, "
                     "adding overhead. Shared memory maps the same physical memory "
                     "region into both processes' address spaces, letting them "
                     "read/write directly without kernel involvement per "
                     "operation — much faster for large or frequent data exchange, "
                     "but requires explicit application-level synchronization."),
                ],
                "negative": [{
                            "question": (
                                "If a parent process calls fork() and then exits immediately, while its "
                                "child is still running normally (not yet finished), does the "
                                "still-running child become a zombie process?"
                            ),
                            "answer": (
                                "No. A zombie process is specifically one that has already finished "
                                "execution but whose entry remains in the process table because its "
                                "parent hasn't yet called wait() to read its exit status — zombie status "
                                "requires the CHILD to have finished. Here, the child is described as "
                                "still running normally, not finished, so it cannot be a zombie. Instead, "
                                "since its parent terminated first, the still-running child becomes an "
                                "orphan process — one whose parent terminated before it did — and it is "
                                "typically re-parented to a special system process (like init/PID 1 on "
                                "Unix), which will later call wait() on it once it does finish, "
                                "preventing it from ever becoming a zombie."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "CPU Scheduling",
        "topics": [
            {
                "id": "scheduling_criteria",
                "context": (
                    "CPU scheduling algorithms are compared using several "
                    "criteria: CPU utilization, throughput, turnaround time, "
                    "waiting time, and response time. Different algorithms "
                    "optimize different criteria, and no single algorithm is best "
                    "for every situation."
                ),
                "qas": [
                    ("List the key criteria used to evaluate CPU scheduling algorithms.",
                     "CPU utilization (keep the CPU as busy as possible), "
                     "throughput (number of processes completed per unit time), "
                     "turnaround time (total time from submission to completion of "
                     "a process), waiting time (total time a process spends "
                     "waiting in the ready queue), and response time (time from "
                     "submission until the first response is produced, important "
                     "for interactive systems)."),
                    ("Define turnaround time and waiting time for a process.",
                     "Turnaround time = Completion Time − Arrival Time (the total "
                     "time the process spends in the system). Waiting time = "
                     "Turnaround Time − Burst Time (the total time the process "
                     "spends waiting in the ready queue, not running or doing "
                     "I/O)."),
                    ("What is the difference between preemptive and non-preemptive scheduling?",
                     "In non-preemptive scheduling, once the CPU is allocated to a "
                     "process, it keeps the CPU until it terminates or switches to "
                     "the waiting state voluntarily. In preemptive scheduling, the "
                     "OS can forcibly take the CPU away from a running process "
                     "(e.g. when a higher-priority process arrives or a time "
                     "quantum expires) and give it to another process."),
                ],
                "analytical": [{
                            "question": (
                                "Why can't a scheduler simultaneously maximize CPU utilization/throughput "
                                "and minimize every individual process's waiting time in all cases?"
                            ),
                            "answer": (
                                "Maximizing CPU utilization and throughput generally means keeping the "
                                "CPU as continuously busy as possible and completing as many processes "
                                "per unit time as possible — objectives best served by favoring whichever "
                                "process can run efficiently right now. Minimizing an individual "
                                "process's waiting time, though, is about how long ONE specific process "
                                "sits in the ready queue before it gets the CPU. These pull in different "
                                "directions once multiple processes compete: keeping the CPU maximally "
                                "busy processing whichever jobs are convenient can mean a specific "
                                "process (e.g. one with a long burst) is deliberately made to wait longer "
                                "while other, shorter jobs run in front of it. Since the listed criteria "
                                "(utilization, throughput, turnaround time, waiting time, response time) "
                                "are being optimized simultaneously for potentially conflicting goals "
                                "across an entire mix of processes, no single scheduling policy can be "
                                "strictly best on every criterion for every workload at once — this is "
                                "exactly why different algorithms exist, each favoring a different "
                                "criterion."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "fcfs",
                "context": (
                    "First-Come, First-Served (FCFS) scheduling allocates the CPU "
                    "to processes in the order they arrive in the ready queue. It "
                    "is implemented with a simple FIFO queue and is "
                    "non-preemptive."
                ),
                "qas": [
                    ("How does FCFS scheduling work?",
                     "FCFS scheduling maintains a simple FIFO queue of processes "
                     "ordered by arrival time. Whichever process arrives first is "
                     "allocated the CPU first, and it runs to completion (or until "
                     "it needs I/O) before the next process in the queue is "
                     "scheduled. It is non-preemptive and easy to implement."),
                    ("What is the main drawback of FCFS scheduling?",
                     "FCFS suffers from the convoy effect — if a long CPU-bound "
                     "process runs first, all the shorter processes behind it in "
                     "the queue must wait a long time, drastically increasing "
                     "average waiting time even though the short processes "
                     "individually need very little CPU time."),
                ],
                "applied": [{
                            "question": (
                                "Three processes arrive in this order: P1 needs 20ms of CPU time, P2 "
                                "needs 2ms, and P3 needs 3ms, all arriving at essentially the same time. "
                                "Under FCFS scheduling, describe what happens to P2 and P3, and name the "
                                "phenomenon."
                            ),
                            "answer": (
                                "Under FCFS, the CPU is allocated strictly in arrival order, so P1 runs "
                                "first and occupies the CPU for its full 20ms even though P2 and P3 each "
                                "need only a few milliseconds. P2 and P3 — despite being very short jobs "
                                "— must sit in the ready queue doing nothing until P1 finishes, so P2 "
                                "finishes at 22ms and P3 at 25ms, even though if either had run first "
                                "they could have completed in just 2-3ms. This is the convoy effect: a "
                                "long CPU-bound process (P1) running first forces much shorter processes "
                                "behind it to wait far longer than their own execution time would "
                                "require, drastically increasing their waiting time even though they "
                                "individually need very little CPU time."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "sjf",
                "context": (
                    "Shortest Job First (SJF) scheduling selects the ready "
                    "process with the smallest next CPU burst time. It can be "
                    "implemented as non-preemptive (classic SJF) or preemptive "
                    "(Shortest Remaining Time First, SRTF)."
                ),
                "qas": [
                    ("How does SJF (Shortest Job First) scheduling work, and why is it considered optimal?",
                     "SJF selects, among all processes currently in the ready "
                     "queue, the one with the smallest CPU burst time to run "
                     "next. It is provably optimal for minimizing average waiting "
                     "time for a given set of processes, because scheduling "
                     "shorter jobs first reduces the total time other jobs spend "
                     "waiting behind them."),
                    ("What is the main practical limitation of SJF scheduling?",
                     "SJF requires knowing (or accurately predicting) the length "
                     "of the next CPU burst in advance, which is generally not "
                     "knowable exactly for real workloads. In practice it is "
                     "approximated using techniques like exponential averaging of "
                     "past burst times."),
                    ("What is Shortest Remaining Time First (SRTF) and how does it differ from SJF?",
                     "SRTF is the preemptive version of SJF: if a new process "
                     "arrives with a burst time shorter than the remaining time of "
                     "the currently executing process, the CPU is preempted and "
                     "given to the new (shorter) process. Classic SJF is "
                     "non-preemptive — once started, a process runs to "
                     "completion."),
                    ("What is starvation in the context of SJF scheduling, and how can it be solved?",
                     "Starvation occurs when a process with a long burst time "
                     "never gets scheduled because shorter jobs keep arriving and "
                     "jumping ahead of it in the queue indefinitely. It is solved "
                     "using aging — gradually increasing the priority (or "
                     "effective priority) of a process the longer it waits, "
                     "guaranteeing it eventually gets scheduled."),
                ],
                "negative": [{
                            "question": (
                                "If every process currently in the ready queue happens to have exactly "
                                "the same CPU burst time, does SJF scheduling produce a different "
                                "execution order than FCFS scheduling?"
                            ),
                            "answer": (
                                "Not necessarily. SJF selects the process with the smallest CPU burst "
                                "time among those in the ready queue — but if every process has the "
                                "identical burst time, there is no single 'smallest' to distinguish "
                                "between them, so SJF's selection rule provides no distinguishing "
                                "information in this case. In practice, ties are broken by some "
                                "secondary rule (commonly arrival order), which means SJF ends up "
                                "selecting processes in the same order FCFS would anyway. SJF's "
                                "advantage over FCFS specifically comes from being able to favor shorter "
                                "jobs over longer ones — when burst times don't differ at all, that "
                                "advantage has nothing to act on."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "priority_scheduling",
                "context": (
                    "In priority scheduling, each process is assigned a priority "
                    "number, and the CPU is allocated to the process with the "
                    "highest priority; ties are typically broken using FCFS order."
                ),
                "qas": [
                    ("How does priority scheduling work?",
                     "Each process is assigned a priority number, and the CPU is "
                     "allocated to the process with the highest priority (by "
                     "convention, often the smallest number = highest priority). "
                     "It can be implemented as preemptive (a higher-priority "
                     "arrival preempts the running process) or non-preemptive."),
                    ("What problem does priority scheduling suffer from, and how is it solved?",
                     "It suffers from starvation (indefinite blocking) of "
                     "low-priority processes if high-priority processes keep "
                     "arriving. This is solved using aging, where a waiting "
                     "process's priority is gradually increased over time so it "
                     "eventually becomes the highest priority and gets "
                     "scheduled."),
                ],
                "applied": [{
                            "question": (
                                "A low-priority background process has been sitting in the ready queue "
                                "for a long time because higher-priority processes keep arriving ahead "
                                "of it. Describe how aging resolves this, referencing what actually "
                                "changes about the process."
                            ),
                            "answer": (
                                "Aging resolves this by gradually increasing the waiting process's "
                                "priority the longer it continues to wait, rather than leaving its "
                                "priority fixed. As new higher-priority processes keep arriving and "
                                "jumping ahead of it, the background process's priority number keeps "
                                "being adjusted upward (toward higher priority) over time. Eventually, "
                                "its aged priority becomes higher than that of newly-arriving processes, "
                                "at which point the scheduler will select it for the CPU instead of "
                                "continuing to pass it over — guaranteeing it eventually runs, which is "
                                "exactly the starvation problem aging is designed to solve."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "round_robin",
                "context": (
                    "Round Robin (RR) scheduling is designed for time-sharing "
                    "systems. Each process is given a small fixed unit of CPU "
                    "time called a time quantum, and the ready queue is treated as "
                    "a circular queue."
                ),
                "qas": [
                    ("How does Round Robin scheduling work?",
                     "Each process in the ready queue is given a fixed time "
                     "quantum (time slice) to run. If it doesn't finish within "
                     "that quantum, it is preempted and placed at the back of the "
                     "(circular) ready queue, and the next process is given the "
                     "CPU. This continues cycling through all ready processes "
                     "until each completes."),
                    ("How does the size of the time quantum affect Round Robin's performance?",
                     "If the quantum is too large, RR behaves like FCFS (poor "
                     "response time for interactive processes). If the quantum is "
                     "too small, context-switching overhead dominates and "
                     "throughput drops significantly. A good rule of thumb is "
                     "choosing a quantum where about 80% of CPU bursts are shorter "
                     "than it."),
                    ("Is Round Robin preemptive or non-preemptive, and why is it well suited to time-sharing systems?",
                     "Round Robin is preemptive — the timer interrupts the "
                     "running process when its quantum expires. It suits "
                     "time-sharing systems because every process gets a fair, "
                     "bounded share of CPU time repeatedly, giving good response "
                     "time to interactive users even under heavy load."),
                ],
                "applied": [{
                            "question": (
                                "A system administrator sets the Round Robin time quantum extremely high "
                                "(larger than any process's total burst time). What does the scheduling "
                                "behavior effectively become, and why?"
                            ),
                            "answer": (
                                "With a quantum larger than any process's total burst time, no process "
                                "ever actually gets preempted for running out of its time slice — each "
                                "process that starts running will finish its entire burst before the "
                                "quantum would expire. Since the quantum is never the limiting factor, "
                                "processes effectively just run to completion in whatever order they "
                                "were taken from the ready queue, which is exactly how FCFS behaves. This "
                                "matches the stated behavior directly: 'if the quantum is too large, RR "
                                "behaves like FCFS,' with the associated downside of poor response time "
                                "for interactive processes, since a process at the back of the queue must "
                                "wait for every process ahead of it to run to completion, uninterrupted."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "multilevel_queue",
                "context": (
                    "Multilevel queue and multilevel feedback queue scheduling "
                    "partition the ready queue into several separate queues, "
                    "often based on process type or observed behavior."
                ),
                "qas": [
                    ("What is multilevel queue scheduling?",
                     "The ready queue is partitioned into several separate queues "
                     "based on process type or priority (e.g. foreground/"
                     "interactive processes and background/batch processes), each "
                     "with its own scheduling algorithm. Processes are "
                     "permanently assigned to one queue, and scheduling between "
                     "queues can be fixed-priority preemptive or time-sliced."),
                    ("What is multilevel feedback queue scheduling and how does it differ from multilevel queue scheduling?",
                     "Multilevel feedback queue scheduling also uses multiple "
                     "queues, but allows processes to move between queues based "
                     "on their behavior — a process that uses too much CPU time is "
                     "demoted to a lower-priority queue, while one that waits a "
                     "lot (I/O-bound) may be promoted. This adapts dynamically, "
                     "unlike plain multilevel queue scheduling where the "
                     "assignment is fixed."),
                ],
                "analytical": [{
                            "question": (
                                "Why is a process being permanently assigned to a single queue a real "
                                "limitation of plain multilevel queue scheduling, motivating the move to "
                                "multilevel feedback queue scheduling?"
                            ),
                            "answer": (
                                "In plain multilevel queue scheduling, each process is permanently "
                                "assigned to one queue (e.g. based on process type), and that assignment "
                                "never changes even if the process's actual behavior changes over time — "
                                "for example, if a process placed in a background/batch queue later "
                                "starts behaving more interactively, it stays stuck in the batch queue's "
                                "scheduling treatment regardless. Multilevel feedback queue scheduling "
                                "fixes exactly this limitation by allowing processes to move between "
                                "queues based on their observed behavior — demoting a process that uses "
                                "too much CPU time to a lower-priority queue, and promoting one that "
                                "waits a lot (I/O-bound) — so the scheduling treatment adapts dynamically "
                                "to how a process actually behaves, instead of being locked in by a "
                                "one-time, permanent classification."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "scheduling_comparison",
                "context": (
                    "Different CPU scheduling algorithms trade off average "
                    "waiting time, responsiveness, fairness, and implementation "
                    "overhead differently."
                ),
                "qas": [
                    ("Compare FCFS and SJF scheduling in terms of average waiting time.",
                     "SJF minimizes average waiting time by always running the "
                     "shortest job available next, whereas FCFS ignores burst "
                     "length entirely and can force short jobs to wait behind "
                     "long ones (the convoy effect), typically producing a higher "
                     "average waiting time than SJF for the same workload."),
                    ("Compare preemptive and non-preemptive scheduling in terms of responsiveness and overhead.",
                     "Preemptive scheduling gives better responsiveness for "
                     "interactive/real-time systems because high-priority or "
                     "time-sensitive processes can interrupt a running process, "
                     "but it incurs more context-switching overhead and requires "
                     "careful synchronization to avoid race conditions on shared "
                     "data. Non-preemptive scheduling has lower overhead and "
                     "simpler implementation but can cause long waits for other "
                     "processes if the running one holds the CPU for a long "
                     "time."),
                ],
                "applied": [{
                            "question": (
                                "A system handles a mix of short interactive user commands and long "
                                "batch computation jobs, and needs consistently low response times for "
                                "the interactive commands even while batch jobs are running. Based on "
                                "the responsiveness/overhead trade-off, should this system favor a "
                                "preemptive or non-preemptive scheduling policy, and what cost does that "
                                "choice accept?"
                            ),
                            "answer": (
                                "This system should favor a preemptive scheduling policy, since "
                                "preemptive scheduling gives better responsiveness for interactive "
                                "systems because a time-sensitive process (here, an interactive command) "
                                "can interrupt whatever is currently running — including a long batch "
                                "job — rather than having to wait for it to finish or voluntarily yield. "
                                "The cost accepted is more context-switching overhead, and the need for "
                                "careful synchronization to avoid race conditions on shared data, since a "
                                "running process (including the batch job) can now be interrupted at "
                                "points a non-preemptive design wouldn't need to guard against. A "
                                "non-preemptive policy would have lower overhead but would let a long "
                                "batch job hold the CPU for its entire run, directly violating the "
                                "requirement for consistently low interactive response times."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Process Synchronization",
        "topics": [
            {
                "id": "critical_section",
                "context": (
                    "The critical-section problem is to design a protocol that "
                    "processes can use to cooperate safely when accessing shared "
                    "data. A solution must satisfy mutual exclusion, progress, and "
                    "bounded waiting."
                ),
                "qas": [
                    ("What is the critical section problem?",
                     "It is the problem of designing a protocol so that when "
                     "multiple concurrent processes/threads access shared data "
                     "(their critical section), only one at a time can execute in "
                     "that critical section, preventing race conditions, while the "
                     "rest of their code (remainder section) can run freely."),
                    ("What are the three requirements a correct solution to the critical-section problem must satisfy?",
                     "(1) Mutual Exclusion — only one process may execute in its "
                     "critical section at a time. (2) Progress — if no process is "
                     "in its critical section, the selection of the next process "
                     "to enter must not be postponed indefinitely, and cannot be "
                     "decided by processes not requesting entry. (3) Bounded "
                     "Waiting — there must be a limit on how many times other "
                     "processes can enter their critical section after a process "
                     "has requested entry and before that request is granted."),
                    ("What is a race condition? Give an example.",
                     "A race condition occurs when multiple processes/threads "
                     "access and manipulate shared data concurrently, and the "
                     "final outcome depends on the particular, unpredictable order "
                     "in which the accesses happen. Example: two processes "
                     "increment a shared counter concurrently by reading it, "
                     "adding 1, and writing it back; if both read the same old "
                     "value before either writes, one increment is lost."),
                ],
                # Topic ids (from this file or kb_advanced.py) this topic is
                # commonly compared/contrasted with in exams. Consumed by
                # build_comparative_examples() via the COMPARISONS list below
                # — listed here too so the relationship is discoverable from
                # either side when scanning the KB.
                "related": ["deadlock_conditions"],
                "applied": [
                    {
                        "question": (
                            "A ticket-booking system runs three server processes that all handle "
                            "requests against the same 'seats_available' counter: each process reads "
                            "the counter, decrements it if the value is greater than zero, and writes "
                            "it back — with no locking. During a flash sale, all three processes read "
                            "the counter at nearly the same instant. What problem can occur, and why?"
                        ),
                        "answer": (
                            "This is a race condition on the shared 'seats_available' counter, caused "
                            "by the read-decrement-write sequence not being protected as a critical "
                            "section. If all three processes read the same value (say, 2 seats "
                            "available) before any of them writes back, each independently computes "
                            "'2 - 1 = 1' and writes 1 back — even though three separate bookings were "
                            "made. Instead of the counter correctly dropping to -1 (oversold, which the "
                            "check should have prevented for the third booking) or being correctly "
                            "serialized down to 0 with the third request rejected, it ends at 1: two "
                            "decrements are silently lost, and the system may believe more seats remain "
                            "than actually exist, or fail to detect that it oversold. The fix is to make "
                            "the entire read-decrement-write sequence a critical section, protected so "
                            "only one process can execute it at a time — e.g. with a mutex."
                        ),
                        "difficulty": "medium",
                    },
                ],
                "analytical": [
                    {
                        "question": (
                            "Why can't a race condition be fixed just by making each individual read "
                            "and each individual write atomic on their own, without protecting the "
                            "whole read-modify-write sequence as a single unit?"
                        ),
                        "answer": (
                            "Making each individual read and each individual write atomic only "
                            "guarantees that no other process can observe a torn/partial value during "
                            "that single operation — it says nothing about what can happen BETWEEN "
                            "operations. The critical section problem exists precisely because a "
                            "read-modify-write sequence is multiple separate steps, and another process "
                            "is free to run its own steps in the gap between them. For example, even if "
                            "each read and each write is individually atomic, one process can still read "
                            "a value, then be interrupted before it writes back, while a second process "
                            "reads the same (still-unmodified) value and writes its own update — both "
                            "processes end up basing their update on the same stale value. Correctness "
                            "requires the entire sequence to be treated as one indivisible critical "
                            "section (only one process executing any part of it at a time), not just "
                            "each individual memory access."
                        ),
                        "difficulty": "hard",
                    },
                ],
                "negative": [
                    {
                        "question": (
                            "If a shared variable is only ever read by multiple concurrent threads and "
                            "is never written by any of them after initialization, can a race condition "
                            "occur on that variable?"
                        ),
                        "answer": (
                            "No. A race condition requires the outcome to depend on the unpredictable "
                            "ORDER of accesses, which only matters when at least one of the concurrent "
                            "accesses is a write that changes the data — as in the classic example of two "
                            "processes incrementing a shared counter, where the danger comes specifically "
                            "from each process reading, modifying, and writing the value back. If every "
                            "concurrent access is a pure read of a value that never changes after "
                            "initialization, there is no write to interleave with, so no matter what order "
                            "the reads happen in, every thread observes the exact same, unchanging value. "
                            "This is also why a solution to the critical-section problem is unnecessary "
                            "here: mutual exclusion, progress, and bounded waiting exist specifically to "
                            "protect a critical section where shared data is being modified — a block of "
                            "code that only reads unchanging data has no critical section to protect in "
                            "the first place, since there is nothing concurrent access could corrupt."
                        ),
                        "difficulty": "easy",
                    },
                ],
            },
            {
                "id": "peterson",
                "context": (
                    "Peterson's solution is a classic software-based algorithm "
                    "for two processes that satisfies mutual exclusion, progress, "
                    "and bounded waiting using two shared variables: a turn "
                    "variable and a boolean flag array."
                ),
                "qas": [
                    ("Describe Peterson's solution for two-process mutual exclusion.",
                     "Peterson's solution uses two shared variables: an integer "
                     "`turn` and a boolean array `flag[2]`. Before entering the "
                     "critical section, process i sets `flag[i] = true` and `turn "
                     "= j` (giving the other process priority), then busy-waits "
                     "while `flag[j] == true && turn == j`. This guarantees mutual "
                     "exclusion, progress, and bounded waiting for two processes, "
                     "but it only works correctly on architectures that preserve "
                     "the exact instruction ordering (an issue on modern "
                     "out-of-order/optimizing hardware)."),
                ],
                "analytical": [{
                            "question": (
                                "Why does Peterson's solution, despite being logically correct on paper, "
                                "fail to guarantee mutual exclusion on modern out-of-order processors?"
                            ),
                            "answer": (
                                "Peterson's solution's correctness depends on the two shared-variable "
                                "writes (setting flag[i] = true and turn = j) actually being visible to "
                                "the other process in that exact order before the busy-wait check runs, "
                                "and on the busy-wait's read of flag[j] and turn happening after those "
                                "writes are visible. It 'only works correctly on architectures that "
                                "preserve the exact instruction ordering' — but modern out-of-order/"
                                "optimizing hardware and compilers are explicitly allowed to reorder "
                                "independent-looking memory operations for performance, as long as a "
                                "SINGLE thread's own view of its own instructions still looks "
                                "sequential. If the hardware reorders when flag[i] or turn actually "
                                "become visible to the OTHER process relative to how the code was "
                                "written, the other process's busy-wait check can observe stale values "
                                "and both processes can end up believing it's safe to enter the critical "
                                "section simultaneously — breaking the exact ordering guarantee the "
                                "algorithm's correctness proof silently assumes."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "semaphores",
                "context": (
                    "A semaphore is an integer synchronization variable accessed "
                    "only through two atomic operations, wait() (P) and signal() "
                    "(V), used to control access to shared resources among "
                    "concurrent processes."
                ),
                "qas": [
                    ("What is a semaphore and what operations does it support?",
                     "A semaphore S is an integer variable that, besides "
                     "initialization, is accessed only through two standard atomic "
                     "operations: wait(S) (also called P), which decrements S and "
                     "blocks the caller if S becomes negative, and signal(S) (also "
                     "called V), which increments S and wakes a blocked process if "
                     "any are waiting."),
                    ("What is the difference between a binary semaphore and a counting semaphore?",
                     "A binary semaphore can only take the values 0 or 1 and "
                     "behaves like a mutex lock, allowing mutual exclusion for a "
                     "single resource. A counting semaphore can take any "
                     "non-negative integer value and is used to control access to "
                     "a resource with multiple identical instances (its value "
                     "represents the number of available instances)."),
                    ("How does a semaphore differ from a mutex lock?",
                     "A mutex lock is a simple binary locking mechanism intended "
                     "purely for mutual exclusion, and must be released by the "
                     "same thread/process that acquired it (ownership matters). A "
                     "semaphore is a more general counting mechanism that can also "
                     "be used for signaling between different processes/threads "
                     "(e.g. process A signals, process B waits), and has no "
                     "ownership requirement."),
                    ("What is priority inversion and how do semaphores relate to it?",
                     "Priority inversion occurs when a lower-priority process "
                     "holds a semaphore/lock needed by a higher-priority process, "
                     "and an unrelated medium-priority process preempts the "
                     "low-priority holder, indefinitely delaying the high-priority "
                     "process. It is commonly mitigated with priority inheritance, "
                     "where the low-priority holder temporarily inherits the "
                     "higher priority until it releases the lock."),
                ],
                "negative": [{
                            "question": (
                                "A resource pool has 5 identical, interchangeable printers, but the "
                                "counting semaphore protecting access to them is initialized to 3. Is "
                                "this semaphore correctly configured for this resource pool?"
                            ),
                            "answer": (
                                "No. A counting semaphore's value is used to control access to a "
                                "resource with multiple identical instances, and its value 'represents "
                                "the number of available instances.' Initializing it to 3 when there are "
                                "actually 5 identical, interchangeable printers means the semaphore "
                                "under-represents the true available capacity: it will block a 4th and "
                                "5th requesting process even while 2 real printers sit completely idle "
                                "and unused, since wait() only allows access while the semaphore's count "
                                "is positive, and it was set to 3 rather than 5. The semaphore's initial "
                                "value should match the actual number of interchangeable resource "
                                "instances it is protecting."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "classic_sync_problems",
                "context": (
                    "The producer-consumer, readers-writers, and dining "
                    "philosophers problems are classic synchronization problems "
                    "used to illustrate and test solutions to concurrent "
                    "resource-sharing challenges."
                ),
                "qas": [
                    ("Explain the producer-consumer (bounded-buffer) problem and how semaphores solve it.",
                     "Producers generate data items and place them into a "
                     "fixed-size shared buffer; consumers remove and process items "
                     "from the buffer. The solution uses three semaphores: `mutex` "
                     "(binary, for mutual exclusion on the buffer), `empty` "
                     "(counts empty slots, initialized to buffer size N), and "
                     "`full` (counts filled slots, initialized to 0). A producer "
                     "waits on `empty` and signals `full`; a consumer waits on "
                     "`full` and signals `empty`; both use `mutex` around the "
                     "actual buffer access."),
                    ("What is the readers-writers problem, and what is the difference between the first and second readers-writers problem?",
                     "Multiple reader processes may read shared data concurrently "
                     "without conflict, but a writer needs exclusive access (no "
                     "readers or other writers) while writing. The first "
                     "readers-writers problem gives readers priority (no reader "
                     "should wait if the resource is already open for reading, "
                     "which can starve writers); the second gives writers priority "
                     "(once a writer is ready, it writes as soon as possible, "
                     "which can starve readers)."),
                    ("Describe the dining philosophers problem and one common solution to avoid deadlock.",
                     "Five philosophers sit around a table with one fork between "
                     "each pair (5 forks total); each needs both adjacent forks to "
                     "eat. If every philosopher picks up their left fork "
                     "simultaneously, all wait forever for the right fork — a "
                     "deadlock. One common solution is to allow at most four "
                     "philosophers to try picking up forks simultaneously (a "
                     "resource-limiting solution), or to have one philosopher pick "
                     "up their right fork first while the others pick up left "
                     "first, breaking the circular wait."),
                    ("What is a monitor, and how does it simplify synchronization compared to using raw semaphores?",
                     "A monitor is a high-level synchronization construct that "
                     "encapsulates shared data along with the procedures that "
                     "operate on it, guaranteeing that only one process can be "
                     "active inside the monitor at a time (built-in mutual "
                     "exclusion). Condition variables (with wait() and signal()) "
                     "inside the monitor handle waiting for specific conditions. "
                     "It simplifies synchronization because the programmer doesn't "
                     "need to manually manage semaphores for mutual exclusion — "
                     "the compiler/runtime enforces it."),
                    ("What advantage does a monitor have over semaphores for solving synchronization problems?",
                     "A monitor bundles the shared data and its access procedures "
                     "together with automatic mutual exclusion enforced by the "
                     "compiler/runtime, so a programmer cannot forget to call "
                     "wait()/signal() correctly around a critical section — a very "
                     "common source of semaphore bugs (e.g. deadlock from a "
                     "missing signal, or races from a missing wait). This makes "
                     "correct concurrent code structurally easier to write and "
                     "verify than using raw semaphores directly."),
                ],
                "analytical": [
                    {
                        "question": (
                            "Why does the readers-writers problem require a different locking "
                            "discipline than plain mutual exclusion, instead of just wrapping every "
                            "access to the shared data in one ordinary mutex?"
                        ),
                        "answer": (
                            "Plain mutual exclusion treats every access as equally exclusive: only one "
                            "process may be inside the critical section at a time, full stop. But in the "
                            "readers-writers setting, multiple readers can safely access the shared data "
                            "SIMULTANEOUSLY without any conflict, since none of them modify it — forcing "
                            "readers to take turns one-at-a-time behind a single mutex would be strictly "
                            "more restrictive than necessary and needlessly hurt concurrency, since "
                            "reads alone can never produce a race condition. What the problem actually "
                            "requires is an asymmetric rule: any number of readers may proceed together, "
                            "OR exactly one writer may proceed alone, but readers and a writer (or two "
                            "writers) can never be active at the same time, since a writer changing the "
                            "data while a reader observes it (or two writers changing it concurrently) "
                            "would corrupt the data or produce an inconsistent read. A single plain mutex "
                            "cannot express 'many readers OR one writer' — it can only express 'one "
                            "process at a time' — so the problem needs its own dedicated construction "
                            "(e.g. counting how many readers are currently active, and only having the "
                            "first reader lock out writers and the last reader unlock them)."
                        ),
                        "difficulty": "hard",
                    },
                ],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Deadlocks",
        "topics": [
            {
                "id": "deadlock_conditions",
                "context": (
                    "A deadlock is a situation where a set of processes are all "
                    "blocked, each waiting for a resource held by another process "
                    "in the set, so none can proceed. Four conditions must hold "
                    "simultaneously for a deadlock to occur."
                ),
                "qas": [
                    ("What is a deadlock?",
                     "A deadlock is a state in which a set of two or more "
                     "processes are each waiting for an event (typically resource "
                     "release) that can only be caused by another process in the "
                     "same set, so none of them can ever proceed — they are stuck "
                     "waiting on each other indefinitely."),
                    ("List and briefly explain the four necessary conditions for deadlock (Coffman conditions).",
                     "(1) Mutual Exclusion — at least one resource must be held "
                     "in a non-shareable mode. (2) Hold and Wait — a process "
                     "holding at least one resource is waiting to acquire "
                     "additional resources held by others. (3) No Preemption — "
                     "resources cannot be forcibly taken away; they must be "
                     "released voluntarily by the process holding them. (4) "
                     "Circular Wait — a set of processes {P0, P1, ..., Pn} exists "
                     "such that P0 is waiting for a resource held by P1, P1 for "
                     "one held by P2, ..., and Pn is waiting for one held by P0."),
                    ("What is a Resource Allocation Graph (RAG) and how is it used to detect deadlock?",
                     "A RAG is a directed graph with process nodes and resource "
                     "nodes; a request edge (P → R) means process P is waiting for "
                     "resource R, and an assignment edge (R → P) means R is "
                     "currently allocated to P. If the graph contains a cycle, and "
                     "each resource in the cycle has only one instance, a deadlock "
                     "exists; if resources have multiple instances, a cycle is "
                     "necessary but not sufficient for deadlock."),
                ],
                "analytical": [{
                            "question": (
                                "Why must all four conditions (mutual exclusion, hold-and-wait, no "
                                "preemption, circular wait) hold simultaneously for deadlock to occur — "
                                "what breaks if just one of them is absent?"
                            ),
                            "answer": (
                                "Each condition removes a specific way processes could otherwise escape "
                                "a stuck waiting cycle, so if even one is absent, that escape route "
                                "reopens. Without mutual exclusion, a resource could be shared freely, so "
                                "no process would ever need to wait for exclusive access to it at all. "
                                "Without hold-and-wait, a process would never hold one resource while "
                                "blocked waiting for another, so it couldn't be 'stuck' holding "
                                "something another waiting process needs. Without no-preemption, the "
                                "resource a process is waiting for could simply be forcibly taken from "
                                "whoever holds it and handed over, breaking the wait immediately rather "
                                "than letting it persist. And without circular wait, the waiting "
                                "relationships among processes couldn't form a closed loop back to the "
                                "starting process, so at least one process in any chain would eventually "
                                "be able to get everything it needs and finish, freeing its resources "
                                "for the rest. Because deadlock specifically requires the situation "
                                "where no process can ever proceed, removing any one of these conditions "
                                "reopens some path to progress, which is why all four together are "
                                "necessary."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "deadlock_handling",
                "context": (
                    "Operating systems can handle deadlocks in four principal "
                    "ways: ignore the problem, prevent deadlock by denying one of "
                    "the four necessary conditions, avoid deadlock using "
                    "algorithms like the Banker's algorithm, or allow deadlock and "
                    "use detection and recovery."
                ),
                "qas": [
                    ("What are the four general strategies for handling deadlocks in an OS?",
                     "(1) Deadlock ignorance — ignore the problem and assume it "
                     "never happens (used by most general-purpose OSes like "
                     "Windows/Linux, the 'ostrich algorithm'). (2) Deadlock "
                     "prevention — design the system so at least one of the four "
                     "necessary conditions can never hold. (3) Deadlock avoidance "
                     "— dynamically check whether granting a resource request "
                     "could lead to an unsafe state before granting it (e.g. "
                     "Banker's algorithm). (4) Deadlock detection and recovery — "
                     "allow deadlocks to occur, periodically check for them, and "
                     "recover (e.g. by aborting or preempting processes)."),
                    ("How can the \"hold and wait\" condition be prevented?",
                     "By requiring a process to request and be allocated all the "
                     "resources it will ever need before it starts execution, or "
                     "by requiring it to release all currently held resources "
                     "before requesting new ones. This guarantees low resource "
                     "utilization and can cause starvation, since a process may "
                     "never get all resources it needs at once."),
                    ("How can the \"circular wait\" condition be prevented?",
                     "By imposing a total ordering of all resource types and "
                     "requiring that each process requests resources in strictly "
                     "increasing order of enumeration. This makes a circular chain "
                     "of waiting impossible, since a process holding a "
                     "higher-numbered resource can never wait for a lower-numbered "
                     "one that would complete a cycle."),
                    ("What is the difference between deadlock prevention and deadlock avoidance?",
                     "Deadlock prevention restricts how requests can be made so "
                     "that one of the four necessary conditions structurally can "
                     "never occur (a static, conservative approach). Deadlock "
                     "avoidance allows all four conditions to potentially hold, "
                     "but dynamically examines the resource-allocation state "
                     "before granting each request and only grants it if the "
                     "resulting state is still 'safe' (won't lead to deadlock) — "
                     "this requires advance knowledge of each process's maximum "
                     "resource needs."),
                ],
                "applied": [{
                            "question": (
                                "A safety-critical embedded system controlling industrial machinery "
                                "cannot tolerate the 'ignore the problem' approach to deadlocks, since an "
                                "undetected deadlock could leave machinery in a dangerous state. Which of "
                                "the four general strategies for handling deadlocks would be "
                                "inappropriate here, and which are viable alternatives?"
                            ),
                            "answer": (
                                "Deadlock ignorance — assuming deadlocks never happen and doing nothing "
                                "about them, as most general-purpose OSes like Windows/Linux do — is "
                                "inappropriate here, since the scenario explicitly cannot tolerate an "
                                "undetected deadlock. The viable alternatives are deadlock prevention "
                                "(designing the system so at least one of the four necessary conditions "
                                "can never hold, structurally ruling deadlock out), deadlock avoidance "
                                "(dynamically checking whether granting a resource request could lead to "
                                "an unsafe state before granting it, e.g. via the Banker's algorithm), or "
                                "deadlock detection and recovery (allowing deadlocks to occur but "
                                "periodically checking for them and recovering, e.g. by aborting or "
                                "preempting processes). For a safety-critical system, prevention or "
                                "avoidance would generally be preferred over detection-and-recovery, "
                                "since they stop a deadlock from ever actually occurring rather than "
                                "reacting after machinery may already be stuck."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "bankers_conceptual",
                "context": (
                    "The Banker's algorithm is a deadlock-avoidance algorithm "
                    "that treats resource allocation like a banker extending "
                    "credit: it only grants a resource request if the system "
                    "remains in a safe state afterward, i.e., there exists some "
                    "order in which all processes can complete."
                ),
                "qas": [
                    ("What is a \"safe state\" in the context of deadlock avoidance?",
                     "A state is safe if there exists at least one safe sequence "
                     "— an ordering of all processes such that each process's "
                     "remaining resource need can be satisfied using the "
                     "currently available resources plus the resources held by "
                     "processes earlier in the sequence. If the system is in a "
                     "safe state, deadlock can always be avoided by following that "
                     "sequence."),
                    ("What data does the Banker's algorithm require about each process, and what does it check before granting a request?",
                     "It requires the Maximum claim (max resources each process "
                     "may ever need), the current Allocation (what it currently "
                     "holds), and Need = Max − Allocation for each process, plus "
                     "the vector of Available resources. Before granting a "
                     "request, it tentatively allocates the requested resources, "
                     "then runs the safety algorithm to check if the resulting "
                     "state is still safe; if yes, the request is granted, "
                     "otherwise the process must wait."),
                    ("What is the difference between deadlock detection and deadlock avoidance?",
                     "Avoidance proactively checks every resource request in "
                     "advance (before granting it) to ensure the system stays in "
                     "a safe state, requiring processes to declare their maximum "
                     "future needs upfront. Detection allows the system to enter a "
                     "deadlock, and periodically runs an algorithm to check if one "
                     "currently exists (using a wait-for graph or an "
                     "allocation/request matrix analysis), then triggers recovery "
                     "only if a deadlock is actually found."),
                ],
                "negative": [{
                            "question": (
                                "If a process declares a Maximum resource need that is larger than the "
                                "total number of instances of that resource type that will ever exist in "
                                "the system, can the system still reach a safe state with respect to "
                                "that process?"
                            ),
                            "answer": (
                                "No. A state is safe only if there exists a safe sequence in which each "
                                "process's remaining resource Need can eventually be satisfied using "
                                "currently available resources plus resources freed by processes earlier "
                                "in the sequence — but the most that could ever become available to any "
                                "process, even in the best case, is the total number of instances of "
                                "that resource type that exist in the system. If a process's declared "
                                "Maximum need exceeds that total, no amount of waiting or sequencing "
                                "could ever supply enough of that resource to satisfy it, since the "
                                "resource simply doesn't exist in sufficient quantity anywhere in the "
                                "system. No safe sequence could ever complete for that process, so the "
                                "system could never be considered safe with respect to it."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "deadlock_recovery",
                "context": (
                    "Once a deadlock is detected, an operating system can "
                    "recover from it either by terminating one or more deadlocked "
                    "processes or by preempting resources from them."
                ),
                "qas": [
                    ("What are the two main approaches to deadlock recovery once a deadlock is detected?",
                     "(1) Process termination — abort all deadlocked processes at "
                     "once (simple but wasteful), or abort them one at a time "
                     "until the deadlock cycle is broken (safer but requires "
                     "re-running detection after each abort). (2) Resource "
                     "preemption — successively preempt resources from some "
                     "processes and give them to others until the deadlock cycle "
                     "is broken, while addressing the issues of selecting a "
                     "victim, rollback, and starvation."),
                    ("What is starvation in the context of resource preemption during deadlock recovery, and how is it typically avoided?",
                     "If the same process is repeatedly chosen as the 'victim' "
                     "for resource preemption, it may never complete — this is "
                     "starvation. It is typically avoided by including the number "
                     "of times a process has been picked as a victim as a factor "
                     "in the selection cost, ensuring no process is picked an "
                     "unlimited number of times."),
                ],
                "negative": [{
                            "question": (
                                "If a system recovers from deadlock using resource preemption rather "
                                "than process termination, does the victim process necessarily have to "
                                "be killed?"
                            ),
                            "answer": (
                                "No. The two main approaches to deadlock recovery are described as "
                                "distinct alternatives: process termination (aborting one or more "
                                "deadlocked processes) and resource preemption (successively preempting "
                                "resources from some processes and giving them to others until the "
                                "deadlock cycle is broken). Resource preemption specifically takes back "
                                "a process's held resources rather than ending the process itself — the "
                                "process can, in principle, continue running later (potentially after "
                                "being rolled back to an earlier state) once the deadlock is resolved "
                                "and it can reacquire what it needs. Termination and preemption are "
                                "presented as two separate strategies, not as one implying the other."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Memory Management",
        "topics": [
            {
                "id": "memory_basics",
                "context": (
                    "Memory management is responsible for keeping track of each "
                    "byte of a computer's memory, allocating and deallocating "
                    "memory space as needed by processes, and protecting each "
                    "process's memory from unauthorized access by others."
                ),
                "qas": [
                    ("What is the difference between logical (virtual) address and physical address?",
                     "A logical address is generated by the CPU during program "
                     "execution and is what the process sees. A physical address "
                     "is the actual location in main memory (RAM). The Memory "
                     "Management Unit (MMU) hardware translates logical addresses "
                     "to physical addresses at run time."),
                    ("What is the difference between static (compile-time) and dynamic (execution-time) address binding?",
                     "Static binding assigns physical memory addresses to a "
                     "program at compile time or load time — the program must "
                     "always load into the same location. Dynamic binding delays "
                     "address assignment until run time, using base and limit "
                     "registers or a memory management unit, allowing a process to "
                     "be relocated in memory even after execution starts."),
                    ("What is swapping?",
                     "Swapping is the technique of temporarily moving a process's "
                     "entire memory image out to secondary storage (a backing "
                     "store) to free up main memory for other processes, and later "
                     "swapping it back in to continue execution. It increases the "
                     "degree of multiprogramming but adds significant I/O "
                     "overhead."),
                    ("How do base and limit registers provide memory protection?",
                     "The base register holds the smallest legal physical memory "
                     "address a process may access, and the limit register holds "
                     "the size (range) of that process's address space. Every "
                     "logical address generated by the CPU is checked in "
                     "hardware: if 0 ≤ logical address < limit, the physical "
                     "address = base + logical address is used; otherwise a trap "
                     "(addressing error) is raised to the OS, preventing a "
                     "process from accessing another process's or the OS's "
                     "memory."),
                ],
                "applied": [{
                            "question": (
                                "A process attempts to access a logical address that falls outside the "
                                "range covered by its limit register. Using the base/limit protection "
                                "mechanism, describe what the hardware does and why."
                            ),
                            "answer": (
                                "The hardware checks every logical address the CPU generates against the "
                                "condition '0 ≤ logical address < limit.' Since this access falls "
                                "outside that range, the check fails, and the hardware raises a trap "
                                "(addressing error) to the OS instead of allowing the access to proceed "
                                "to physical memory. This happens specifically because the base and "
                                "limit registers exist to prevent a process from accessing another "
                                "process's or the OS's memory — the limit register defines the size of "
                                "this process's legitimate address space, and any logical address at or "
                                "beyond that limit is, by construction, outside memory that belongs to "
                                "this process, so the hardware refuses to translate it into a physical "
                                "address at all."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "contiguous_allocation",
                "context": (
                    "In contiguous memory allocation, each process is allocated "
                    "a single contiguous block of main memory. Free memory blocks "
                    "(holes) are managed and allocated to new processes using "
                    "strategies like first-fit, best-fit, and worst-fit."
                ),
                "qas": [
                    ("Explain first-fit, best-fit, and worst-fit memory allocation strategies.",
                     "First-fit allocates the first free hole that is large "
                     "enough for the request, scanning from the start of memory — "
                     "fast but can leave odd fragments near the front. Best-fit "
                     "searches the entire list and allocates the smallest hole "
                     "big enough — minimizes wasted space per allocation but "
                     "tends to leave many tiny unusable fragments and is slower "
                     "(must search the whole list). Worst-fit allocates the "
                     "largest available hole, leaving the biggest possible "
                     "leftover fragment — reduces the number of tiny fragments "
                     "but tends to waste large amounts of memory."),
                    ("What is the difference between internal fragmentation and external fragmentation?",
                     "Internal fragmentation is wasted memory inside an allocated "
                     "block — the block is slightly larger than what's needed "
                     "(common in fixed-size partitioning/paging when a request "
                     "doesn't exactly fill a block/page). External fragmentation "
                     "is wasted memory between allocated blocks — enough total "
                     "free memory exists to satisfy a request, but it's scattered "
                     "in small non-contiguous holes so no single hole is big "
                     "enough (common in contiguous/variable-size allocation)."),
                    ("What is compaction, and why is it used?",
                     "Compaction is the process of shuffling memory contents to "
                     "relocate all allocated blocks together, consolidating all "
                     "free memory (external fragmentation) into one large "
                     "contiguous block. It solves external fragmentation but is "
                     "expensive since it requires dynamic relocation and copying "
                     "live process memory."),
                ],
                "applied": [{
                            "question": (
                                "Free memory currently has one large 500KB hole and several tiny 5-10KB "
                                "holes scattered elsewhere. A new process needs 480KB. Compare how "
                                "first-fit and worst-fit would handle this specific request."
                            ),
                            "answer": (
                                "First-fit scans from the start of memory and allocates the first hole "
                                "large enough for the request — if the 500KB hole happens to be "
                                "encountered before any other hole big enough (and none of the tiny "
                                "5-10KB holes qualify, since they're too small for a 480KB request), "
                                "first-fit would allocate into that 500KB hole, leaving a small ~20KB "
                                "leftover fragment there. Worst-fit specifically allocates the largest "
                                "available hole to reduce the number of tiny leftover fragments — here, "
                                "the 500KB hole is almost certainly the largest hole available, so "
                                "worst-fit would also choose it, again leaving roughly 20KB free. In "
                                "this particular scenario, because only one hole is actually large "
                                "enough to satisfy the request at all, first-fit and worst-fit converge "
                                "on the same choice — their difference in strategy would only really "
                                "diverge if multiple holes were large enough to hold the process."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "paging",
                "context": (
                    "Paging is a memory management scheme that permits a "
                    "process's physical address space to be non-contiguous. "
                    "Logical memory is divided into fixed-size blocks called "
                    "pages, and physical memory into equal-size blocks called "
                    "frames."
                ),
                "qas": [
                    ("What is paging, and how does it eliminate external fragmentation?",
                     "Paging divides a process's logical address space into "
                     "fixed-size blocks called pages and physical memory into "
                     "equal-size blocks called frames. Any page can be placed in "
                     "any available frame, so a process's memory need not be "
                     "contiguous in physical RAM, eliminating external "
                     "fragmentation entirely (though it can still cause internal "
                     "fragmentation in the last page)."),
                    ("What is a page table, and what information does each entry typically contain?",
                     "A page table is a per-process data structure maintained by "
                     "the OS that maps each logical page number to the physical "
                     "frame number where it currently resides. Each entry "
                     "typically also contains a valid/invalid bit (whether the "
                     "page is currently in memory), protection bits "
                     "(read/write/execute), a reference bit, and a dirty/modified "
                     "bit."),
                    ("How is a logical address translated to a physical address in a paged system?",
                     "The logical address is split into a page number (p) and a "
                     "page offset (d). The page number indexes into the page "
                     "table to find the corresponding frame number (f). The "
                     "physical address is then computed as (f × page size) + d — "
                     "the frame number combined with the same offset within that "
                     "frame."),
                    ("What is a Translation Look-aside Buffer (TLB), and why is it used?",
                     "A TLB is a small, fast, associative (content-addressable) "
                     "hardware cache that stores recent page-number-to-frame-"
                     "number translations. It is used to avoid the extra memory "
                     "access required to consult the full page table in main "
                     "memory for every reference — a TLB hit gives the frame "
                     "number immediately, while a TLB miss requires walking the "
                     "page table (and then caching the result in the TLB)."),
                    ("What is a multilevel (hierarchical) page table, and why is it needed?",
                     "A multilevel page table splits a single large page table "
                     "into a tree of smaller tables (e.g. an outer page table "
                     "indexing inner page tables) so that only the portions of "
                     "the page table actually needed are kept in memory. It is "
                     "needed because for large (e.g. 32-bit or 64-bit) address "
                     "spaces, a single flat page table would itself require a "
                     "huge, mostly-unused contiguous block of memory."),
                    ("What is an inverted page table, and what problem does it solve?",
                     "An inverted page table has one entry per physical frame "
                     "(not per logical page), recording which process and which "
                     "page currently occupies that frame. It solves the problem "
                     "of page tables growing too large for huge address spaces "
                     "(since its size depends on physical memory size, not "
                     "virtual address space size), at the cost of slower lookups "
                     "(often requiring a hash table) since it's no longer indexed "
                     "directly by page number."),
                ],
                "applied": [{
                            "question": (
                                "A process has a page table where page 5 maps to frame 12, and the "
                                "system uses a page size of 4096 bytes. What is the physical address "
                                "corresponding to logical address 20500, and what are the intermediate "
                                "page number and offset?"
                            ),
                            "answer": (
                                "The logical address is split into a page number and offset using the "
                                "page size: page number = 20500 // 4096 = 5 (since 5×4096 = 20480, and "
                                "20500 is within the next 4096-byte block), and offset = 20500 − 20480 = "
                                "20. Looking up page 5 in the page table gives frame 12. The physical "
                                "address is then computed as (frame number × page size) + offset = "
                                "(12 × 4096) + 20 = 49152 + 20 = 49172."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
                "negative": [{
                            "question": (
                                "In a paging system, can two different logical pages ever need to be "
                                "placed in the same physical frame at the same time?"
                            ),
                            "answer": (
                                "No. Paging works precisely because any page can be placed in any "
                                "AVAILABLE frame — but 'available' means not already occupied. Each "
                                "physical frame holds exactly one page's worth of data at a time; a page "
                                "table entry maps a specific logical page to the one frame currently "
                                "holding its data. If a second page needed to occupy a frame already "
                                "holding a different page, the first page's data would simply be "
                                "overwritten and lost (which is exactly what happens deliberately during "
                                "page replacement, where an old page is evicted before a new one takes "
                                "its frame) — but at any single instant, one frame holds at most one "
                                "page, never two simultaneously."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "segmentation",
                "context": (
                    "Segmentation is a memory management scheme that supports "
                    "the programmer's view of memory as a collection of "
                    "variable-sized logical segments, such as code, stack, and "
                    "data, rather than a single linear array."
                ),
                "qas": [
                    ("What is segmentation, and how does it differ from paging?",
                     "Segmentation divides a process's logical address space "
                     "into variable-sized, logically meaningful segments (e.g. "
                     "code, data, stack, heap), each with its own base and "
                     "limit. Paging divides memory into fixed-size pages/frames "
                     "with no inherent logical meaning. Segmentation matches the "
                     "programmer's logical view of a program but can suffer from "
                     "external fragmentation (since segments are variable-sized), "
                     "while paging avoids external fragmentation but has no "
                     "logical structure."),
                    ("How is a logical address translated in a pure segmentation scheme?",
                     "A logical address consists of a segment number (s) and an "
                     "offset (d). The segment number indexes a segment table to "
                     "retrieve the segment's base address and limit (length). If "
                     "the offset is less than the limit, the physical address is "
                     "computed as base + offset; if the offset exceeds the limit, "
                     "a segmentation fault (trap) occurs."),
                    ("What is segmentation with paging, and why is it used?",
                     "It's a hybrid scheme where each segment (code, data, "
                     "stack, etc.) is itself divided into fixed-size pages, "
                     "combining the logical structure of segmentation with the "
                     "fragmentation-free allocation of paging. Each segment has "
                     "its own page table, giving the benefits of both: "
                     "meaningful logical division plus no external fragmentation "
                     "(only small internal fragmentation in the last page of "
                     "each segment)."),
                ],
                "negative": [{
                            "question": (
                                "If a program's logical address has an offset exactly equal to a "
                                "segment's limit value, is that a valid access?"
                            ),
                            "answer": (
                                "No. The rule stated is: if the offset is less than the limit, the "
                                "physical address is computed as base + offset; if the offset EXCEEDS "
                                "the limit, a segmentation fault occurs. An offset exactly equal to the "
                                "limit is neither 'less than' the limit, so it doesn't satisfy the valid "
                                "condition — since segment sizes/limits are typically defined so valid "
                                "offsets run from 0 up to (limit − 1), an offset equal to the limit "
                                "value itself is one byte past the last valid byte in the segment, and "
                                "would trigger a segmentation fault (trap) just as an offset that "
                                "exceeds the limit would."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Virtual Memory",
        "topics": [
            {
                "id": "virtual_memory_basics",
                "context": (
                    "Virtual memory is a technique that allows the execution of "
                    "processes that are not completely in main memory, giving "
                    "each process the illusion of a large, contiguous address "
                    "space backed by disk."
                ),
                "qas": [
                    ("What is virtual memory, and what are its main benefits?",
                     "Virtual memory separates a process's logical address space "
                     "(as seen by the program) from physical memory, allowing a "
                     "process to run even if its entire memory image doesn't fit "
                     "in RAM (only actively used parts are kept resident). "
                     "Benefits include supporting larger programs than physical "
                     "memory allows, increasing the degree of multiprogramming "
                     "(more processes fit since each uses less physical RAM), and "
                     "simplifying programming (no manual overlay management)."),
                    ("What is demand paging?",
                     "Demand paging is a virtual memory technique where a page "
                     "is only loaded into physical memory when it is actually "
                     "referenced (demanded) by the running process, rather than "
                     "loading the entire process at start. This reduces memory "
                     "usage and speeds up process start-up, since unused pages "
                     "(e.g. rarely-executed error-handling code) are never "
                     "loaded."),
                    ("What is a page fault, and what steps does the OS take to handle one?",
                     "A page fault is a trap that occurs when a process "
                     "references a page marked invalid (not currently in "
                     "physical memory) in its page table. The OS: (1) checks if "
                     "the reference is valid or an illegal access, (2) if valid, "
                     "finds a free frame (or evicts a page using a replacement "
                     "algorithm), (3) reads the required page in from disk into "
                     "that frame, (4) updates the page table to mark it valid and "
                     "record the frame number, and (5) restarts the instruction "
                     "that caused the fault."),
                ],
                "applied": [{
                            "question": (
                                "A process references a page that is marked invalid in its page table "
                                "(not currently in physical memory). Walk through what the OS does, in "
                                "order, referencing the page fault handling steps."
                            ),
                            "answer": (
                                "This triggers a page fault, a trap caused by referencing an invalid "
                                "page. The OS then: (1) checks whether the reference is valid or an "
                                "illegal access — assuming it's valid, (2) finds a free frame, or evicts "
                                "a page using a replacement algorithm if none is free, (3) reads the "
                                "required page in from disk into that frame, (4) updates the page table "
                                "to mark the page valid and record its new frame number, and (5) "
                                "restarts the very instruction that caused the fault, which now succeeds "
                                "since the page table correctly reflects that the page is in memory."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "page_replacement_conceptual",
                "context": (
                    "When a page fault occurs and no free frame is available, "
                    "the OS must select an existing page to evict from memory "
                    "using a page-replacement algorithm, such as FIFO, LRU, or "
                    "Optimal."
                ),
                "qas": [
                    ("Name and briefly describe three common page replacement algorithms.",
                     "FIFO (First-In-First-Out) evicts the page that has been in "
                     "memory the longest, regardless of usage — simple but can "
                     "perform poorly. LRU (Least Recently Used) evicts the page "
                     "that hasn't been referenced for the longest time, "
                     "approximating future behavior using past behavior — "
                     "generally performs well but needs extra hardware/software "
                     "to track recency. Optimal (also called MIN or Belady's "
                     "algorithm) evicts the page that will not be used for the "
                     "longest time in the future — it gives the theoretical "
                     "minimum number of page faults but requires future "
                     "knowledge, so it's used only as a benchmark, not in "
                     "practice."),
                    ("What is Belady's Anomaly?",
                     "Belady's Anomaly is the counter-intuitive phenomenon where, "
                     "for certain page-reference strings under the FIFO "
                     "replacement algorithm, increasing the number of available "
                     "frames actually increases the number of page faults, "
                     "instead of decreasing or staying the same. Stack-based "
                     "algorithms like LRU and Optimal never exhibit this "
                     "anomaly."),
                    ("What is thrashing, and what causes it?",
                     "Thrashing is a condition where a process (or system) "
                     "spends more time paging (swapping pages in and out) than "
                     "executing actual instructions, causing CPU utilization to "
                     "collapse. It is caused by over-committing memory — too many "
                     "processes running with too little physical memory, so each "
                     "process doesn't have enough frames to hold its actively "
                     "used pages (its 'working set'), causing continuous page "
                     "faults."),
                    ("What is the working set model, and how does it help prevent thrashing?",
                     "The working set of a process at time t is the set of pages "
                     "it has referenced in the most recent Δ time units (the "
                     "working-set window). The working set model allocates each "
                     "process enough frames to hold its entire working set; if "
                     "the total working sets of all active processes exceed "
                     "available memory, the OS suspends (swaps out) some "
                     "processes rather than letting all of them thrash, keeping "
                     "the remaining processes running smoothly."),
                    ("What is the difference between local and global page replacement?",
                     "In local replacement, a process can only replace pages "
                     "that belong to itself (from its own allocated frame set) "
                     "when it faults, ensuring predictable performance per "
                     "process but limiting its allocation to a fixed number of "
                     "frames. In global replacement, a process can select a "
                     "victim frame from any process in the system, which can "
                     "lead to better overall throughput but means one process's "
                     "paging behavior can affect the performance of unrelated "
                     "processes."),
                ],
                "negative": [{
                            "question": (
                                "If a system uses LRU or Optimal page replacement, and increases the "
                                "number of available frames, can the number of page faults ever "
                                "increase?"
                            ),
                            "answer": (
                                "No. Belady's Anomaly — where increasing the number of frames can "
                                "increase page faults instead of decreasing them — is specifically "
                                "stated to occur under FIFO replacement; 'stack-based algorithms like "
                                "LRU and Optimal never exhibit this anomaly.' For stack-based "
                                "algorithms, the set of pages held with N frames is always a subset of "
                                "what would be held with N+1 frames, so adding frames can only help or "
                                "leave the fault count unchanged, never make it worse. So for LRU or "
                                "Optimal specifically, increasing frames cannot increase the number of "
                                "page faults."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "File Systems",
        "topics": [
            {
                "id": "file_concept",
                "context": (
                    "A file is a named collection of related information "
                    "recorded on secondary storage. The operating system "
                    "provides a uniform logical view of storage by abstracting "
                    "the physical properties of storage devices into files."
                ),
                "qas": [
                    ("What are the common file access methods?",
                     "Sequential access (records/bytes are read or written in "
                     "order, one after another, like a tape), direct/random "
                     "access (records can be read or written in any order using "
                     "a fixed-length block/record number, like a disk), and "
                     "indexed access (an index maps keys to record locations, "
                     "enabling fast lookup of specific records without scanning "
                     "sequentially)."),
                    ("What are common directory structure types?",
                     "Single-level directory (all files in one flat list — "
                     "simple but causes naming conflicts), two-level directory "
                     "(a separate directory per user), tree-structured directory "
                     "(hierarchical directories and subdirectories, the most "
                     "common today), and acyclic-graph directory (allows shared "
                     "subdirectories/files via links, as long as no cycles are "
                     "created)."),
                ],
                "applied": [{
                            "question": (
                                "A phone book application needs to instantly jump to any specific "
                                "contact's record by contact ID, without scanning through unrelated "
                                "contacts first. Which file access method fits this requirement, and why "
                                "would sequential access be a poor fit?"
                            ),
                            "answer": (
                                "Direct/random access fits this requirement, since it allows records to "
                                "be read or written in any order using a fixed-length block/record "
                                "number, like a disk — a contact ID can be used directly to jump "
                                "straight to that record's location. Sequential access would be a poor "
                                "fit because it only supports reading or writing records in order, one "
                                "after another, like a tape — retrieving one specific contact by ID "
                                "would require reading through every preceding record first, which is "
                                "exactly the unrelated-scanning behavior the requirement rules out."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "file_allocation",
                "context": (
                    "File allocation methods determine how disk blocks are "
                    "allocated to files: contiguous allocation, linked "
                    "allocation, and indexed allocation, each with different "
                    "trade-offs for access speed and fragmentation."
                ),
                "qas": [
                    ("Compare contiguous, linked, and indexed file allocation methods.",
                     "Contiguous allocation stores each file in a set of "
                     "contiguous disk blocks — supports fast sequential and "
                     "direct access, but suffers from external fragmentation and "
                     "requires knowing the file size in advance. Linked "
                     "allocation stores each file as a linked list of disk "
                     "blocks scattered anywhere on disk — no external "
                     "fragmentation and files can grow easily, but only supports "
                     "efficient sequential access (direct access requires "
                     "traversing the list) and is vulnerable to pointer "
                     "corruption. Indexed allocation stores all block pointers "
                     "for a file in a separate index block — supports efficient "
                     "direct access without external fragmentation, but wastes "
                     "some space on the index block itself and may need multiple "
                     "levels of indexing for very large files."),
                    ("What is the FAT (File Allocation Table) method, and how does it relate to linked allocation?",
                     "FAT is a variation of linked allocation where, instead of "
                     "storing the 'next block' pointer inside each data block "
                     "itself, a separate table (the FAT) at the start of the "
                     "disk holds one entry per block, pointing to the next block "
                     "of each file. This keeps the pointer chain out of the data "
                     "blocks and allows the whole FAT to be cached in memory for "
                     "faster traversal, at the cost of needing to keep the FAT "
                     "itself safe and space for it."),
                    ("What is an inode, and what is its role in Unix-style file systems?",
                     "An inode (index node) is a data structure that stores all "
                     "metadata about a file — permissions, owner, size, "
                     "timestamps, and pointers to the data blocks (often direct "
                     "pointers plus one or more levels of indirect pointers for "
                     "large files) — but not the filename itself (filenames are "
                     "stored in directory entries that point to inode numbers)."),
                ],
                "negative": [{
                            "question": (
                                "Does linked allocation suffer from external fragmentation the way "
                                "contiguous allocation does?"
                            ),
                            "answer": (
                                "No. Linked allocation stores each file as a linked list of disk blocks "
                                "scattered anywhere on disk, so a file's blocks never need to occupy one "
                                "contiguous region — any free block anywhere can be linked into a file, "
                                "regardless of its neighbors. This is explicitly stated as an advantage: "
                                "linked allocation has 'no external fragmentation and files can grow "
                                "easily.' External fragmentation specifically arises from needing a "
                                "contiguous run of free space, which is a problem unique to allocation "
                                "schemes (like contiguous allocation) that require a file's blocks to "
                                "sit together — linked allocation avoids the requirement entirely, so "
                                "the problem doesn't apply to it."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "free_space_management",
                "context": (
                    "Free space on a disk must be tracked so the file system "
                    "knows which blocks are available for new allocations; "
                    "common techniques include bit vectors, linked lists, "
                    "grouping, and counting."
                ),
                "qas": [
                    ("What are the common techniques for tracking free disk space?",
                     "Bit vector/bitmap (one bit per block, 1 = free, 0 = "
                     "allocated — simple and compact, easy to find contiguous "
                     "free blocks), linked list of free blocks (each free block "
                     "holds a pointer to the next free block — no wasted space "
                     "for a bitmap but slow to traverse and search), grouping "
                     "(the first free block stores pointers to several other "
                     "free blocks, one of which stores pointers to further "
                     "blocks, speeding up finding multiple free blocks), and "
                     "counting (since blocks are often allocated/freed in "
                     "contiguous chunks, store the address of the first free "
                     "block in a run plus a count of how many follow)."),
                ],
                "analytical": [{
                            "question": (
                                "Why is a bit vector (bitmap) described as easy for finding contiguous "
                                "free blocks, while a linked list of free blocks is described as slow to "
                                "search?"
                            ),
                            "answer": (
                                "A bit vector represents the entire disk's free/allocated status "
                                "compactly as one bit per block, laid out in block order — scanning it "
                                "for a run of consecutive 1-bits (free blocks) can be done efficiently "
                                "in memory, often examining many blocks' status at once (e.g. a whole "
                                "word of bits together), directly revealing contiguous stretches of "
                                "free space. A linked list of free blocks, by contrast, only tells you "
                                "the very next free block via a pointer stored in the current one — "
                                "finding out whether several free blocks happen to be contiguous, or "
                                "finding a free block with any particular property, requires following "
                                "the chain one link at a time, with no way to jump ahead or examine "
                                "multiple candidates simultaneously, which is inherently slower to "
                                "traverse and search."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "file_system_extra",
                "context": (
                    "File systems are made available to users through mounting, "
                    "and their integrity must periodically be checked and "
                    "repaired using consistency-checking utilities."
                ),
                "qas": [
                    ("What does \"mounting\" a file system mean?",
                     "Mounting is the process of attaching a file system (from "
                     "a disk partition, removable device, or network share) to a "
                     "specific point (the mount point) within the existing "
                     "directory tree, making its files and directories "
                     "accessible as part of the unified file system hierarchy, "
                     "rather than as a separate, disconnected namespace."),
                    ("Why is file system consistency checking (e.g. fsck) necessary, and when is it typically run?",
                     "Because a system crash or power failure can occur while "
                     "file system metadata (directory entries, free-space lists, "
                     "inodes) is only partially updated, leaving it in an "
                     "inconsistent state (e.g. a block marked both free and "
                     "allocated). A consistency checker (like fsck on Unix or "
                     "chkdsk on Windows) compares the actual state of "
                     "blocks/directories against the metadata and repairs "
                     "inconsistencies, typically run automatically at boot after "
                     "an unclean shutdown, or manually on demand."),
                ],
                "negative": [{
                            "question": (
                                "After an unclean shutdown, if the file system consistency checker finds "
                                "no inconsistencies at all, does that mean the checker didn't need to "
                                "run?"
                            ),
                            "answer": (
                                "No. A consistency checker like fsck or chkdsk is 'typically run "
                                "automatically at boot after an unclean shutdown' specifically because a "
                                "crash or power failure CAN leave file system metadata in a partially-"
                                "updated, inconsistent state — but whether it actually did so on any "
                                "particular occasion isn't known in advance. The checker still needs to "
                                "run to verify the state and confirm nothing is inconsistent; finding no "
                                "problems is a valid, expected outcome of the check, not evidence that "
                                "the check was unnecessary. The check is what establishes that the "
                                "metadata is consistent — without running it, that fact simply isn't "
                                "known after an unclean shutdown."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Mass Storage and Disk Scheduling",
        "topics": [
            {
                "id": "disk_structure",
                "context": (
                    "Modern disks are addressed as large one-dimensional arrays "
                    "of logical blocks. Disk performance depends primarily on "
                    "seek time (moving the disk arm to the correct cylinder), "
                    "rotational latency (waiting for the desired sector to "
                    "rotate under the head), and transfer time."
                ),
                "qas": [
                    ("What are seek time, rotational latency, and transfer time in disk performance?",
                     "Seek time is the time for the disk arm to move the "
                     "read/write head to the cylinder (track) containing the "
                     "desired sector. Rotational latency is the time waiting for "
                     "the disk to rotate so the desired sector is under the "
                     "head. Transfer time is the time actually taken to transfer "
                     "the data once the head is correctly positioned. Total "
                     "access time = seek time + rotational latency + transfer "
                     "time."),
                    ("Why do disk-scheduling algorithms mainly try to minimize total seek time (head movement)?",
                     "Because seek time is typically the dominant, most "
                     "variable, and most controllable component of disk access "
                     "latency (rotational latency and transfer time are largely "
                     "fixed by the hardware's rotation speed and are largely "
                     "unaffected by request order). Minimizing the total "
                     "distance the disk head moves across all pending requests "
                     "directly reduces overall I/O completion time and increases "
                     "disk throughput."),
                ],
                "analytical": [{
                            "question": (
                                "Why does total disk access time = seek time + rotational latency + "
                                "transfer time, and why can't a disk begin transferring data before both "
                                "seek time and rotational latency have elapsed?"
                            ),
                            "answer": (
                                "Transferring the requested data requires the read/write head to be "
                                "correctly positioned over the specific bit of the disk holding that "
                                "data, which needs two separate physical alignments: the head must first "
                                "be over the correct cylinder/track (seek time), and then the disk must "
                                "rotate until the correct sector on that track passes under the head "
                                "(rotational latency). Only once both of these physical positioning "
                                "steps are complete can the head actually begin reading or writing the "
                                "requested bits (transfer time) — data transfer literally cannot start "
                                "correctly until the head is both radially and rotationally aligned with "
                                "the target sector, which is why the three components are sequential "
                                "contributors that sum together rather than overlapping."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "disk_scheduling_conceptual",
                "context": (
                    "Disk scheduling algorithms decide the order in which "
                    "pending I/O requests (each targeting a specific cylinder) "
                    "are serviced, aiming to minimize total head movement and/or "
                    "provide fairness."
                ),
                "qas": [
                    ("Explain how the SSTF (Shortest Seek Time First) disk scheduling algorithm works, and its main drawback.",
                     "SSTF services the pending request whose cylinder is "
                     "closest to the current head position next, similar to SJF "
                     "for CPU scheduling. Its main drawback is that it can cause "
                     "starvation of requests far from the current head position "
                     "if a steady stream of requests keeps arriving near the "
                     "head's current location."),
                    ("Explain how the SCAN (elevator) disk scheduling algorithm works.",
                     "The disk arm starts at one end of the disk and moves "
                     "toward the other end, servicing every pending request it "
                     "encounters along the way, then reverses direction at the "
                     "far end and services requests on the way back — like an "
                     "elevator. This bounds the maximum wait for any request and "
                     "gives more uniform service than SSTF."),
                    ("What is the difference between SCAN and C-SCAN (Circular SCAN) disk scheduling?",
                     "SCAN services requests while moving in one direction to "
                     "the end of the disk, then reverses and services requests "
                     "while moving back in the other direction (both directions "
                     "service requests). C-SCAN only services requests while "
                     "moving in one direction; when it reaches the end, it "
                     "immediately jumps back (without servicing) to the "
                     "beginning and starts scanning in the same direction again, "
                     "treating the disk as circular — this gives more uniform "
                     "wait times since it doesn't 'double service' the middle "
                     "cylinders like SCAN does."),
                    ("What is the difference between SCAN and LOOK (and C-SCAN vs C-LOOK)?",
                     "SCAN/C-SCAN always move the head all the way to the "
                     "physical end of the disk before reversing (or jumping "
                     "back), even if there are no requests out there. LOOK/"
                     "C-LOOK only go as far as the last request in the current "
                     "direction before reversing (or jumping back), avoiding "
                     "unnecessary head movement to the very edge of the disk "
                     "when no requests exist there — making them slightly more "
                     "efficient in practice."),
                    ("Compare FCFS and SSTF disk scheduling algorithms.",
                     "FCFS services disk requests strictly in arrival order "
                     "regardless of their cylinder location, which is fair and "
                     "starvation-free but can cause a lot of unnecessary head "
                     "movement (poor throughput). SSTF always services the "
                     "closest pending request to the current head position, "
                     "minimizing head movement in the short term and improving "
                     "throughput, but it can starve requests that are far from "
                     "the head if closer requests keep arriving."),
                ],
                "applied": [{
                            "question": (
                                "A disk head is positioned at cylinder 50. Requests arrive for cylinders "
                                "10, 90, and 55, in that arrival order. Using SSTF, which request is "
                                "serviced first, and why might a request like cylinder 10 risk "
                                "starvation under SSTF if a steady stream of nearby requests keeps "
                                "arriving?"
                            ),
                            "answer": (
                                "Under SSTF, the request whose cylinder is closest to the current head "
                                "position (50) is serviced next — cylinder 55 (distance 5) is closer "
                                "than cylinder 90 (distance 40) or cylinder 10 (distance 40), so "
                                "cylinder 55 is serviced first. Request 10's risk of starvation comes "
                                "from SSTF always favoring whatever is currently nearest to the head: if "
                                "new requests keep arriving near the head's current location (e.g. "
                                "clustered around cylinder 50-60), SSTF will keep servicing those "
                                "instead of ever traveling the longer distance out to cylinder 10, since "
                                "something closer is repeatedly available — exactly the drawback "
                                "stated: SSTF 'can cause starvation of requests far from the current "
                                "head position if a steady stream of requests keeps arriving near the "
                                "head's current location.'"
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "raid",
                "context": (
                    "RAID (Redundant Array of Independent Disks) uses multiple "
                    "physical disks to improve performance and/or reliability "
                    "through techniques such as striping, mirroring, and "
                    "parity."
                ),
                "qas": [
                    ("What is RAID, and what are its two main goals?",
                     "RAID (Redundant Array of Independent/Inexpensive Disks) "
                     "combines multiple physical disks into one logical unit to "
                     "improve performance (by striping data across disks for "
                     "parallel I/O) and/or reliability (by storing redundant "
                     "data — mirrors or parity — so the array survives one or "
                     "more disk failures without data loss)."),
                    ("Briefly describe RAID 0, RAID 1, and RAID 5.",
                     "RAID 0 (striping) splits data evenly across multiple "
                     "disks with no redundancy — maximizes performance and "
                     "capacity but offers zero fault tolerance (any single disk "
                     "failure loses all data). RAID 1 (mirroring) duplicates all "
                     "data identically on two (or more) disks — provides full "
                     "redundancy and fast reads but halves usable capacity. RAID "
                     "5 (striping with distributed parity) stripes data across "
                     "disks and stores a rotating parity block on each disk, "
                     "tolerating a single disk failure (data can be "
                     "reconstructed from parity) while using less redundant "
                     "capacity than mirroring."),
                ],
                "negative": [{
                            "question": (
                                "If a single disk fails in a RAID 0 array, can the lost data be "
                                "reconstructed from the remaining disks?"
                            ),
                            "answer": (
                                "No. RAID 0 (striping) splits data evenly across multiple disks 'with no "
                                "redundancy' — it stores no mirrored copies and no parity information "
                                "anywhere. It 'maximizes performance and capacity but offers zero fault "
                                "tolerance (any single disk failure loses all data).' Since there is no "
                                "redundant data of any kind stored on the other disks to reconstruct "
                                "from, a single disk failure in RAID 0 causes permanent, unrecoverable "
                                "data loss for the portions of every file striped across the failed "
                                "disk."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
            {
                "id": "disk_extra",
                "context": (
                    "Booting a computer and handling defective disk sectors are "
                    "low-level responsibilities closely tied to disk and mass "
                    "storage management."
                ),
                "qas": [
                    ("What is the boot block, and how does the system boot?",
                     "The boot block is a small, fixed location on a bootable "
                     "disk (typically the first sector) containing a bootstrap "
                     "loader program. When the computer is powered on, the "
                     "ROM-resident bootstrap code loads and executes this "
                     "bootstrap loader, which in turn locates and loads the full "
                     "operating system kernel into memory and transfers control "
                     "to it."),
                    ("How does an OS typically handle bad blocks/sectors on a disk?",
                     "Modern disk controllers maintain a list of bad "
                     "(defective) sectors and transparently remap them to spare "
                     "sectors set aside for this purpose during low-level "
                     "formatting, so the OS and file system usually never see "
                     "the bad sectors at all (sector sparing/forwarding). Older "
                     "or simpler systems handled this at the OS level, marking "
                     "bad blocks in the file system's free-space structures so "
                     "they are never allocated to a file."),
                ],
                "analytical": [{
                            "question": (
                                "Why does sector sparing (transparently remapping bad sectors to spares "
                                "during low-level formatting) mean the OS and file system usually never "
                                "need to handle bad blocks themselves?"
                            ),
                            "answer": (
                                "Sector sparing is handled entirely inside the disk controller, below "
                                "the level the OS or file system ever interacts with: the controller "
                                "maintains its own list of defective sectors and transparently redirects "
                                "any access meant for a bad sector to a spare sector set aside during "
                                "low-level formatting. Because this remapping happens automatically at "
                                "the hardware level before a logical block number is even translated to "
                                "a physical location, the OS and file system continue addressing the "
                                "disk using its normal logical block numbers with no awareness that some "
                                "of the underlying physical sectors are actually spares standing in for "
                                "defective ones — the substitution is invisible above the controller, "
                                "which is exactly why 'the OS and file system usually never see the bad "
                                "sectors at all.'"
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "I/O Systems",
        "topics": [
            {
                "id": "io_hardware",
                "context": (
                    "The I/O subsystem manages communication between the CPU "
                    "and peripheral devices using techniques such as polling, "
                    "interrupts, DMA, spooling, buffering, and caching."
                ),
                "qas": [
                    ("What is the difference between polling and interrupt-driven I/O?",
                     "Polling has the CPU repeatedly check (busy-wait on) a "
                     "device's status register to see if it's ready, wasting CPU "
                     "cycles while waiting. Interrupt-driven I/O lets the CPU do "
                     "other useful work and only responds when the device "
                     "signals completion via a hardware interrupt, which is far "
                     "more efficient for most devices, though polling can be "
                     "faster for very quick, predictable operations since it "
                     "avoids interrupt overhead."),
                    ("What is DMA (Direct Memory Access), and why is it used?",
                     "DMA is a hardware mechanism that allows an I/O device (via "
                     "a DMA controller) to transfer data directly to/from main "
                     "memory without the CPU being involved in each individual "
                     "byte transfer. The CPU only sets up the transfer (source, "
                     "destination, length) and is interrupted once when the "
                     "entire transfer completes, freeing it from the overhead of "
                     "moving large blocks of data itself."),
                    ("What is spooling, and why is it used?",
                     "Spooling (Simultaneous Peripheral Operations On-Line) "
                     "stores data destined for a slow device (like a printer) in "
                     "a disk buffer/queue, allowing multiple processes to "
                     "'print' without waiting for the physical device or for "
                     "each other, and letting a scheduler manage the order of "
                     "dispatch. It decouples fast processes from slow I/O "
                     "devices and enables device sharing among multiple users."),
                    ("What is the difference between buffering and caching in I/O systems?",
                     "Buffering temporarily holds data during transfer between "
                     "two devices/processes with different speeds or "
                     "data-transfer sizes (e.g. smoothing out data flow between "
                     "a producer and consumer). Caching keeps a copy of data "
                     "that already exists elsewhere (typically slower storage) "
                     "in a faster area, purely to speed up repeated access to "
                     "that same data — a buffer may sometimes serve double duty "
                     "as a cache if it holds the only copy of data currently in "
                     "use."),
                ],
                "applied": [{
                            "question": (
                                "A device transfers a very large file (megabytes) into memory. Using "
                                "DMA, describe what the CPU actually does during this transfer, and "
                                "contrast it with what the CPU would have to do without DMA."
                            ),
                            "answer": (
                                "With DMA, the CPU only sets up the transfer — specifying the source, "
                                "destination, and length — and then is free to do other useful work "
                                "while a DMA controller handles moving the data directly between the "
                                "device and main memory. The CPU is interrupted only once, when the "
                                "entire transfer completes. Without DMA, the CPU would need to be "
                                "involved in each individual byte transfer itself, which for a "
                                "multi-megabyte file means an enormous number of individual "
                                "CPU-mediated transfers, consuming CPU cycles that could otherwise be "
                                "spent on other work — exactly the overhead DMA is designed to eliminate "
                                "for large data transfers."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Protection and Security",
        "topics": [
            {
                "id": "protection_security",
                "context": (
                    "Protection mechanisms control access of processes and "
                    "users to system resources according to policy, while "
                    "security defends the whole system against internal and "
                    "external threats."
                ),
                "qas": [
                    ("What is the difference between protection and security in an operating system?",
                     "Protection refers to internal mechanisms that control "
                     "access of processes/users to resources within a system as "
                     "defined by policy (e.g. file permissions, access control "
                     "lists) — it deals with legitimate users misusing their "
                     "granted access. Security is a broader concern that also "
                     "defends the system against external threats — "
                     "unauthorized/malicious access, viruses, worms, "
                     "denial-of-service attacks — from both inside and outside "
                     "the system."),
                    ("What is an access matrix, and what does it represent?",
                     "An access matrix is a conceptual model representing the "
                     "protection state of a system as rows (domains/subjects, "
                     "e.g. users or processes) and columns (objects, e.g. files "
                     "or devices), where each cell lists the operations (read, "
                     "write, execute, etc.) that domain is permitted to perform "
                     "on that object."),
                    ("Differentiate between authentication and authorization.",
                     "Authentication verifies who a user/process actually is "
                     "(e.g. via passwords, biometrics, tokens). Authorization "
                     "determines what an already-authenticated user/process is "
                     "allowed to do — which resources and operations they are "
                     "permitted to access — typically enforced afterward via "
                     "access control mechanisms like an access matrix or ACLs."),
                    ("What is the principle of least privilege?",
                     "The principle of least privilege states that every "
                     "process, user, or program should be granted only the "
                     "minimum set of access rights/permissions necessary to "
                     "perform its intended task, and no more. This limits the "
                     "potential damage from bugs, misuse, or compromise, since a "
                     "component with limited privileges can do limited harm even "
                     "if exploited."),
                ],
                "negative": [{
                            "question": (
                                "If a legitimate, authenticated user misuses file permissions they were "
                                "correctly granted (e.g. reading a file they're allowed to read, but for "
                                "a purpose the owner didn't intend), is that a protection failure or a "
                                "security failure by the definitions given?"
                            ),
                            "answer": (
                                "It's a protection concern, not necessarily a security failure by these "
                                "definitions. Protection is described as dealing with 'legitimate users "
                                "misusing their granted access' — internal mechanisms controlling access "
                                "according to policy. Security is described as defending against threats "
                                "that are unauthorized or malicious, from both inside and outside the "
                                "system — but here, the user IS authorized (correctly authenticated, "
                                "correctly granted read access) and isn't described as acting "
                                "maliciously or without authorization; they're simply using a "
                                "legitimately-granted permission in a way the file owner didn't "
                                "anticipate. Since the access itself was authorized and within the "
                                "granted permissions, this scenario is squarely the protection concern "
                                "the definitions describe, rather than the unauthorized/malicious access "
                                "security is defined around."
                            ),
                            "difficulty": "hard",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
    # ══════════════════════════════════════════════════════════════════
    {
        "chapter": "Virtualization",
        "topics": [
            {
                "id": "virtualization",
                "context": (
                    "Virtualization allows a single physical machine to host "
                    "multiple isolated virtual machines, each running its own "
                    "guest operating system, managed by a hypervisor."
                ),
                "qas": [
                    ("What is a virtual machine, and what is a hypervisor?",
                     "A virtual machine (VM) is a software-based emulation of a "
                     "complete physical computer, allowing an entire guest "
                     "operating system to run isolated on top of a host system "
                     "as if it had its own dedicated hardware. A hypervisor "
                     "(Virtual Machine Monitor) is the software layer that "
                     "creates and manages VMs, allocating and multiplexing the "
                     "underlying physical hardware resources (CPU, memory, I/O) "
                     "among them."),
                    ("What is the difference between a Type 1 and a Type 2 hypervisor?",
                     "A Type 1 (bare-metal) hypervisor runs directly on the "
                     "physical hardware, with no underlying host OS — it is the "
                     "'OS' for VMs (e.g. VMware ESXi, Microsoft Hyper-V), "
                     "generally giving better performance. A Type 2 (hosted) "
                     "hypervisor runs as an application on top of a conventional "
                     "host operating system (e.g. VirtualBox, VMware Workstation "
                     "running on Windows/Linux), which is easier to set up but "
                     "adds an extra layer of overhead."),
                ],
                "applied": [{
                            "question": (
                                "A cloud provider wants to run many customers' virtual machines on bare "
                                "physical servers with maximum performance, with no separate host "
                                "operating system involved at all. Which hypervisor type fits this "
                                "requirement, and why would the other type be a worse fit?"
                            ),
                            "answer": (
                                "A Type 1 (bare-metal) hypervisor fits this requirement, since it runs "
                                "directly on the physical hardware with no underlying host OS — it IS "
                                "the 'OS' for the VMs, and is generally described as giving better "
                                "performance, matching the goal of maximum performance with no separate "
                                "host OS. A Type 2 (hosted) hypervisor would be a worse fit here because "
                                "it runs as an application on top of a conventional host operating "
                                "system, meaning there IS a separate host OS layer between the hardware "
                                "and the hypervisor — directly contradicting the stated requirement, and "
                                "adding the extra overhead that comes with that additional layer."
                            ),
                            "difficulty": "medium",
                            "status": "REVIEWED",
                        }],
            },
        ],
    },
]


# ══════════════════════════════════════════════════════════════════════
# GLOSSARY — short, accurate one-line definitions of key OS terms.
# Used to generate multiple short-definition examples with varied phrasing.
# ══════════════════════════════════════════════════════════════════════
GLOSSARY = [
    ("Busy waiting", "A situation where a process repeatedly checks a condition in a loop (spinning) while waiting for it to become true, wasting CPU cycles instead of blocking."),
    ("Spooling", "Simultaneous Peripheral Operations On-Line — buffering output for a slow device (like a printer) on disk so processes don't have to wait for the device directly."),
    ("Virtual machine", "A software-based emulation of a physical computer that lets a guest operating system run isolated on top of a host system's hardware."),
    ("Zombie process", "A process that has finished execution but still has an entry in the process table because its parent has not yet called wait() to collect its exit status."),
    ("Orphan process", "A process whose parent process has terminated before it did; it is typically re-parented to the init process."),
    ("Daemon process", "A background process that runs without direct user interaction, typically providing a system service (e.g. a print spooler or web server)."),
    ("Interrupt vector table", "A table of pointers (addresses) to interrupt service routines, indexed by interrupt number, used by the CPU to quickly locate the correct handler for an interrupt."),
    ("Bootstrap program", "A small initial program stored in ROM/firmware that runs when a computer powers on and loads the operating system kernel into memory."),
    ("Fragmentation", "Wasted memory that accumulates over time due to allocation and deallocation patterns; can be internal (wasted space inside an allocated block) or external (wasted space between allocated blocks)."),
    ("Aging", "A technique that gradually increases the priority of a process the longer it waits in the ready queue, to prevent starvation."),
    ("Convoy effect", "A phenomenon in FCFS scheduling where a long CPU-bound process causes many shorter processes behind it to wait an unnecessarily long time."),
    ("Starvation", "A situation where a process is perpetually denied the resources it needs to proceed, often because other processes are repeatedly given priority."),
    ("Livelock", "A situation where two or more processes continuously change their state in response to each other without making any real progress, unlike deadlock where they are simply blocked."),
    ("Mutual exclusion", "The requirement that only one process/thread can execute in a critical section (accessing a shared resource) at any given time."),
    ("Dirty bit (modified bit)", "A page table entry flag indicating that a page has been written to since it was loaded, so it must be written back to disk before being evicted."),
    ("Reference bit", "A page table entry flag set by hardware whenever a page is accessed, used by approximation algorithms like Second-Chance/Clock to estimate recent usage for page replacement."),
    ("Valid-invalid bit", "A page table entry flag indicating whether the associated page is currently loaded in physical memory (valid) or not (invalid, causing a page fault if referenced)."),
    ("Memory Management Unit (MMU)", "The hardware component that translates logical (virtual) addresses generated by the CPU into physical addresses in main memory at run time."),
    ("Trap", "A software-generated interrupt caused either by an error (e.g. division by zero) or a deliberate user program request such as a system call."),
    ("Cache coherency", "The property that all processors/caches in a system see a consistent, up-to-date view of shared memory data despite each holding local cached copies."),
    ("Compaction", "The technique of relocating allocated memory blocks to be contiguous, consolidating all free memory into a single block to eliminate external fragmentation."),
    ("Multiprogramming", "Keeping multiple jobs in main memory simultaneously so the CPU can switch to another job whenever the current one needs to wait for I/O, maximizing CPU utilization."),
    ("Time-sharing", "A technique that rapidly switches the CPU between multiple users' processes, giving each the illusion of a dedicated, responsive system."),
    ("File descriptor", "A small non-negative integer that a process uses as a handle to refer to an open file or other I/O resource when making system calls."),
    ("Mount point", "The directory within an existing file system hierarchy where a separate file system (from a disk, partition, or network share) is attached and made accessible."),
    ("Working set", "The set of pages a process has referenced within a recent fixed time window, used to estimate the minimum number of frames it needs to run without excessive faulting."),
    ("Belady's Anomaly", "The counter-intuitive phenomenon where, under FIFO page replacement, increasing the number of available memory frames can increase (rather than decrease) the number of page faults."),
    ("Thrashing", "A state where a system spends more time swapping/paging than executing actual process instructions, caused by insufficient physical memory for the active working sets."),
    ("Priority inversion", "A scheduling anomaly where a higher-priority process is indirectly delayed by a lower-priority process holding a resource it needs, while an unrelated medium-priority process runs in between."),
    ("Race condition", "A flaw where the outcome of concurrent execution depends on the unpredictable timing/order of accesses to shared data by multiple processes or threads."),
]


# ══════════════════════════════════════════════════════════════════════
# COMPARISONS — cross-topic COMPARATIVE questions. Each entry references two
# existing topic ids (topic_a/topic_b, looked up by build_comparative_
# examples() in build_conceptual.py, from either kb_conceptual.py or
# kb_advanced.py's CHAPTERS). Both topics' context passages are concatenated
# into the "Textbook passages:" block, and the answer must be grounded ONLY
# in facts drawn from those two passages — no third-party facts invented.
# ══════════════════════════════════════════════════════════════════════
COMPARISONS = [
    {
        "topic_a": "critical_section",
        "topic_b": "deadlock_conditions",
        "question": (
            "How does a race condition differ from a deadlock in terms of the "
            "necessary conditions that produce it?"
        ),
        "answer": (
            "A race condition needs only concurrent, unsynchronized access to shared "
            "data where at least one access is a write — no locking or resource-holding "
            "is required at all for it to occur; the bug is simply the ABSENCE of mutual "
            "exclusion around a critical section, so the final result depends on "
            "whatever order the accesses happen to interleave in. A deadlock is the "
            "opposite kind of failure: it requires synchronization (resource-holding) to "
            "already be in place, and specifically requires four conditions to hold "
            "simultaneously — mutual exclusion, hold-and-wait, no preemption, and "
            "circular wait — so that a set of processes end up permanently blocked "
            "waiting on each other. In short: a race condition is what can go wrong when "
            "there is too LITTLE synchronization around shared data, while a deadlock is "
            "what can go wrong when there is synchronization (locks/resources being "
            "held), but it is acquired and ordered in a way that creates a circular "
            "waiting dependency. Fixing a race condition means ADDING mutual exclusion "
            "around the critical section; fixing/avoiding a deadlock means removing (or "
            "carefully managing) one of the four Coffman conditions that mutual-exclusion "
            "-style locking can itself introduce."
        ),
        "difficulty": "medium",
    },
    {
        "topic_a": "semaphores",
        "topic_b": "deadlock_conditions",
        "question": (
            "How can improper semaphore usage lead to a deadlock, and which of the four "
            "necessary deadlock conditions does a semaphore create by construction?"
        ),
        "answer": (
            "A semaphore used as a lock (initialized to 1) inherently creates the Mutual "
            "Exclusion condition by design — it restricts a resource to being held by only "
            "one process at a time, exactly what wait()/signal() are built to enforce. "
            "Deadlock arises on top of this when multiple semaphores are acquired in "
            "inconsistent order across different processes: for example, if Thread A "
            "executes wait(S1) then wait(S2), while Thread B executes wait(S2) then "
            "wait(S1), and both threads execute their first wait() at nearly the same "
            "time, Thread A ends up holding S1 while waiting for S2, and Thread B ends up "
            "holding S2 while waiting for S1. This single scenario satisfies all four "
            "necessary conditions at once: Mutual Exclusion (each semaphore is held "
            "exclusively), Hold-and-Wait (each thread holds one semaphore while waiting "
            "for another), No Preemption (neither semaphore can be forcibly taken away), "
            "and Circular Wait (A waits on B's semaphore, B waits on A's) — so the two "
            "threads deadlock permanently. The standard prevention technique is to impose "
            "a consistent global ordering on semaphore acquisition (e.g. always acquire "
            "S1 before S2, in every thread), which eliminates the circular-wait condition."
        ),
        "difficulty": "hard",
    },
    {
            "topic_a": "fcfs",
            "topic_b": "sjf",
            "question": (
                "Compare FCFS and SJF scheduling in terms of what each needs to know about "
                "a process in advance, and what problem that creates for SJF."
            ),
            "answer": (
                "FCFS needs no advance knowledge about a process at all beyond its arrival "
                "time — it maintains a simple FIFO queue of processes ordered by arrival "
                "time, and whichever arrives first runs first, regardless of how long it "
                "will take. SJF, by contrast, needs to know the length of each process's "
                "next CPU burst in advance in order to select the shortest one — but this "
                "is generally not knowable exactly for real workloads, which is stated as "
                "SJF's main practical limitation; it is approximated in practice using "
                "techniques like exponential averaging of past burst times. So while SJF is "
                "provably optimal for minimizing average waiting time, it pays for that "
                "optimality with a predictive requirement FCFS never has to satisfy at all."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "round_robin",
            "topic_b": "priority_scheduling",
            "question": (
                "Compare Round Robin and priority scheduling in terms of how each achieves "
                "(or fails to achieve) fairness among processes."
            ),
            "answer": (
                "Round Robin achieves fairness by construction: every process in the ready "
                "queue is given a fixed time quantum in turn, cycling through the circular "
                "queue, so every process gets a fair, bounded share of CPU time repeatedly "
                "regardless of when it arrived. Priority scheduling has no such built-in "
                "fairness — the CPU is allocated to whichever process has the highest "
                "priority, which means a low-priority process can be passed over "
                "indefinitely if higher-priority processes keep arriving, causing "
                "starvation. Priority scheduling only regains a form of fairness through an "
                "added mechanism, aging, which artificially increases a waiting process's "
                "priority over time until it eventually gets scheduled — fairness that "
                "Round Robin provides structurally, without needing a separate corrective "
                "mechanism."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "paging",
            "topic_b": "segmentation",
            "question": (
                "Compare paging and segmentation in terms of what determines the size of "
                "each allocation unit, and how that choice affects external fragmentation."
            ),
            "answer": (
                "In paging, the allocation unit (a page/frame) has a fixed size chosen by "
                "the system design, unrelated to the logical structure of any particular "
                "program — because every page is the same size and any page can be placed "
                "in any available frame, a process's memory need not be contiguous, which "
                "eliminates external fragmentation entirely (though it can still cause "
                "internal fragmentation in the last page). In segmentation, the allocation "
                "unit (a segment) has a variable size that matches a logically meaningful "
                "part of the program itself — code, data, stack, each with its own base and "
                "limit — which matches the programmer's logical view of the program, but "
                "because segments vary in size and must each occupy contiguous memory, "
                "segmentation can suffer from external fragmentation the way paging cannot. "
                "The fixed-vs-variable sizing choice is exactly what produces this "
                "difference in fragmentation behavior."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "deadlock_handling",
            "topic_b": "bankers_conceptual",
            "question": (
                "How does the Banker's algorithm specifically implement the 'deadlock "
                "avoidance' strategy described among the four general strategies for "
                "handling deadlocks?"
            ),
            "answer": (
                "Deadlock avoidance is described as dynamically checking whether granting a "
                "resource request could lead to an unsafe state before granting it, giving "
                "the Banker's algorithm as the example. The Banker's algorithm implements "
                "this exactly: it requires each process's Maximum claim, current "
                "Allocation, and Need (Max − Allocation), plus the Available vector; before "
                "granting any request, it tentatively allocates the requested resources and "
                "runs the safety algorithm to check whether the resulting state is still "
                "safe — a safe sequence must exist in which every process's remaining Need "
                "can eventually be satisfied. If the resulting state is safe, the request is "
                "granted; otherwise, the process must wait. This is precisely the 'dynamic "
                "check before granting' that defines avoidance as a strategy, made concrete "
                "with the specific data (Max/Allocation/Need/Available) and the specific "
                "safety-check procedure the Banker's algorithm provides."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "contiguous_allocation",
            "topic_b": "paging",
            "question": (
                "Compare contiguous allocation and paging in terms of how each handles (or "
                "fails to handle) external fragmentation, and what remedy each relies on."
            ),
            "answer": (
                "Contiguous allocation requires each process to occupy a single contiguous "
                "block of memory; over time, as processes come and go, free memory becomes "
                "scattered into small non-contiguous holes even when the total free space "
                "would be enough — this is external fragmentation, and its stated remedy is "
                "compaction: shuffling memory contents to relocate allocated blocks "
                "together and consolidate free memory into one large block, at the expense "
                "of the overhead of dynamic relocation and copying live process memory. "
                "Paging avoids the problem structurally instead of remedying it after the "
                "fact: because logical memory is divided into fixed-size pages that can be "
                "placed in any available physical frame, a process's memory never needs to "
                "be contiguous in physical RAM, so external fragmentation is eliminated "
                "entirely — paging needs no compaction-style remedy because the problem "
                "compaction solves for contiguous allocation doesn't arise for it in the "
                "first place."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "sjf",
            "topic_b": "round_robin",
            "question": (
                "Compare SJF and Round Robin in terms of which specific scheduling "
                "criterion each is designed to optimize."
            ),
            "answer": (
                "SJF is designed to optimize average waiting time — it is stated to be "
                "'provably optimal for minimizing average waiting time for a given set of "
                "processes,' by always running whichever ready process has the smallest "
                "burst time next, so other jobs spend the least total time waiting behind "
                "long ones. Round Robin is designed to optimize response time and fairness "
                "for interactive systems, not minimum average waiting time — it 'suits "
                "time-sharing systems because every process gets a fair, bounded share of "
                "CPU time repeatedly, giving good response time to interactive users even "
                "under heavy load,' at the cost of not necessarily minimizing waiting time "
                "the way SJF does. The two algorithms optimize for genuinely different "
                "criteria: SJF for the shortest possible average wait assuming burst times "
                "are known, and RR for consistently good responsiveness across many "
                "processes regardless of burst length."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "disk_scheduling_conceptual",
            "topic_b": "sjf",
            "question": (
                "How is SSTF disk scheduling analogous to SJF CPU scheduling, and how are "
                "their respective drawbacks similar?"
            ),
            "answer": (
                "SSTF disk scheduling is explicitly described as being 'similar to SJF for "
                "CPU scheduling': just as SJF always selects the ready process with the "
                "smallest CPU burst time next, SSTF always services the pending disk "
                "request whose cylinder is closest to the current head position next — both "
                "algorithms greedily pick whatever looks 'cheapest' to service right now, "
                "based on a size/distance measure, rather than respecting arrival order. "
                "Their drawbacks are similarly shaped, too: SJF's short jobs can "
                "indefinitely delay a long-burst process if shorter jobs keep arriving "
                "(addressed via aging), and SSTF 'can cause starvation of requests far from "
                "the current head position if a steady stream of requests keeps arriving "
                "near the head's current location' — in both cases, the greedy 'shortest/"
                "closest first' strategy can starve whichever item is unlucky enough to "
                "look expensive relative to a constant stream of cheaper alternatives."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "deadlock_conditions",
            "topic_b": "bankers_conceptual",
            "question": (
                "How does a cycle in the Resource Allocation Graph relate to the Banker's "
                "algorithm's concept of a 'safe state' — are they describing the same "
                "thing?"
            ),
            "answer": (
                "They are related but not the same thing, and they come from two different "
                "approaches to the deadlock problem. A Resource Allocation Graph cycle is a "
                "DETECTION-style concept: if the graph contains a cycle and each resource in "
                "it has only one instance, a deadlock EXISTS — it's a way of recognizing a "
                "deadlock that has already happened, and with multiple instances a cycle is "
                "necessary but not sufficient for deadlock. A safe state is an AVOIDANCE-"
                "style concept: it asks whether a safe sequence exists in which every "
                "process's remaining Need can eventually be satisfied, checked BEFORE "
                "granting a request, specifically so a deadlock never gets the chance to "
                "occur in the first place. So a RAG cycle is a symptom examined after (or "
                "as) resources are actually allocated, while a safe state is a predictive "
                "check performed before a request is granted — one detects, the other "
                "prevents."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "memory_basics",
            "topic_b": "virtual_memory_basics",
            "question": (
                "Compare swapping and demand paging as two different techniques for "
                "handling processes that don't fully fit in available physical memory."
            ),
            "answer": (
                "Swapping operates at whole-process granularity: it temporarily moves a "
                "process's ENTIRE memory image out to secondary storage (a backing store) "
                "to free up main memory for other processes, later swapping the whole thing "
                "back in to continue execution — it increases the degree of "
                "multiprogramming but adds significant I/O overhead, since the full image "
                "moves at once. Demand paging operates at per-page granularity instead: "
                "rather than moving whole processes, it loads (or evicts) individual pages "
                "only when actually referenced (demanded) by the running process, meaning a "
                "process's rarely-used parts (like error-handling code) may never be loaded "
                "at all. Both techniques let the system handle more total memory demand "
                "than physical RAM alone could hold, but swapping does so by relocating "
                "entire processes wholesale, while demand paging does so by keeping only "
                "the actively-used slice of each process resident at any given time."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
    {
            "topic_a": "peterson",
            "topic_b": "semaphores",
            "question": (
                "Compare Peterson's solution and semaphores in terms of how many processes "
                "each is described as handling, and what that implies about their "
                "practical scope."
            ),
            "answer": (
                "Peterson's solution is specifically described as 'a classic software-based "
                "algorithm for two processes,' using exactly two shared variables (an "
                "integer turn and a boolean array flag[2]) sized for exactly that case — "
                "its construction is inherently tied to coordinating two participants. "
                "Semaphores, by contrast, are described more generally as being 'used to "
                "control access to shared resources among concurrent processes,' with "
                "counting semaphores specifically supporting 'a resource with multiple "
                "identical instances' — nothing in their description limits them to two "
                "processes. This implies a real difference in practical scope: Peterson's "
                "solution as described is a two-process construction, while semaphores are "
                "described as a general-purpose mechanism usable across an arbitrary number "
                "of concurrent processes and resource instances."
            ),
            "difficulty": "hard",
            "status": "REVIEWED",
        },
]
