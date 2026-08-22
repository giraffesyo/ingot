// Package par provides a persistent worker pool and a parallel-for used by
// kernels.
//
// Design:
//   - MaxWorkers-1 helper goroutines are started lazily on first use and live
//     for the life of the process. The calling goroutine always participates
//     (worker id 0), so a region never waits on a goroutine spawn.
//   - Helpers spin for a short while (SpinNS) watching a generation counter
//     before parking, so back-to-back regions — the normal case during
//     inference — hand off in ~100ns instead of paying a thread wake-up (tens
//     of µs on macOS). Spinning touches only atomics: no channel or scheduler
//     locks, which at 16+ helpers otherwise collapse into runtime mutex
//     contention (runtime.lock2/usleep dominated profiles before this design).
//   - A region publishes its job through an atomic pointer + generation bump;
//     only helpers that have parked are woken (buffered channel). A helper that
//     picks up a job after its region has closed just drops it. The caller
//     never blocks on a helper that has not started, so nested and concurrent
//     Run calls cannot deadlock — the caller simply does more of the work
//     itself.
//   - Indices are distributed dynamically via an atomic counter in chunks of
//     `grain`, which balances uneven work and heterogeneous cores.
//
// Run with a pooled *Task allocates nothing; For with a closure allocates the
// closure (use Run in hot kernels).
package par

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// MaxWorkers is the total number of workers (including the caller). It is read
// once, when the pool starts on first use.
var MaxWorkers = runtime.GOMAXPROCS(0)

// SpinNS is how long an idle helper spin-polls before parking.
var SpinNS int64 = 50_000

// Task is a unit of parallel work. Run(i, w) is called for each index i in
// [0,n) by worker w. Implementations must be safe for concurrent calls.
type Task interface {
	Run(i, w int)
}

// Func adapts a closure to Task.
type Func func(i, w int)

// Run implements Task.
func (f Func) Run(i, w int) { f(i, w) }

type job struct {
	n, grain int
	task     Task
	next     atomic.Int64
	closed   atomic.Bool
	active   atomic.Int32 // helpers currently running this job
}

var (
	once     sync.Once
	nworkers int
	jobPool  = sync.Pool{New: func() any { return new(job) }}

	cur    atomic.Pointer[job] // most recently published job (nil once closed)
	gen    atomic.Uint64       // bumped on every publish; helpers spin on it
	parked atomic.Int32        // helpers blocked on wake
	wake   chan struct{}       // one token per parked helper to wake
)

func start() {
	nworkers = max(MaxWorkers, 1)
	wake = make(chan struct{}, nworkers)
	for w := 1; w < nworkers; w++ {
		go worker(w)
	}
}

// spinCheck is how many spin iterations pass between clock reads.
const spinCheck = 1024

func worker(w int) {
	var seen uint64
	for {
		// Wait for a new generation: spin on the atomic, then park.
		if gen.Load() == seen {
			deadline := time.Now().UnixNano() + SpinNS
			spins := 0
			for gen.Load() == seen {
				spins++
				if spins%spinCheck != 0 {
					continue
				}
				if time.Now().UnixNano() > deadline {
					parked.Add(1)
					if gen.Load() == seen {
						<-wake
					}
					parked.Add(-1)
					break
				}
			}
			continue // re-check gen (wake may be spurious)
		}
		seen = gen.Load()
		j := cur.Load()
		if j == nil || j.closed.Load() {
			continue
		}
		// Claim: bump active, then re-check closed (see Run for the protocol).
		j.active.Add(1)
		if j.closed.Load() {
			j.active.Add(-1)
			continue
		}
		j.run(w)
		j.active.Add(-1)
	}
}

func (j *job) run(w int) {
	for {
		start := int(j.next.Add(int64(j.grain))) - j.grain
		if start >= j.n {
			return
		}
		end := min(start+j.grain, j.n)
		for i := start; i < end; i++ {
			j.task.Run(i, w)
		}
	}
}

// Workers returns the pool size (caller + helpers). Valid worker ids passed to
// Task.Run are in [0, Workers()).
func Workers() int {
	once.Do(start)
	return nworkers
}

// For runs fn(i, w) for i in [0,n); see Run. The closure escapes (one
// allocation per call) — prefer Run with a pooled Task in hot paths.
func For(n, grain int, fn func(i, w int)) { Run(n, grain, Func(fn)) }

// Run calls t.Run(i, w) for i in [0,n), where w is the id of the worker running
// it (0 is the caller). Consecutive indices are handed out in chunks of grain.
// If the region would use a single worker, t runs inline on the caller.
// Passing a pointer-typed Task allocates nothing.
//
// Choose grain so that one chunk is at least a few microseconds of work; a
// region with a single chunk never leaves the caller.
func Run(n, grain int, t Task) {
	if n <= 0 {
		return
	}
	if grain < 1 {
		grain = 1
	}
	once.Do(start)
	tasks := (n + grain - 1) / grain
	helpers := min(nworkers, tasks) - 1
	if helpers <= 0 {
		for i := 0; i < n; i++ {
			t.Run(i, 0)
		}
		return
	}
	j := jobPool.Get().(*job)
	j.n, j.grain, j.task = n, grain, t
	j.next.Store(0)
	j.closed.Store(false)
	// Publish: spinning helpers see the generation change and pick up cur;
	// parked helpers (if any) get one wake token each, up to the number of
	// helpers this region can use.
	cur.Store(j)
	gen.Add(1)
	if p := int(parked.Load()); p > 0 {
		for i := 0; i < min(p, helpers); i++ {
			select {
			case wake <- struct{}{}:
			default:
			}
		}
	}
	j.run(0)
	// Close the job: helpers that have not claimed it yet will drop it; wait
	// for those that have.
	j.closed.Store(true)
	cur.CompareAndSwap(j, nil)
	for spins := 0; j.active.Load() != 0; spins++ {
		if spins%64 == 63 {
			runtime.Gosched()
		}
	}
	j.task = nil
	jobPool.Put(j)
}
