// Package par provides a persistent worker pool and a parallel-for used by
// kernels.
//
// Design:
//   - MaxWorkers-1 helper goroutines are started lazily on first use and live
//     for the life of the process. The calling goroutine always participates
//     (worker id 0), so a region never waits on a goroutine spawn.
//   - Work is handed to helpers with a non-blocking send on an unbuffered
//     channel: only helpers that are idle *right now* join a region. This makes
//     nested or concurrent For calls safe (no deadlock, the caller just does
//     more of the work itself) at the cost of occasionally using fewer helpers.
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
)

// MaxWorkers is the total number of workers (including the caller). It is read
// once, when the pool starts on first use.
var MaxWorkers = runtime.GOMAXPROCS(0)

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
	wg       sync.WaitGroup
}

var (
	once     sync.Once
	jobs     chan *job
	nworkers int
	jobPool  = sync.Pool{New: func() any { return new(job) }}
)

func start() {
	nworkers = max(MaxWorkers, 1)
	jobs = make(chan *job)
	for w := 1; w < nworkers; w++ {
		go worker(w)
	}
}

func worker(w int) {
	for j := range jobs {
		j.run(w)
		j.wg.Done()
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
// For's fn are in [0, Workers()).
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
	for sent := 0; sent < helpers; sent++ {
		j.wg.Add(1)
		select {
		case jobs <- j:
		default:
			j.wg.Done()
			sent = helpers // no idle helper: stop offering
		}
	}
	j.run(0)
	j.wg.Wait()
	j.task = nil
	jobPool.Put(j)
}
