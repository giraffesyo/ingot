// Package par provides a minimal, allocation-light parallel-for used by kernels.
//
// It spawns at most MaxWorkers goroutines per region (never one per work item)
// and hands out indices through an atomic counter (dynamic scheduling), which
// balances uneven tiles without a central queue.
package par

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// MaxWorkers caps the goroutines used per region. Defaults to GOMAXPROCS at init.
var MaxWorkers = runtime.GOMAXPROCS(0)

// For runs fn(i) for i in [0,n). If the region would use a single worker it
// runs inline. grain is the number of consecutive indices handed to a worker
// at a time.
func For(n, grain int, fn func(i int)) {
	if grain < 1 {
		grain = 1
	}
	workers := MaxWorkers
	if w := (n + grain - 1) / grain; w < workers {
		workers = w
	}
	if workers <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				start := int(next.Add(int64(grain))) - grain
				if start >= n {
					return
				}
				end := min(start+grain, n)
				for i := start; i < end; i++ {
					fn(i)
				}
			}
		}()
	}
	wg.Wait()
}
