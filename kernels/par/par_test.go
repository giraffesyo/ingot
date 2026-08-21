package par

import (
	"sync/atomic"
	"testing"
)

func TestForCoversAllIndices(t *testing.T) {
	for _, n := range []int{0, 1, 2, 17, 1000, 12345} {
		for _, grain := range []int{1, 3, 64} {
			seen := make([]atomic.Int32, n)
			For(n, grain, func(i, w int) {
				if w < 0 || w >= Workers() {
					t.Errorf("bad worker id %d", w)
				}
				seen[i].Add(1)
			})
			for i := range seen {
				if seen[i].Load() != 1 {
					t.Fatalf("n=%d grain=%d: index %d visited %d times", n, grain, i, seen[i].Load())
				}
			}
		}
	}
}

func TestForNested(t *testing.T) {
	var total atomic.Int64
	For(50, 1, func(i, _ int) {
		For(50, 1, func(j, _ int) {
			total.Add(1)
		})
	})
	if total.Load() != 2500 {
		t.Fatalf("nested total %d", total.Load())
	}
}

type countTask struct{ n atomic.Int64 }

func (c *countTask) Run(i, w int) { c.n.Add(1) }

func TestRunTask(t *testing.T) {
	var c countTask
	Run(1000, 7, &c)
	if c.n.Load() != 1000 {
		t.Fatalf("got %d", c.n.Load())
	}
}

func BenchmarkRunOverhead(b *testing.B) {
	b.ReportAllocs()
	var c countTask
	for i := 0; i < b.N; i++ {
		Run(64, 1, &c)
	}
}
