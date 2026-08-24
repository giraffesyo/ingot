package sme

import (
	"math/rand/v2"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// stressOnce runs one m=31,n=33,k=512 Sgemm and reports max abs error.
func stressOnce(a, b []float32, c []float32, want []float32) float64 {
	Sgemm(31, 33, 512, a, 512, b, 33, c, 33)
	var worst float64
	for i := range want {
		d := float64(c[i] - want[i])
		if d < 0 {
			d = -d
		}
		if d > worst {
			worst = d
		}
	}
	return worst
}

func stressSetup() (a, b, c, want []float32) {
	r := rand.New(rand.NewPCG(3, 4))
	a = make([]float32, 31*512)
	b = make([]float32, 512*33)
	for i := range a {
		a[i] = r.Float32()*2 - 1
	}
	for i := range b {
		b[i] = r.Float32()*2 - 1
	}
	c = make([]float32, 31*33)
	want = make([]float32, 31*33)
	for i := 0; i < 31; i++ {
		for j := 0; j < 33; j++ {
			var s float64
			for p := 0; p < 512; p++ {
				s += float64(a[i*512+p]) * float64(b[p*33+j])
			}
			want[i*33+j] = float32(s)
		}
	}
	return
}

func TestStressSerial(t *testing.T) {
	if !Available() {
		t.Skip()
	}
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	a, b, c, want := stressSetup()
	for i := 0; i < 3000; i++ {
		if e := stressOnce(a, b, c, want); e > 1e-3 {
			t.Fatalf("iter %d: err %g (serial)", i, e)
		}
	}
}

func TestStressParallel(t *testing.T) {
	if !Available() {
		t.Skip()
	}
	a, b, _, want := stressSetup()
	var bad atomic.Int64
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c := make([]float32, 31*33)
			for i := 0; i < 1500; i++ {
				if e := stressOnce(a, b, c, want); e > 1e-3 {
					bad.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	if n := bad.Load(); n > 0 {
		t.Fatalf("%d corrupted results out of 12000 (concurrent)", n)
	}
}

func TestStressGCStorm(t *testing.T) {
	if !Available() {
		t.Skip()
	}
	a, b, c, want := stressSetup()
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				runtime.GC()
			}
		}
	}()
	defer close(stop)
	for i := 0; i < 2000; i++ {
		if e := stressOnce(a, b, c, want); e > 1e-3 {
			t.Fatalf("iter %d: err %g (GC storm)", i, e)
		}
	}
}
