package tensor

import (
	"sync"
)

// Pool recycles tensor buffers by size class to keep the hot path allocation-free.
// It is safe for concurrent use. Buffers are bucketed by power-of-two byte size.
type Pool struct {
	mu      sync.Mutex
	buckets map[int][][]byte
}

// NewPool returns an empty pool.
func NewPool() *Pool { return &Pool{buckets: make(map[int][][]byte)} }

func sizeClass(n int) int {
	c := 64
	for c < n {
		c <<= 1
	}
	return c
}

// Get returns a zeroed tensor of the given dtype/shape backed by pooled storage.
func (p *Pool) Get(dt DType, shape ...int) *Tensor {
	s := Shape(shape)
	need := s.Numel() * dt.Size()
	cls := sizeClass(need)
	p.mu.Lock()
	bs := p.buckets[cls]
	var buf []byte
	if n := len(bs); n > 0 {
		buf = bs[n-1]
		p.buckets[cls] = bs[:n-1]
	}
	p.mu.Unlock()
	if buf == nil {
		buf = make([]byte, cls)
	}
	buf = buf[:need]
	clear(buf)
	return &Tensor{dtype: dt, shape: s.Clone(), strides: s.Strides(), buf: buf, pool: p}
}

// Put returns the tensor's storage to the pool. The tensor must not be used after.
func (p *Pool) Put(t *Tensor) {
	if t.pool != p || t.offset != 0 {
		return // not ours, or a view
	}
	cls := sizeClass(cap(t.buf))
	if cls != cap(t.buf) {
		return // not a pooled buffer size
	}
	p.mu.Lock()
	p.buckets[cls] = append(p.buckets[cls], t.buf[:cap(t.buf)])
	p.mu.Unlock()
	t.buf = nil
}
