package tensor

import (
	"sync"
)

// Pool recycles tensor buffers by size class to keep the hot path allocation-free.
// It is safe for concurrent use. Buffers are bucketed by power-of-two byte size.
type Pool struct {
	mu      sync.Mutex
	buckets map[int][][]byte
	free    []*Tensor // recycled headers (Put clears them; Get reinitialises)
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
	t := p.GetUninit(dt, shape...)
	clear(t.buf)
	return t
}

// GetUninit is Get without zeroing the storage. Use only when the caller writes
// every element before it is read.
func (p *Pool) GetUninit(dt DType, shape ...int) *Tensor {
	need := Shape(shape).Numel() * dt.Size()
	cls := sizeClass(need)
	p.mu.Lock()
	bs := p.buckets[cls]
	var buf []byte
	if n := len(bs); n > 0 {
		buf = bs[n-1]
		p.buckets[cls] = bs[:n-1]
	}
	var t *Tensor
	if n := len(p.free); n > 0 {
		t = p.free[n-1]
		p.free[n-1] = nil
		p.free = p.free[:n-1]
	}
	p.mu.Unlock()
	if buf == nil {
		buf = make([]byte, cls)
	}
	if t == nil {
		t = &Tensor{}
	}
	*t = Tensor{dtype: dt, buf: buf[:need], pool: p}
	t.setShape(shape)
	return t
}

// Put returns the tensor's storage — and its header — to the pool. The tensor
// must not be used after (its header may back an unrelated tensor next Get).
func (p *Pool) Put(t *Tensor) {
	if t.pool != p || t.offset != 0 {
		return // not ours, or a view
	}
	buf := t.buf
	cls := sizeClass(cap(buf))
	if cls != cap(buf) {
		return // not a pooled buffer size
	}
	*t = Tensor{} // stale use now fails loudly (nil buf) instead of aliasing
	p.mu.Lock()
	p.buckets[cls] = append(p.buckets[cls], buf[:cap(buf)])
	p.free = append(p.free, t)
	p.mu.Unlock()
}
