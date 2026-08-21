package tensor

import "testing"

func TestNewAndAccess(t *testing.T) {
	x := New(F32, 2, 3)
	if x.Numel() != 6 || x.DType() != F32 {
		t.Fatalf("bad tensor %v", x)
	}
	f := x.F32()
	f[4] = 1.5
	if x.F32()[4] != 1.5 {
		t.Fatal("write not visible")
	}
	if !x.IsContiguous() {
		t.Fatal("expected contiguous")
	}
	y := x.Reshape(3, 2)
	if y.F32()[4] != 1.5 {
		t.Fatal("reshape must share storage")
	}
}

func TestFromF32(t *testing.T) {
	d := []float32{1, 2, 3, 4}
	x := FromF32(d, 2, 2)
	x.F32()[0] = 9
	if d[0] != 9 {
		t.Fatal("FromF32 must not copy")
	}
}

func TestPool(t *testing.T) {
	p := NewPool()
	a := p.Get(F32, 10)
	a.F32()[0] = 3
	p.Put(a)
	b := p.Get(F32, 10)
	if b.F32()[0] != 0 {
		t.Fatal("pooled tensor must be zeroed")
	}
}

func TestShape(t *testing.T) {
	s := Shape{2, 3, 4}
	st := s.Strides()
	if st[0] != 12 || st[1] != 4 || st[2] != 1 {
		t.Fatalf("bad strides %v", st)
	}
	if s.String() != "[2,3,4]" {
		t.Fatal(s.String())
	}
}
