package metal

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
	"unsafe"
)

const saxpySrc = `
#include <metal_stdlib>
using namespace metal;
kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                  constant float& a [[buffer(2)]], constant uint& n [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) {
	if (i < n) y[i] = a * x[i] + y[i];
}`

func f32s(b []byte) []float32 { return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4) }

func le32(v uint32) []byte { return binary.LittleEndian.AppendUint32(nil, v) }

func openDev(t testing.TB) *Device {
	t.Helper()
	if err := Supported(); err != nil {
		t.Skip(err)
	}
	d, err := Open()
	if err != nil {
		t.Skip(err)
	}
	return d
}

// TestSaxpy compiles a kernel at runtime, runs it over shared buffers and
// checks every element (ragged n: the grid is not a multiple of the group).
func TestSaxpy(t *testing.T) {
	d := openDev(t)
	t.Logf("device %s (unified memory %v)", d.Name, d.Unified)
	p, err := d.Compile(saxpySrc, "saxpy")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Compile(saxpySrc, "missing"); err == nil {
		t.Fatal("missing kernel function: want error")
	}
	if _, err := d.Compile("kernel void broken(", "broken"); err == nil {
		t.Fatal("bad source: want compile error")
	}
	const n = 1<<20 + 3
	bx, err := d.NewBuffer(4 * n)
	if err != nil {
		t.Fatal(err)
	}
	defer bx.Release()
	by, _ := d.NewBuffer(4 * n)
	defer by.Release()
	x, y := f32s(bx.Bytes()), f32s(by.Bytes())
	for i := range x {
		x[i], y[i] = float32(i%1000), 1
	}
	if err := p.Dispatch([3]int{n, 1, 1}, [3]int{256, 1, 1}, bx, by, le32(math.Float32bits(2)), le32(n)); err != nil {
		t.Fatal(err)
	}
	for i := range y {
		if want := 1 + 2*float32(i%1000); y[i] != want {
			t.Fatalf("y[%d] = %g, want %g", i, y[i], want)
		}
	}
}

// BenchmarkSaxpy times one dispatch round trip (encode, commit, wait).
func BenchmarkSaxpy(b *testing.B) {
	d := openDev(b)
	p, err := d.Compile(saxpySrc, "saxpy")
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{1 << 10, 1 << 24} {
		bx, _ := d.NewBuffer(4 * n)
		by, _ := d.NewBuffer(4 * n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(12 * n))
			for i := 0; i < b.N; i++ {
				if err := p.Dispatch([3]int{n, 1, 1}, [3]int{256, 1, 1}, bx, by, le32(math.Float32bits(1)), le32(uint32(n))); err != nil {
					b.Fatal(err)
				}
			}
		})
		bx.Release()
		by.Release()
	}
}
