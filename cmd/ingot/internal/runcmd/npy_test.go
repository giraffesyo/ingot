package runcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/giraffesyo/ingot/tensor"
)

// TestNPYRoundTrip: writeNPY then readNPY returns the same tensor, for
// each writable dtype and for rank 0-3 shapes.
func TestNPYRoundTrip(t *testing.T) {
	dir := t.TempDir()
	for i, src := range []*tensor.Tensor{
		tensor.FromF32([]float32{1.5, -2, 3.25, 0, 7, 8}, 2, 3),
		tensor.FromI64([]int64{1, -2, 1 << 40}, 3),
		tensor.FromF32([]float32{42}),
		tensor.FromF32(make([]float32, 24), 2, 3, 4),
	} {
		var b bytes.Buffer
		if err := writeNPY(&b, src); err != nil {
			t.Fatal(err)
		}
		if (b.Len()-len(src.Bytes()))%64 != 0 {
			t.Errorf("case %d: header not padded to 64 bytes", i)
		}
		p := filepath.Join(dir, "x.npy")
		if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		got, err := readNPY(p)
		if err != nil {
			t.Fatal(err)
		}
		if got.DType() != src.DType() || !got.Shape().Equal(src.Shape()) || !bytes.Equal(got.Bytes(), src.Bytes()) {
			t.Errorf("case %d: %s%v vs %s%v", i, got.DType(), got.Shape(), src.DType(), src.Shape())
		}
	}
}
