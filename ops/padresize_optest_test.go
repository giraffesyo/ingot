package ops

import (
	"testing"
)

func TestPadConstant(t *testing.T) {
	x := f32t([]int{1, 2}, 5, 6)
	op := mkOp(t, "Pad", 11, Attrs{"mode": {Kind: KindString, S: "constant"}}, 3, 1)
	out := run(t, op, x, i64t([]int{4}, 0, 1, 0, 2), f32t(nil, -1))[0]
	// pad col-begin 1 (=-1), col-end 2 (=-1): [-1, 5, 6, -1, -1]
	eqF32(t, "padConst", out, []int{1, 5}, []float32{-1, 5, 6, -1, -1})
}

func TestPadReflect(t *testing.T) {
	x := f32t([]int{1, 4}, 1, 2, 3, 4)
	op := mkOp(t, "Pad", 11, Attrs{"mode": {Kind: KindString, S: "reflect"}}, 2, 1)
	out := run(t, op, x, i64t([]int{4}, 0, 2, 0, 2))[0]
	// reflect pad 2 each side of [1,2,3,4]: [3,2, 1,2,3,4, 3,2]
	eqF32(t, "padReflect", out, []int{1, 8}, []float32{3, 2, 1, 2, 3, 4, 3, 2})
}

func TestPadEdge(t *testing.T) {
	x := f32t([]int{1, 3}, 7, 8, 9)
	op := mkOp(t, "Pad", 11, Attrs{"mode": {Kind: KindString, S: "edge"}}, 2, 1)
	out := run(t, op, x, i64t([]int{4}, 0, 2, 0, 1))[0]
	// edge: [7,7, 7,8,9, 9]
	eqF32(t, "padEdge", out, []int{1, 6}, []float32{7, 7, 7, 8, 9, 9})
}

func TestResizeNearest2x(t *testing.T) {
	x := f32t([]int{1, 1, 2, 2}, 1, 2, 3, 4)
	op := mkOp(t, "Resize", 13, Attrs{"mode": {Kind: KindString, S: "nearest"}, "coordinate_transformation_mode": {Kind: KindString, S: "asymmetric"}, "nearest_mode": {Kind: KindString, S: "floor"}}, 4, 1)
	// scales input (X, roi(empty), scales)
	out := run(t, op, x, nil, f32t([]int{4}, 1, 1, 2, 2))[0]
	eqF32(t, "resizeNN", out, []int{1, 1, 4, 4}, []float32{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	})
}

func TestResizeLinearAlignCorners(t *testing.T) {
	x := f32t([]int{1, 1, 1, 2}, 0, 10)
	op := mkOp(t, "Resize", 13, Attrs{"mode": {Kind: KindString, S: "linear"}, "coordinate_transformation_mode": {Kind: KindString, S: "align_corners"}}, 4, 1)
	out := run(t, op, x, nil, nil, i64t([]int{4}, 1, 1, 1, 3))[0]
	// align_corners over [0,10] at 3 points -> 0,5,10
	eqF32(t, "resizeLin", out, []int{1, 1, 1, 3}, []float32{0, 5, 10})
}

func TestResizeSizes(t *testing.T) {
	x := f32t([]int{1, 1, 2, 2}, 1, 2, 3, 4)
	op := mkOp(t, "Resize", 13, Attrs{"mode": {Kind: KindString, S: "nearest"}, "coordinate_transformation_mode": {Kind: KindString, S: "asymmetric"}, "nearest_mode": {Kind: KindString, S: "floor"}}, 4, 1)
	// sizes input: X, roi, scales(empty), sizes
	out := run(t, op, x, nil, nil, i64t([]int{4}, 1, 1, 4, 2))[0]
	eqF32(t, "resizeSizes", out, []int{1, 1, 4, 2}, []float32{1, 2, 1, 2, 3, 4, 3, 4})
}

func TestConvTransposeStride2(t *testing.T) {
	// 2x2 input, 2x2 kernel, stride 2 -> 4x4, each input pixel scatters a block.
	x := f32t([]int{1, 1, 2, 2}, 1, 2, 3, 4)
	w := f32t([]int{1, 1, 2, 2}, 1, 1, 1, 1) // sums into a 2x2 block
	op := mkOp(t, "ConvTranspose", 1, Attrs{"strides": {Kind: KindInts, Ints: []int64{2, 2}}, "kernel_shape": {Kind: KindInts, Ints: []int64{2, 2}}}, 2, 1)
	out := run(t, op, x, w)[0]
	// each pixel v -> 2x2 block of v at stride 2; non-overlapping
	eqF32(t, "convT", out, []int{1, 1, 4, 4}, []float32{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	})
}
