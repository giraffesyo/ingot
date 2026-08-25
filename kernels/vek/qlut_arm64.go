//go:build arm64

package vek

//go:noescape
func qlut_asm(dst *uint8, src *uint8, n int64, tab *uint8)

// QLut applies a 256-entry u8→u8 lookup table.
func QLut(dst, src []uint8, tab *[256]uint8) {
	n := min(len(dst), len(src))
	m := n &^ 15
	if m > 0 {
		qlut_asm(&dst[0], &src[0], int64(m), &tab[0])
	}
	for i := m; i < n; i++ {
		dst[i] = tab[src[i]]
	}
}
