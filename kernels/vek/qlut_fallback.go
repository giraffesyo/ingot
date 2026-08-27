//go:build !arm64 && !amd64

package vek

// QLut applies a 256-entry u8→u8 lookup table.
func QLut(dst, src []uint8, tab *[256]uint8) {
	for i := 0; i < min(len(dst), len(src)); i++ {
		dst[i] = tab[src[i]]
	}
}
