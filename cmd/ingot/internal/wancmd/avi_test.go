package wancmd

import (
	"encoding/binary"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"testing"
)

// TestAVIStructure checks the RIFF sizes, the frame count and the index.
func TestAVIStructure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "x.avi")
	a, err := newAVI(path, 24)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		img := image.NewNRGBA(image.Rect(0, 0, 32, 16))
		for p := range img.Pix {
			img.Pix[p] = uint8(40 * i)
		}
		img.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 255})
		if err := a.add(img); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.close(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	u32 := func(o int) int { return int(binary.LittleEndian.Uint32(b[o:])) }
	if string(b[:4]) != "RIFF" || string(b[8:12]) != "AVI " || u32(4) != len(b)-8 {
		t.Fatalf("RIFF header: %q size %d, file %d", b[:12], u32(4), len(b))
	}
	if string(b[12:16]) != "LIST" || string(b[20:24]) != "hdrl" {
		t.Fatalf("hdrl list: %q", b[12:24])
	}
	movi := 20 + u32(16) // the LIST after hdrl
	if string(b[movi:movi+4]) != "LIST" || string(b[movi+8:movi+12]) != "movi" {
		t.Fatalf("movi list at %d: %q", movi, b[movi:movi+12])
	}
	idx := movi + 8 + u32(movi+4)
	if string(b[idx:idx+4]) != "idx1" || u32(idx+4) != 3*16 || idx+8+48 != len(b) {
		t.Fatalf("idx1 at %d: %q size %d", idx, b[idx:idx+4], u32(idx+4))
	}
	if frames := u32(24 + 8 + 16); frames != 3 { // avih total frames
		t.Fatalf("avih frames %d", frames)
	}
	for i := range 3 {
		off := movi + 8 + u32(idx+8+16*i+8)
		if string(b[off:off+4]) != "00dc" || b[off+8] != 0xFF || b[off+9] != 0xD8 {
			t.Fatalf("frame %d at %d: %q", i, off, b[off:off+10])
		}
	}
}
