package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// TestWAVRoundTrip: WriteWAV's 16-bit output reads back within one LSB,
// and a hand-built stereo float WAV mixes down to mono.
func TestWAVRoundTrip(t *testing.T) {
	in := []float32{0, 0.5, -0.5, 1, -1, 0.123}
	var buf bytes.Buffer
	if err := WriteWAV(&buf, in, 24000); err != nil {
		t.Fatal(err)
	}
	got, rate, err := ReadWAV(&buf)
	if err != nil || rate != 24000 || len(got) != len(in) {
		t.Fatalf("read: %v rate %d len %d", err, rate, len(got))
	}
	for i := range in {
		if math.Abs(float64(got[i]-in[i])) > 1.0/32767 {
			t.Errorf("sample %d: %g want %g", i, got[i], in[i])
		}
	}
	// Stereo IEEE float: (L, R) pairs average.
	var f bytes.Buffer
	le := binary.LittleEndian
	pcm := []float32{0.25, 0.75, -1, 0}
	f.WriteString("RIFF")
	binary.Write(&f, le, uint32(36+4*len(pcm)))
	f.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(3), uint16(2), uint32(16000), uint32(16000 * 8), uint16(8), uint16(32)} {
		binary.Write(&f, le, v)
	}
	f.WriteString("data")
	binary.Write(&f, le, uint32(4*len(pcm)))
	binary.Write(&f, le, pcm)
	got, rate, err = ReadWAV(&f)
	if err != nil || rate != 16000 || len(got) != 2 || got[0] != 0.5 || got[1] != -0.5 {
		t.Fatalf("stereo float: %v %d %v", err, rate, got)
	}
}
