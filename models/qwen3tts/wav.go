package qwen3tts

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// WriteWAV writes mono samples in [-1, 1] as 16-bit PCM WAV.
func WriteWAV(w io.Writer, samples []float32, rate int) error {
	data := make([]byte, 44+2*len(samples))
	le := binary.LittleEndian
	copy(data[0:], "RIFF")
	le.PutUint32(data[4:], uint32(36+2*len(samples)))
	copy(data[8:], "WAVEfmt ")
	le.PutUint32(data[16:], 16)             // fmt chunk size
	le.PutUint16(data[20:], 1)              // PCM
	le.PutUint16(data[22:], 1)              // mono
	le.PutUint32(data[24:], uint32(rate))   // sample rate
	le.PutUint32(data[28:], uint32(2*rate)) // byte rate
	le.PutUint16(data[32:], 2)              // block align
	le.PutUint16(data[34:], 16)             // bits per sample
	copy(data[36:], "data")
	le.PutUint32(data[40:], uint32(2*len(samples)))
	for i, s := range samples {
		v := math.Round(float64(max(-1, min(1, s))) * 32767)
		le.PutUint16(data[44+2*i:], uint16(int16(v)))
	}
	_, err := w.Write(data)
	return err
}

// ReadWAV reads a PCM (8/16/24/32-bit) or IEEE-float (32/64-bit) WAV file,
// mixing channels down to mono, as samples in [-1, 1] and the sample rate.
func ReadWAV(r io.Reader) ([]float32, int, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, 0, err
	}
	le := binary.LittleEndian
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("qwen3tts: not a RIFF/WAVE file")
	}
	var format, channels, bits int
	rate := 0
	var data []byte
	for p := 12; p+8 <= len(raw); {
		id, size := string(raw[p:p+4]), int(le.Uint32(raw[p+4:]))
		body := raw[p+8 : min(len(raw), p+8+size)]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return nil, 0, fmt.Errorf("qwen3tts: short fmt chunk")
			}
			format, channels = int(le.Uint16(body[0:])), int(le.Uint16(body[2:]))
			rate, bits = int(le.Uint32(body[4:])), int(le.Uint16(body[14:]))
			if format == 0xFFFE && len(body) >= 26 { // WAVE_FORMAT_EXTENSIBLE: the subformat's tag
				format = int(le.Uint16(body[24:]))
			}
		case "data":
			data = body
		}
		p += 8 + size + size&1
	}
	if channels == 0 || data == nil {
		return nil, 0, fmt.Errorf("qwen3tts: WAV without fmt or data chunk")
	}
	width := bits / 8
	var sample func(b []byte) float64
	switch {
	case format == 1 && bits == 8:
		sample = func(b []byte) float64 { return (float64(b[0]) - 128) / 128 }
	case format == 1 && bits == 16:
		sample = func(b []byte) float64 { return float64(int16(le.Uint16(b))) / 32768 }
	case format == 1 && bits == 24:
		sample = func(b []byte) float64 {
			return float64(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / 8388608
		}
	case format == 1 && bits == 32:
		sample = func(b []byte) float64 { return float64(int32(le.Uint32(b))) / 2147483648 }
	case format == 3 && bits == 32:
		sample = func(b []byte) float64 { return float64(math.Float32frombits(le.Uint32(b))) }
	case format == 3 && bits == 64:
		sample = func(b []byte) float64 { return math.Float64frombits(le.Uint64(b)) }
	default:
		return nil, 0, fmt.Errorf("qwen3tts: WAV format %d with %d bits not supported", format, bits)
	}
	frame := width * channels
	out := make([]float32, len(data)/frame)
	for i := range out {
		var acc float64
		for c := range channels {
			acc += sample(data[i*frame+c*width:])
		}
		out[i] = float32(acc / float64(channels))
	}
	return out, rate, nil
}
