package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"os"
)

// aviWriter writes a Motion-JPEG AVI (RIFF AVI 1.0 with an idx1 index):
// each frame one JPEG, readable by common players and editors without a
// codec dependency here.
type aviWriter struct {
	f          *os.File
	fps        int
	w, h       int
	frames     int
	offsets    []uint32 // of each 00dc chunk, from the movi list's type tag
	sizes      []uint32
	moviStart  int64
	headerSize int64
	maxFrame   uint32
}

func newAVI(path string, fps int) (*aviWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &aviWriter{f: f, fps: fps}, nil
}

func le32(b *bytes.Buffer, v uint32) { _ = binary.Write(b, binary.LittleEndian, v) }
func le16(b *bytes.Buffer, v uint16) { _ = binary.Write(b, binary.LittleEndian, v) }

// header writes RIFF/hdrl with placeholder counts, rewritten on close.
func (a *aviWriter) header() []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	le32(&b, 0) // file size - 8, patched
	b.WriteString("AVI LIST")
	le32(&b, 4+64+8+116) // hdrl size: avih chunk, strl list
	b.WriteString("hdrlavih")
	le32(&b, 56)
	le32(&b, uint32(1000000/a.fps)) // µs per frame
	le32(&b, 0)                     // max bytes/s
	le32(&b, 0)                     // padding granularity
	le32(&b, 0x10)                  // AVIF_HASINDEX
	le32(&b, uint32(a.frames))      // total frames
	le32(&b, 0)                     // initial frames
	le32(&b, 1)                     // streams
	le32(&b, a.maxFrame)            // suggested buffer
	le32(&b, uint32(a.w))
	le32(&b, uint32(a.h))
	for range 4 {
		le32(&b, 0)
	}
	b.WriteString("LIST")
	le32(&b, 4+8+56+8+40)
	b.WriteString("strlstrh")
	le32(&b, 56)
	b.WriteString("vidsMJPG")
	le32(&b, 0) // flags
	le16(&b, 0) // priority
	le16(&b, 0) // language
	le32(&b, 0) // initial frames
	le32(&b, 1) // scale
	le32(&b, uint32(a.fps))
	le32(&b, 0) // start
	le32(&b, uint32(a.frames))
	le32(&b, a.maxFrame)
	le32(&b, 0xFFFFFFFF) // quality
	le32(&b, 0)          // sample size
	le16(&b, 0)
	le16(&b, 0)
	le16(&b, uint16(a.w))
	le16(&b, uint16(a.h))
	b.WriteString("strf")
	le32(&b, 40)
	le32(&b, 40) // BITMAPINFOHEADER
	le32(&b, uint32(a.w))
	le32(&b, uint32(a.h))
	le16(&b, 1)
	le16(&b, 24)
	b.WriteString("MJPG")
	le32(&b, uint32(a.w*a.h*3))
	for range 4 {
		le32(&b, 0)
	}
	b.WriteString("LIST")
	le32(&b, 0) // movi size, patched
	b.WriteString("movi")
	return b.Bytes()
}

func (a *aviWriter) add(img image.Image) error {
	if a.frames == 0 {
		a.w, a.h = img.Bounds().Dx(), img.Bounds().Dy()
		h := a.header()
		if _, err := a.f.Write(h); err != nil {
			return err
		}
		a.headerSize = int64(len(h))
		a.moviStart = a.headerSize - 4 // the "movi" tag
	} else if img.Bounds().Dx() != a.w || img.Bounds().Dy() != a.h {
		return fmt.Errorf("avi: frame size changed")
	}
	var j bytes.Buffer
	if err := jpeg.Encode(&j, img, &jpeg.Options{Quality: 92}); err != nil {
		return err
	}
	if j.Len()%2 == 1 {
		j.WriteByte(0)
	}
	pos, err := a.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	var c bytes.Buffer
	c.WriteString("00dc")
	le32(&c, uint32(j.Len()))
	c.Write(j.Bytes())
	if _, err := a.f.Write(c.Bytes()); err != nil {
		return err
	}
	a.offsets = append(a.offsets, uint32(pos-a.moviStart))
	a.sizes = append(a.sizes, uint32(j.Len()))
	a.maxFrame = max(a.maxFrame, uint32(j.Len()))
	a.frames++
	return nil
}

func (a *aviWriter) close() error {
	if a.frames == 0 {
		return a.f.Close()
	}
	end, err := a.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	var idx bytes.Buffer
	idx.WriteString("idx1")
	le32(&idx, uint32(16*a.frames))
	for i := range a.frames {
		idx.WriteString("00dc")
		le32(&idx, 0x10) // keyframe
		le32(&idx, a.offsets[i])
		le32(&idx, a.sizes[i])
	}
	if _, err := a.f.Write(idx.Bytes()); err != nil {
		return err
	}
	total, err := a.f.Seek(0, io.SeekCurrent)
	if err != nil {
		return err
	}
	// Rewrite the header with the final counts and sizes.
	h := a.header()
	binary.LittleEndian.PutUint32(h[4:], uint32(total-8))
	binary.LittleEndian.PutUint32(h[len(h)-8:], uint32(end-a.moviStart))
	if _, err := a.f.WriteAt(h, 0); err != nil {
		return err
	}
	return a.f.Close()
}
