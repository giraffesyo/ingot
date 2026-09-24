//go:build !(darwin && arm64)

package metal

import "errors"

// Device is unavailable off darwin/arm64.
type Device struct {
	Name    string
	Unified bool
}

// Pipeline is unavailable off darwin/arm64.
type Pipeline struct{ MaxThreads int }

// Buffer is unavailable off darwin/arm64.
type Buffer struct{}

// Arg is a kernel argument.
type Arg any

var errUnavailable = errors.New("metal: not available on this platform")

func Available() bool                                              { return false }
func Open() (*Device, error)                                       { return nil, errUnavailable }
func (d *Device) Compile(src, name string) (*Pipeline, error)      { return nil, errUnavailable }
func (d *Device) NewBuffer(n int) (*Buffer, error)                 { return nil, errUnavailable }
func (b *Buffer) Bytes() []byte                                    { return nil }
func (b *Buffer) Release()                                         {}
func (p *Pipeline) Dispatch(grid, group [3]int, args ...Arg) error { return errUnavailable }
