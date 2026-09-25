package metal

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// objc holds the Objective-C runtime entry points.
var objc struct {
	once                         sync.Once
	err                          error
	msgSend, sel, class          uintptr
	poolPush, poolPop, createDev uintptr
	selCache                     sync.Map // string → SEL
}

func loadObjC() error {
	objc.once.Do(func() {
		lib, err := dlopen("/usr/lib/libobjc.A.dylib")
		if err != nil {
			objc.err = err
			return
		}
		if _, err := dlopen("/System/Library/Frameworks/Foundation.framework/Foundation"); err != nil {
			objc.err = err
			return
		}
		mtl, err := dlopen("/System/Library/Frameworks/Metal.framework/Metal")
		if err != nil {
			objc.err = err
			return
		}
		for _, s := range []struct {
			h    uintptr
			name string
			dst  *uintptr
		}{
			{lib, "objc_msgSend", &objc.msgSend}, {lib, "sel_registerName", &objc.sel},
			{lib, "objc_getClass", &objc.class}, {lib, "objc_autoreleasePoolPush", &objc.poolPush},
			{lib, "objc_autoreleasePoolPop", &objc.poolPop}, {mtl, "MTLCreateSystemDefaultDevice", &objc.createDev},
		} {
			if *s.dst, objc.err = dlsym(s.h, s.name); objc.err != nil {
				return
			}
		}
	})
	return objc.err
}

func sel(name string) uintptr {
	if v, ok := objc.selCache.Load(name); ok {
		return v.(uintptr)
	}
	b := cstr(name)
	s := call(objc.sel, uintptr(unsafe.Pointer(&b[0])))
	runtime.KeepAlive(b)
	objc.selCache.Store(name, s)
	return s
}

func class(name string) uintptr {
	b := cstr(name)
	c := call(objc.class, uintptr(unsafe.Pointer(&b[0])))
	runtime.KeepAlive(b)
	return c
}

// send is objc_msgSend(obj, sel(name), args...).
func send(obj uintptr, name string, args ...uintptr) uintptr {
	return call(objc.msgSend, append([]uintptr{obj, sel(name)}, args...)...)
}

// sendF is send for a method returning a double.
func sendF(obj uintptr, name string, args ...uintptr) float64 {
	return callF(objc.msgSend, append([]uintptr{obj, sel(name)}, args...)...)
}

func nsstring(s string) uintptr {
	b := cstr(s)
	r := send(class("NSString"), "stringWithUTF8String:", uintptr(unsafe.Pointer(&b[0])))
	runtime.KeepAlive(b)
	return r
}

// nserror renders an NSError (0 → "unknown error").
func nserror(e uintptr) error {
	if e == 0 {
		return errors.New("unknown error")
	}
	return errors.New(gostr(send(send(e, "localizedDescription"), "UTF8String")))
}

// Device is a Metal GPU with one command queue. Its methods are safe for
// concurrent use: every Metal call runs on the device's worker goroutine,
// which is locked to one OS thread and wraps each job in an autorelease
// pool.
type Device struct {
	id, queue uintptr
	Name      string
	Unified   bool // CPU and GPU share memory (Apple silicon)
	jobs      chan func()
}

var shared struct {
	once sync.Once
	dev  *Device
	err  error
}

// Available reports whether the GPU backend is usable (see Supported).
func Available() bool { return Supported() == nil }

var support struct {
	once sync.Once
	err  error
}

// Supported reports why the GPU backend cannot run here, or nil: a Metal
// device must open and compile the GEMM kernels, which need Metal 4 tensor
// ops (MetalPerformancePrimitives; macOS 26+ on Apple silicon — not, e.g.,
// virtualized CI runners).
func Supported() error {
	support.once.Do(func() {
		d, err := Open()
		if err == nil {
			err = d.Prepare()
		}
		if err != nil {
			support.err = fmt.Errorf("metal: GPU backend unsupported: %w", err)
		}
	})
	return support.err
}

// Open returns the system default device (opened once, shared).
func Open() (*Device, error) {
	shared.once.Do(func() { shared.dev, shared.err = open() })
	return shared.dev, shared.err
}

func open() (*Device, error) {
	if err := loadObjC(); err != nil {
		return nil, err
	}
	d := &Device{jobs: make(chan func())}
	go func() {
		runtime.LockOSThread() // never unlocked: this goroutine is the device thread
		for f := range d.jobs {
			pool := call(objc.poolPush)
			f()
			call(objc.poolPop, pool)
		}
	}()
	var err error
	d.do(func() {
		if d.id = call(objc.createDev); d.id == 0 {
			err = errors.New("metal: no GPU device")
			return
		}
		d.queue = send(d.id, "newCommandQueue")
		d.Name = gostr(send(send(d.id, "name"), "UTF8String"))
		d.Unified = send(d.id, "hasUnifiedMemory")&1 == 1
	})
	if err != nil {
		return nil, err
	}
	return d, nil
}

// MaxWorkingSet is the device's recommendedMaxWorkingSetSize: about how
// many bytes of buffers its command buffers can keep resident. Work that
// references more fails at run time (kIOGPUCommandBufferCallbackErrorOutOfMemory).
func (d *Device) MaxWorkingSet() int {
	var n uintptr
	d.do(func() { n = send(d.id, "recommendedMaxWorkingSetSize") })
	return int(n)
}

// Allocated is the device's currentAllocatedSize: bytes of buffers alive
// now (wrapped memory included).
func (d *Device) Allocated() int {
	var n uintptr
	d.do(func() { n = send(d.id, "currentAllocatedSize") })
	return int(n)
}

// do runs f on the device thread and waits for it.
func (d *Device) do(f func()) {
	done := make(chan struct{})
	d.jobs <- func() { f(); close(done) }
	<-done
}

// Pipeline is a compiled compute kernel.
type Pipeline struct {
	dev  *Device
	pso  uintptr
	name string
	// MaxThreads is the kernel's maxTotalThreadsPerThreadgroup.
	MaxThreads int
}

// Compile compiles Metal Shading Language source and returns the pipeline
// for its kernel function name.
func (d *Device) Compile(src, name string) (*Pipeline, error) {
	var p *Pipeline
	var err error
	d.do(func() {
		var e uintptr
		lib := send(d.id, "newLibraryWithSource:options:error:", nsstring(src), 0, uintptr(unsafe.Pointer(&e)))
		if lib == 0 {
			err = fmt.Errorf("metal: compile: %w", nserror(e))
			return
		}
		fn := send(lib, "newFunctionWithName:", nsstring(name))
		if fn == 0 {
			err = fmt.Errorf("metal: no kernel function %q", name)
			return
		}
		pso := send(d.id, "newComputePipelineStateWithFunction:error:", fn, uintptr(unsafe.Pointer(&e)))
		if pso == 0 {
			err = fmt.Errorf("metal: pipeline %q: %w", name, nserror(e))
			return
		}
		p = &Pipeline{dev: d, pso: pso, name: name, MaxThreads: int(send(pso, "maxTotalThreadsPerThreadgroup"))}
	})
	return p, err
}

// Buffer is GPU memory in shared storage: on unified-memory devices the CPU
// reads and writes it in place through Bytes, with no copies.
type Buffer struct {
	dev  *Device
	id   uintptr
	n    int
	base unsafe.Pointer // contents: stable for shared-storage buffers
}

const storageModeShared = 0

// NewBuffer allocates n bytes of shared, zeroed GPU memory.
func (d *Device) NewBuffer(n int) (*Buffer, error) {
	b := &Buffer{dev: d, n: n}
	d.do(func() {
		if b.id = send(d.id, "newBufferWithLength:options:", uintptr(n), storageModeShared); b.id != 0 {
			b.base = cptr(send(b.id, "contents"))
		}
	})
	if b.id == 0 {
		return nil, fmt.Errorf("metal: cannot allocate %d-byte buffer", n)
	}
	return b, nil
}

// PageSize is the alignment Wrap requires.
var PageSize = syscall.Getpagesize()

// Wrap exposes existing memory to the GPU without copying — e.g. a
// memory-mapped weight file. mem must start on a page boundary; the buffer
// spans len(mem) rounded up to whole pages (the mapping covers them). The
// memory must outlive the buffer.
func (d *Device) Wrap(mem []byte) (*Buffer, error) {
	if len(mem) == 0 || uintptr(unsafe.Pointer(&mem[0]))%uintptr(PageSize) != 0 {
		return nil, fmt.Errorf("metal: Wrap needs non-empty page-aligned memory")
	}
	n := (len(mem) + PageSize - 1) / PageSize * PageSize
	b := &Buffer{dev: d, n: len(mem), base: unsafe.Pointer(&mem[0])}
	d.do(func() {
		b.id = send(d.id, "newBufferWithBytesNoCopy:length:options:deallocator:",
			uintptr(unsafe.Pointer(&mem[0])), uintptr(n), storageModeShared, 0)
	})
	if b.id == 0 {
		return nil, fmt.Errorf("metal: cannot wrap %d bytes", len(mem))
	}
	return b, nil
}

// Len is the buffer's size in bytes.
func (b *Buffer) Len() int { return b.n }

// Bytes is the buffer's memory, valid until Release.
func (b *Buffer) Bytes() []byte { return unsafe.Slice((*byte)(b.base), b.n) }

// Release frees the buffer (never the wrapped memory).
func (b *Buffer) Release() {
	if b.id != 0 {
		b.dev.do(func() { send(b.id, "release") })
		b.id = 0
	}
}

// Region binds a buffer at a byte offset.
type Region struct {
	B   *Buffer
	Off int
}

// At returns the region of b starting off bytes in.
func (b *Buffer) At(off int) Region { return Region{b, off} }

// Arg is a kernel argument: a *Buffer, a Region, or a small constant
// ([]byte, set with setBytes, at most 4 KiB).
type Arg any

type mtlSize struct{ w, h, d uint64 }

// Encoder records dispatches into one command buffer (see Device.Run).
// Dispatches execute in order; each sees the previous ones' writes.
type Encoder struct {
	enc   uintptr
	err   error
	count map[string]int // dispatches per kernel (Stream.CountDispatches)
}

// Run records the dispatches fn makes into a single command buffer, commits
// it and waits — one CPU/GPU round trip for a whole sequence of kernels.
// fn runs on the device thread: it must not call other Device methods.
func (d *Device) Run(fn func(e *Encoder)) error {
	var err error
	d.do(func() {
		cb := send(d.queue, "commandBuffer")
		e := &Encoder{enc: send(cb, "computeCommandEncoder")}
		fn(e)
		send(e.enc, "endEncoding")
		if e.err != nil {
			err = e.err
			return
		}
		send(cb, "commit")
		send(cb, "waitUntilCompleted")
		if ce := send(cb, "error"); ce != 0 {
			err = fmt.Errorf("metal: command buffer: %w", nserror(ce))
		}
	})
	return err
}

// Dispatch encodes p over a grid of threads (non-uniform threadgroups: the
// grid need not be a multiple of group) with args bound to indices 0, 1, ….
func (e *Encoder) Dispatch(p *Pipeline, grid, group [3]int, args ...Arg) {
	if e.err != nil {
		return
	}
	send(e.enc, "setComputePipelineState:", p.pso)
	if e.count != nil {
		e.count[p.name]++
	}
	for i, a := range args {
		switch v := a.(type) {
		case *Buffer:
			send(e.enc, "setBuffer:offset:atIndex:", v.id, 0, uintptr(i))
		case Region:
			if v.Off < 0 || v.Off > v.B.n {
				e.err = fmt.Errorf("metal: arg %d: offset %d outside %d-byte buffer", i, v.Off, v.B.n)
				return
			}
			send(e.enc, "setBuffer:offset:atIndex:", v.B.id, uintptr(v.Off), uintptr(i))
		case []byte:
			if len(v) == 0 || len(v) > 4096 {
				e.err = fmt.Errorf("metal: constant arg %d is %d bytes (want 1..4096)", i, len(v))
				return
			}
			send(e.enc, "setBytes:length:atIndex:", uintptr(unsafe.Pointer(&v[0])), uintptr(len(v)), uintptr(i))
			runtime.KeepAlive(v)
		default:
			e.err = fmt.Errorf("metal: arg %d: unsupported type %T", i, a)
			return
		}
	}
	g := mtlSize{uint64(grid[0]), uint64(grid[1]), uint64(grid[2])}
	t := mtlSize{uint64(group[0]), uint64(group[1]), uint64(group[2])}
	// MTLSize is 24 bytes: the arm64 ABI passes it by reference.
	send(e.enc, "dispatchThreads:threadsPerThreadgroup:", uintptr(unsafe.Pointer(&g)), uintptr(unsafe.Pointer(&t)))
}

// Stream records dispatches across calls — for an executor encoding op by
// op. Encode queues its callbacks; every streamBatch of them are recorded
// on the device thread into a command buffer that is committed at once
// (a thread hand-off per call would dominate small kernels, and waiting
// for Flush to commit anything left the GPU idle while the host encoded).
// Command buffers on one queue execute in commit order. Flush records the
// rest, commits, and waits for all of them. Not safe for concurrent use.
type Stream struct {
	d       *Device
	cbs     []uintptr // committed, retained, not yet waited on
	err     error
	queued  []func(e *Encoder)
	pending int
	gpu     time.Duration
	// Counts, when non-nil, accumulates dispatches per kernel name
	// (diagnostics: which kernels a graph issues, how often).
	Counts map[string]int
}

// streamBatch is how many queued callbacks are recorded and committed as
// one command buffer before Flush.
const streamBatch = 32

// NewStream returns an empty stream on d.
func (d *Device) NewStream() *Stream { return &Stream{d: d} }

// Encode queues fn's dispatches. fn runs later on the device thread and
// must not call Device methods; it sees only what it captured. The first
// error sticks until Flush reports it (work recorded after it is dropped).
func (s *Stream) Encode(fn func(e *Encoder)) {
	s.queued = append(s.queued, fn)
	s.pending++
	if len(s.queued) >= streamBatch {
		s.d.do(s.commitQueued)
	}
}

// commitQueued records the queued callbacks into a new command buffer and
// commits it without waiting (on the device thread).
func (s *Stream) commitQueued() {
	if len(s.queued) == 0 {
		return
	}
	cb := send(send(s.d.queue, "commandBuffer"), "retain")
	e := &Encoder{enc: send(cb, "computeCommandEncoder"), count: s.Counts}
	for i, fn := range s.queued {
		if s.err == nil {
			fn(e)
			s.err = e.err
		}
		s.queued[i] = nil
	}
	s.queued = s.queued[:0]
	send(e.enc, "endEncoding")
	if s.err != nil {
		send(cb, "release") // never committed
		return
	}
	send(cb, "commit")
	s.cbs = append(s.cbs, cb)
}

// Pending reports whether dispatches are waiting for Flush.
func (s *Stream) Pending() bool { return s.pending > 0 }

// Flush commits what is queued, waits for every committed command buffer,
// and reports the first error since the last Flush.
func (s *Stream) Flush() error {
	var err error
	if s.pending > 0 {
		s.d.do(func() {
			s.commitQueued()
			err = s.err
			s.gpu = 0
			for _, cb := range s.cbs {
				send(cb, "waitUntilCompleted")
				if ce := send(cb, "error"); ce != 0 && err == nil {
					err = fmt.Errorf("metal: command buffer: %w", nserror(ce))
				}
				s.gpu += time.Duration((sendF(cb, "GPUEndTime") - sendF(cb, "GPUStartTime")) * 1e9)
				send(cb, "release")
			}
		})
	}
	s.cbs, s.err, s.pending = s.cbs[:0], nil, 0
	return err
}

// GPUTime is the GPU execution time of the last Flush, summed over its
// command buffers (Metal's GPUStartTime to GPUEndTime; excludes encoding
// and scheduling).
func (s *Stream) GPUTime() time.Duration { return s.gpu }

// Dispatch runs p once in its own command buffer and waits.
func (p *Pipeline) Dispatch(grid, group [3]int, args ...Arg) error {
	return p.dev.Run(func(e *Encoder) { e.Dispatch(p, grid, group, args...) })
}
