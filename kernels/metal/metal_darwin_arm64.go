package metal

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
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

// Available reports whether a Metal device can be opened.
func Available() bool { _, err := Open(); return err == nil }

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

// do runs f on the device thread and waits for it.
func (d *Device) do(f func()) {
	done := make(chan struct{})
	d.jobs <- func() { f(); close(done) }
	<-done
}

// Pipeline is a compiled compute kernel.
type Pipeline struct {
	dev *Device
	pso uintptr
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
		p = &Pipeline{dev: d, pso: pso, MaxThreads: int(send(pso, "maxTotalThreadsPerThreadgroup"))}
	})
	return p, err
}

// Buffer is GPU memory in shared storage: on unified-memory devices the CPU
// reads and writes it in place through Bytes, with no copies.
type Buffer struct {
	dev *Device
	id  uintptr
	n   int
}

const storageModeShared = 0

// NewBuffer allocates n bytes of shared, zeroed GPU memory.
func (d *Device) NewBuffer(n int) (*Buffer, error) {
	b := &Buffer{dev: d, n: n}
	d.do(func() { b.id = send(d.id, "newBufferWithLength:options:", uintptr(n), storageModeShared) })
	if b.id == 0 {
		return nil, fmt.Errorf("metal: cannot allocate %d-byte buffer", n)
	}
	return b, nil
}

// Bytes is the buffer's memory, valid until Release.
func (b *Buffer) Bytes() []byte {
	var p uintptr
	b.dev.do(func() { p = send(b.id, "contents") })
	return unsafe.Slice((*byte)(cptr(p)), b.n)
}

// Release frees the buffer.
func (b *Buffer) Release() {
	if b.id != 0 {
		b.dev.do(func() { send(b.id, "release") })
		b.id = 0
	}
}

// Arg is a kernel argument: a *Buffer or a small constant ([]byte, set with
// setBytes, at most 4 KiB).
type Arg any

type mtlSize struct{ w, h, d uint64 }

// Dispatch runs p over a grid of threads (non-uniform threadgroups: the grid
// need not be a multiple of group) with args bound to indices 0, 1, …, and
// waits for completion.
func (p *Pipeline) Dispatch(grid, group [3]int, args ...Arg) error {
	d := p.dev
	var err error
	d.do(func() {
		cb := send(d.queue, "commandBuffer")
		enc := send(cb, "computeCommandEncoder")
		send(enc, "setComputePipelineState:", p.pso)
		for i, a := range args {
			switch v := a.(type) {
			case *Buffer:
				send(enc, "setBuffer:offset:atIndex:", v.id, 0, uintptr(i))
			case []byte:
				if len(v) == 0 || len(v) > 4096 {
					err = fmt.Errorf("metal: constant arg %d is %d bytes (want 1..4096)", i, len(v))
					send(enc, "endEncoding")
					return
				}
				send(enc, "setBytes:length:atIndex:", uintptr(unsafe.Pointer(&v[0])), uintptr(len(v)), uintptr(i))
				runtime.KeepAlive(v)
			default:
				err = fmt.Errorf("metal: arg %d: unsupported type %T", i, a)
				send(enc, "endEncoding")
				return
			}
		}
		g := mtlSize{uint64(grid[0]), uint64(grid[1]), uint64(grid[2])}
		t := mtlSize{uint64(group[0]), uint64(group[1]), uint64(group[2])}
		// MTLSize is 24 bytes: the arm64 ABI passes it by reference.
		send(enc, "dispatchThreads:threadsPerThreadgroup:", uintptr(unsafe.Pointer(&g)), uintptr(unsafe.Pointer(&t)))
		send(enc, "endEncoding")
		send(cb, "commit")
		send(cb, "waitUntilCompleted")
		if e := send(cb, "error"); e != 0 {
			err = fmt.Errorf("metal: command buffer: %w", nserror(e))
		}
	})
	return err
}
