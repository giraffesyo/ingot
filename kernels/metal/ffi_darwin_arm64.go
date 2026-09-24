package metal

import (
	"fmt"
	"runtime"
	"sync"
	"syscall"
	"unsafe"
)

//go:cgo_import_dynamic ingot_dlopen dlopen "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic ingot_dlsym dlsym "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic ingot_dlerror dlerror "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _ _ "/usr/lib/libSystem.B.dylib"

// Addresses of the assembly trampolines that jump to the imported symbols.
var dlopenABI0, dlsymABI0, dlerrorABI0 uintptr

// The runtime keeps these linkable (its "hall of shame": purego et al.; "do
// not remove or change the type signature"). Between them the goroutine is
// in _Gsyscall: never preempted, its stack never moved or shrunk, and its P
// free to run other goroutines.
//
//go:linkname entersyscall runtime.entersyscall
func entersyscall()

//go:linkname exitsyscall runtime.exitsyscall
func exitsyscall()

// callC switches SP to stack, loads x0..x7 from args, calls fn, restores SP
// and returns x0 (ffi_darwin_arm64.s).
func callC(fn uintptr, args *[8]uintptr, stack uintptr) uintptr

// C code must not run on a small, movable goroutine stack: each call gets a
// pooled 1 MiB mmap'd stack.
const cStackSize = 1 << 20

var cStacks = sync.Pool{New: func() any {
	b, err := syscall.Mmap(-1, 0, cStackSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(fmt.Sprintf("metal: C stack: %v", err))
	}
	return &b
}}

//go:nosplit
func ccall(fn uintptr, args *[8]uintptr, top uintptr) uintptr {
	entersyscall()
	r := callC(fn, args, top)
	exitsyscall()
	return r
}

// call invokes the C function at fn with up to 8 integer/pointer args.
// Pointers into Go memory must be kept alive by the caller.
func call(fn uintptr, args ...uintptr) uintptr {
	if len(args) > 8 {
		panic("metal: more than 8 integer arguments")
	}
	var a [8]uintptr
	copy(a[:], args)
	st := cStacks.Get().(*[]byte)
	top := (uintptr(unsafe.Pointer(&(*st)[0])) + cStackSize) &^ 15
	r := ccall(fn, &a, top)
	cStacks.Put(st)
	return r
}

const rtldNow, rtldGlobal = 0x2, 0x8

func cstr(s string) []byte { return append([]byte(s), 0) }

// cptr converts a C address (never Go memory) to a pointer.
func cptr(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

// gostr copies a NUL-terminated C string.
func gostr(p uintptr) string {
	if p == 0 {
		return ""
	}
	b := cptr(p)
	n := 0
	for *(*byte)(unsafe.Add(b, n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(b), n))
}

func dlopen(path string) (uintptr, error) {
	p := cstr(path)
	h := call(dlopenABI0, uintptr(unsafe.Pointer(&p[0])), rtldNow|rtldGlobal)
	runtime.KeepAlive(p)
	if h == 0 {
		return 0, fmt.Errorf("metal: dlopen %s: %s", path, gostr(call(dlerrorABI0)))
	}
	return h, nil
}

func dlsym(h uintptr, name string) (uintptr, error) {
	n := cstr(name)
	p := call(dlsymABI0, h, uintptr(unsafe.Pointer(&n[0])))
	runtime.KeepAlive(n)
	if p == 0 {
		return 0, fmt.Errorf("metal: dlsym %s: not found", name)
	}
	return p, nil
}
