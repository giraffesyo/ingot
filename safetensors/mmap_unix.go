//go:build unix

package safetensors

import (
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

// mapFile maps path read-only. Empty files fall back to a read (mmap of
// length 0 fails).
func mapFile(path string) ([]byte, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if st.Size() == 0 {
		return readFile(path)
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, err
	}
	return b, func() error { return syscall.Munmap(b) }, nil
}

// willNeed asks the kernel to read b ahead (MADV_WILLNEED; best effort).
func willNeed(b []byte) {
	if len(b) > 0 {
		_ = unix.Madvise(b, unix.MADV_WILLNEED)
	}
}
