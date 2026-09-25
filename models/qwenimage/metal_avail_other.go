//go:build !(darwin && arm64)

package qwenimage

func metalAvailable() bool { return false }

func preflight(string, []int64, int, []condition, int, int, Options, func(string, ...any)) error {
	return nil
}
