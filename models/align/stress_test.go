package align

import (
	"reflect"
	"testing"
)

func TestParseStress(t *testing.T) {
	for _, c := range []struct {
		in, clean string
		marked    []int
	}{
		{"I *never* said that.", "I never said that.", []int{1}},
		{"*Never*, ever *again*!", "Never, ever again!", []int{0, 2}},
		{"no markup here", "no markup here", nil},
		{"a * lone star", "a * lone star", nil},
	} {
		clean, marked := ParseStress(c.in)
		if clean != c.clean || !reflect.DeepEqual(marked, c.marked) {
			t.Errorf("ParseStress(%q) = %q %v, want %q %v", c.in, clean, marked, c.clean, c.marked)
		}
	}
}
