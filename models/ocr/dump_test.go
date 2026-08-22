package ocr

import (
	"fmt"
	"os"
	"testing"

	"github.com/giraffesyo/ingot/graph"
	"github.com/giraffesyo/ingot/onnx"
	"github.com/giraffesyo/ingot/ops"
)

func TestDumpDetNodes(t *testing.T) {
	if os.Getenv("OCR_DUMP_NODES") == "" {
		t.Skip()
	}
	m, err := onnx.DecodeFile("../../testdata/ocr/" + os.Getenv("OCR_DUMP_NODES"))
	if err != nil {
		t.Skip(err)
	}
	g, err := graph.FromONNX(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range g.Nodes {
		line := n.OpType + " " + n.Name + " attrs=" + sprintAttrs(n) + " in:"
		for _, v := range n.Inputs {
			if v == nil {
				continue
			}
			if v.Const != nil {
				line += " const" + v.Const.Shape().String()
			} else {
				line += " " + v.Name
			}
		}
		line += " out:"
		for _, v := range n.Outputs {
			line += " " + v.Name
		}
		t.Log(line)
	}
}

func sprintAttrs(n *graph.Node) string {
	s := ""
	for k, v := range n.Attrs {
		s += k + "=" + fmtAny(v) + ","
	}
	return s
}

func fmtAny(a ops.Attr) string {
	switch a.Kind {
	case ops.KindInt:
		return fmt.Sprint(a.I)
	case ops.KindFloat:
		return fmt.Sprint(a.F)
	case ops.KindString:
		return a.S
	case ops.KindInts:
		return fmt.Sprint(a.Ints)
	case ops.KindFloats:
		return fmt.Sprint(a.Floats)
	}
	return "?"
}
