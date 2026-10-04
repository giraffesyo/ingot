package trellis2

import (
	"fmt"
	"math"

	"github.com/giraffesyo/ingot/kernels/par"
	"github.com/giraffesyo/ingot/sparse"
	"github.com/giraffesyo/ingot/tensor"
)

// Camera is the view the input image was taken from, for models with
// projected image features (Pixal3D): a pinhole camera on the grid's z
// axis looking at the object.
type Camera struct {
	FovX      float64 // horizontal field of view, radians
	Distance  float64 // camera to object centre, in object units
	MeshScale float64 // object scale; 1 fills the unit cube
}

// DefaultFovX is the field of view the reference pipeline assumes when it
// is given none.
const DefaultFovX = 0.8575560450553894

// CameraFromFov places the camera so that the object's cube spans the
// image width at its near side — the distance the reference derives from a
// field of view (its distance_from_fov with no pixel margin).
func CameraFromFov(fovX, meshScale float64) Camera {
	if meshScale == 0 {
		meshScale = 1
	}
	return Camera{FovX: fovX, Distance: 0.5 / (math.Tan(fovX/2) * meshScale), MeshScale: meshScale}
}

// project returns where the centre of cell c of an R³ grid lands in the
// image, in grid_sample's normalised coordinates ([-1, 1] across the
// image, +v down). The grid spans [-1, 1]³ before the mesh scale; x maps
// to image x, y to image up, and z towards the camera. imageRes enters
// only through the reference's half-pixel offset.
func (cam Camera) project(c sparse.Coord, R, imageRes int) (u, v float32) {
	lin := func(i int32) float64 {
		if R == 1 {
			return -1
		}
		return -1 + 2*float64(i)/float64(R-1)
	}
	s := 2 * cam.MeshScale
	x, y, depth := lin(c[0])/s, lin(c[1])/s, cam.Distance-lin(c[2])/s
	res := float64(imageRes)
	f := 16 / math.Tan(cam.FovX/2) * res / 32
	px := f*x/(depth+1e-8) + res/2
	py := -f*y/(depth+1e-8) + res/2
	return float32((px+0.5)/res*2 - 1), float32((py+0.5)/res*2 - 1)
}

// bilinear samples a w×h map at normalised (u, v) the way
// grid_sample(align_corners=False, padding_mode="border") does, adding
// each of the four texels' contribution through add(x, y, weight).
func bilinear(u, v float32, w, h int, add func(x, y int, wt float32)) {
	fx := min(float32(w-1), max(0, ((u+1)*float32(w)-1)/2))
	fy := min(float32(h-1), max(0, ((v+1)*float32(h)-1)/2))
	x0, y0 := int(fx), int(fy)
	x1, y1 := min(x0+1, w-1), min(y0+1, h-1)
	ax, ay := fx-float32(x0), fy-float32(y0)
	add(x0, y0, (1-ax)*(1-ay))
	add(x1, y0, ax*(1-ay))
	add(x0, y1, (1-ax)*ay)
	add(x1, y1, ax*ay)
}

// ProjectFeatures samples image features at the cells' projections:
// feats is the encoder's patch tokens [n·n, C] (row-major, top row first)
// from an imageRes-pixel image, cells are cells of an R³ grid. With up,
// the upsampler's features at the same points are appended, doubling the
// channels. The result is [len(cells), C] or [len(cells), 2C].
func ProjectFeatures(cam Camera, cells []sparse.Coord, R, imageRes int, feats *tensor.Tensor, up *Upsampled) (*tensor.Tensor, error) {
	fs := feats.Shape()
	if len(fs) != 2 {
		return nil, fmt.Errorf("trellis2: patch features must be [patches, channels], got %v", fs)
	}
	n := int(math.Round(math.Sqrt(float64(fs[0]))))
	if n*n != fs[0] {
		return nil, fmt.Errorf("trellis2: %d patch tokens do not form a square map", fs[0])
	}
	C := fs[1]
	W := C
	if up != nil {
		if up.channels != C {
			return nil, fmt.Errorf("trellis2: upsampled features have %d channels, patch features %d", up.channels, C)
		}
		W = 2 * C
	}
	out := tensor.New(tensor.F32, len(cells), W)
	ff, of := feats.F32(), out.F32()
	scratch := make([][]float32, par.Workers())
	par.For(len(cells), 16, func(i, wk int) {
		u, v := cam.project(cells[i], R, imageRes)
		row := of[i*W : (i+1)*W]
		bilinear(u, v, n, n, func(x, y int, wt float32) {
			src := ff[(y*n+x)*C : (y*n+x+1)*C]
			for c, s := range src {
				row[c] += wt * s
			}
		})
		if up == nil {
			return
		}
		if scratch[wk] == nil {
			scratch[wk] = make([]float32, C)
		}
		px := scratch[wk]
		bilinear(u, v, up.size, up.size, func(x, y int, wt float32) {
			up.at(x, y, px)
			for c, s := range px {
				row[C+c] += wt * s
			}
		})
	})
	return out, nil
}
