package graph

// SetBlkDense forces the dense blocked conv kind on or off for a test and
// returns the restore func.
func SetBlkDense(on bool) func() {
	old := blkDense
	blkDense = on
	return func() { blkDense = old }
}
