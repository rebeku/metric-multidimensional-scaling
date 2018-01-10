package mms

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestEquidistantPoints(t *testing.T) {
	// Create dissimilarity matrix of equidistant points
	v := make([]float64, 9)
	for i := 0; i < 9; i++ {
		if i/3 == i%3 {
			v[i] = 0
		} else {
			v[i] = 1
		}
	}

	d := mat.NewSymDense(3, v)

	mms := NewMMS(2, 30, .01)
	X, sigma := mms.FitTransform(d)

	n, m := X.Dims()
	if n != 3 || m != 2 {
		t.Errorf("Expected a 3 * 2 matrix but found %d by %d", n, m)
	}
	if math.IsNaN(sigma) {
		t.Error("Unexpeced NaN for sigma")
	}
	if math.Abs(sigma) > .001 {
		t.Error("Expected stress close to 0 but found ", sigma)
	}
}

func TestEquidistantWithCentroid(t *testing.T) {
	// Create dissimilarity matrix of 3 equidistant points with a fourth point at the centroid
	a := 1 / math.Sqrt(3.0)

	v := make([]float64, 16)
	for i := 0; i < 15; i++ {
		if i%4 == 3 || i >= 12 {
			v[i] = a
		} else if i/4 == i%4 {
			v[i] = 0
		} else {
			v[i] = 1
		}
	}
	d := mat.NewSymDense(4, v)
	mms := NewMMS(2, 30, .01)
	X, sigma := mms.FitTransform(d)

	n, m := X.Dims()
	if n != 4 || m != 2 {
		t.Errorf("Expected a 4 * 2 matrix but found %d by %d", n, m)
	}
	if math.Abs(sigma) > .001 {
		t.Error("Expected stress close to 0 but found ", sigma)
	} else {
		t.Log("Stress: ", sigma)
	}
	// Verify that centroid is appropriate distance from all other points
	c := X.RowView(3)
	for i := 0; i < 3; i++ {
		u := X.RowView(i)
		di := mat.NewVecDense(2, nil)
		di.SubVec(u.(*mat.VecDense), c.(*mat.VecDense))
		norm := mat.Norm(di, 2)
		if math.IsNaN(norm) {
			t.Error("Unexpected NaN norm")
		}
		if math.Abs(norm-a) > .02 {
			t.Errorf("Expected distance to the centroid to be close to 1 / sqrt(3) for all outer points, but found %f for the %dth point", norm, i)
		}
		for j := i + 1; j < 3; j++ {
			w := X.RowView(j)
			di := mat.NewVecDense(2, nil)
			di.SubVec(w.(*mat.VecDense), u.(*mat.VecDense))
			norm := mat.Norm(di, 2)
			if math.Abs(norm-1.0) > .02 {
				t.Errorf("Expected distance between two outer points to be close to 1 but found %f for %dth and %dth point", norm, i, j)
			}
		}
	}
}

func TestImpossibleSetup(t *testing.T) {
	// Create dissimilarity matrix that should not work well in 2 dimensions
	v := make([]float64, 16)
	for i := 0; i < 15; i++ {
		if i%4 == 3 || i >= 12 {
			v[i] = .1
		} else if i/4 == i%4 {
			v[i] = 0
		} else {
			v[i] = 1
		}
	}
	d := mat.NewSymDense(4, v)
	mms := NewMMS(2, 30, .01)
	_, sigma := mms.FitTransform(d)
	if sigma < .5 || sigma > .52 {
		t.Error("Expected sigma of around .51, but found ", sigma)
	}
	if math.IsNaN(sigma) {
		t.Error("Unexpected NaN for sigma")
	}
}

func TestEmpty(t *testing.T) {
	d := mat.NewSymDense(0, nil)
	mms := NewMMS(2, 30, .01)
	_, stress := mms.FitTransform(d)

	if stress != 0.0 {
		t.Error("Expected 0 but found ", stress)
	}
}

func TestStressPerfect(t *testing.T) {
	a := math.Sqrt(2)
	u := make([]float64, 9) // Unit vectors along X, Y, Z axes
	v := make([]float64, 9) // Distances between vectors in u
	for i := 0; i < 9; i++ {
		if i/3 == i%3 {
			u[i] = 1
			v[i] = 0
		} else {
			u[i] = 0
			v[i] = a
		}
	}
	X := mat.NewDense(3, 3, u)
	Xdis := euclideanDists(X)
	d := mat.NewSymDense(3, v)
	sigma := stress(Xdis, d)
	if sigma != 0.0 {
		t.Error("Expected 0 but found ", sigma)
	}
}

func TestStressHigh(t *testing.T) {
	u := make([]float64, 9) // Unit vectors along X, Y, Z axes
	v := make([]float64, 9) // Ideal distances are 0 between all points
	for i := 0; i < 9; i++ {
		if i/3 == i%3 {
			u[i] = 1
		} else {
			u[i] = 0
		}
	}
	X := mat.NewDense(3, 3, u)
	d := mat.NewSymDense(3, v)
	Xdis := euclideanDists(X)
	sigma := stress(Xdis, d)
	if math.Abs(sigma-6.0) > 0.001 {
		t.Error("Expected 6 but found ", sigma)
	}
}

func TestRowDistance(t *testing.T) {

	v := make([]float64, 15)
	for i := 0; i < 15; i++ {
		v[i] = float64(i)
	}
	d := mat.NewDense(5, 3, v)
	if math.Abs(rowDistance(0, 1, d)-5.1961524) > .00001 {
		t.Error("Expected 5.1961524 but found ", rowDistance(0, 1, d))
	}
}

func TestReplaceZeros(t *testing.T) {
	v := []float64{0, 0, 0, 0, 0, 1, 0, 1, 0}
	d := mat.NewSymDense(3, v)
	replaceZeros(d)

	tests := []struct {
		i     int
		j     int
		value float64
	}{
		{
			i:     0,
			j:     0,
			value: .00001,
		}, {
			i:     0,
			j:     1,
			value: .00001,
		}, {
			i:     0,
			j:     2,
			value: .00001,
		}, {
			i:     1,
			j:     0,
			value: .00001,
		}, {
			i:     1,
			j:     1,
			value: .00001,
		}, {
			i:     1,
			j:     2,
			value: 1,
		}, {
			i:     2,
			j:     0,
			value: .00001,
		}, {
			i:     2,
			j:     1,
			value: 1,
		}, {
			i:     2,
			j:     2,
			value: .00001,
		},
	}
	for _, test := range tests {
		val := d.At(test.i, test.j)
		if val != test.value {
			t.Errorf("Expected %f at position %d, %d, but found %f", test.value, test.i, test.j, val)
		}
	}
}
