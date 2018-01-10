// rebeku/mms is available under the MIT Creative Commons license.
// See license.MD for details.

package mms

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

/*
Compute metric multidimensional scaling using the SMACOF algorithm

Metric multidimensional scaling takes a set of n objects whose distances
from each other are known.  Given these distances in the form of a symmetric
n * n similarity matrix, we assign each of these objects to a location in
r-dimensional space such that the Euclidean distance between the ith and
jth objects resembles the given distance between i and j as closely as possible.

This implementation borrows extensively from scikit-learn's multidimensional
scaling implementation, available at
"https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/manifold/mds.py",
with some simplifications.

Here is a brief overview of the algorithm, quoted directly from scikit-learn:

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.
    The SMACOF algorithm for metric MDS can summarized by the following steps:
    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    Further reading
    -----
    "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
    Groenen P. Springer Series in Statistics (1997)
    "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
    Psychometrika, 29 (1964)
    "Multidimensional scaling by optimizing goodness of fit to a nonmetric
    hypothesis" Kruskal, J. Psychometrika, 29, (1964)

Please see tests for sample usage.
*/

type MMS struct {
	dissimilarity *mat.SymDense
	nComponents   int
	maxIter       int
	epsilon       float64
	results       []*result
}

func NewMMS(nComponents, maxIter int, epsilon float64) *MMS {
	return &MMS{
		nComponents: nComponents,
		maxIter:     maxIter,
		epsilon:     epsilon,
	}
}

// FitTransform takes an n * n dissimilarity matrix d and
// and returns an n * MMS.r matrix of n points in r-dimensional space
// with distances between ith and jth points close to n[i][j]
// along with the final minimum value of the stress function.
func (mms *MMS) FitTransform(d *mat.SymDense) (*mat.Dense, float64) {
	mms.dissimilarity = d
	init := randomMatrix(d.Symmetric(), mms.nComponents)
	return mms.smacof(0, init)
}

//TODO: add multiple trials with concurrency
func (mms *MMS) smacof(nIter int, Z *mat.Dense) (*mat.Dense, float64) {
	X := Z
	Xdis := euclideanDists(X)
	oldSigma := -1.0
	sigma := stress(Xdis, mms.dissimilarity)
	for math.Abs(sigma-oldSigma) > mms.epsilon && nIter < mms.maxIter {
		guttmanTransformation(X, Xdis, mms.dissimilarity)
		Xdis = euclideanDists(X)
		sigma = stress(Xdis, mms.dissimilarity)
		nIter++
	}
	return X, sigma
}

// Compute the Euclidean distances between each row of X
// Xdis[i][j] = ||x[i] - x[j]||
func euclideanDists(X *mat.Dense) *mat.SymDense {
	n, _ := X.Dims()
	Xdis := mat.NewSymDense(n, nil)
	pairs := pairwiseIterator(n)
	for _, pair := range pairs {
		i, j := pair[0], pair[1]
		dij := rowDistance(i, j, X)
		Xdis.SetSym(i, j, dij)
	}
	return Xdis
}

// TODO: improve performance by optimizing like sklearn:
// dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
func rowDistance(i, j int, X *mat.Dense) float64 {
	_, n := X.Dims()
	ri := X.RowView(i).(*mat.VecDense)
	rj := X.RowView(j).(*mat.VecDense)
	u := mat.NewVecDense(n, nil)
	u.SubVec(ri, rj)
	return mat.Norm(u, 2)
}

// stress computes the sum of sqares of the differences between
// current and ideal distances for each point.
func stress(Xdis *mat.SymDense, d *mat.SymDense) float64 {
	sigma := 0.0
	n := Xdis.Symmetric()
	pairs := pairwiseIterator(n)
	for _, pair := range pairs {
		i, j := pair[0], pair[1]
		deltaij := d.At(i, j)
		dij := Xdis.At(i, j)
		diff := dij - deltaij
		sigma = sigma + diff*diff
	}
	return sigma
}

// Return all i, j pairs with i, j < n, and i < j
func pairwiseIterator(n int) [][]int {
	pairs := make([][]int, 0, n)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			pair := []int{i, j}
			pairs = append(pairs, pair)
		}
	}
	return pairs
}

func guttmanTransformation(X *mat.Dense, Xdis, dis *mat.SymDense) {
	n := Xdis.Symmetric()
	replaceZeros(Xdis)
	B := mat.NewDense(n, n, nil)
	B.DivElem(dis, Xdis)
	ratioSums := sumOfEachColumn(B)
	B.Scale(-1, B)
	addToDiagonals(B, ratioSums)
	X.Mul(B, X)
	X.Scale(1.0/float64(n), X)
}

func replaceZeros(Xdis *mat.SymDense) {
	n := Xdis.Symmetric()
	pairs := pairwiseIterator(n)
	for _, pair := range pairs {
		i, j := pair[0], pair[1]
		v := Xdis.At(i, j)
		if v == 0.0 {
			Xdis.SetSym(i, j, .00001)
		}
	}
	// Diagonal will be all zeros since every vector has
	// distance 0 to itself
	for i := 0; i < n; i++ {
		Xdis.SetSym(i, i, .00001)
	}
}

func sumOfEachColumn(B *mat.Dense) *mat.VecDense {
	n, _ := B.Dims()
	ones := oneByNMatrix(n)
	ones.Mul(ones, B)
	return ones.RowView(0).(*mat.VecDense)
}

func oneByNMatrix(n int) *mat.Dense {
	nums := make([]float64, n)
	for i := range nums {
		nums[i] = 1.0
	}
	return mat.NewDense(1, n, nums)
}

// Add numbers in V to diagonal of B
func addToDiagonals(B *mat.Dense, v *mat.VecDense) {
	n := v.Len()
	for i := 0; i < n; i++ {
		num := v.At(i, 0)
		B.Set(i, i, num)
	}
}

// create a random n * m matrix
func randomMatrix(n, m int) *mat.Dense {
	nums := make([]float64, n*m)
	for i := range nums {
		nums[i] = rand.Float64()
	}
	return mat.NewDense(n, m, nums)
}

type result struct {
	embedding *mat.Dense
	stress    float64
}
