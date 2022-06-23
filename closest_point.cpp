#include <stdio.h>
#include <math.h>

typedef double VEC[3];
typedef VEC MAT[3];

void solve(double *a, double *b, double *x, int n);  // linear solver

double dot(VEC a, VEC b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

void find_nearest_point(VEC p, VEC a[], VEC d[], int n) {
  MAT m = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  VEC b = {0, 0, 0};
  for (int i = 0; i < n; ++i) {
    double d2 = dot(d[i], d[i]), da = dot(d[i], a[i]);
    for (int ii = 0; ii < 3; ++ii) {
      for (int jj = 0; jj < 3; ++jj) m[ii][jj] += d[i][ii] * d[i][jj];
      m[ii][ii] -= d2;
      b[ii] += d[i][ii] * da - a[i][ii] * d2;
    }
  }
  solve(&m[0][0], b, p, 3);
}

void pp(VEC v, char *l, char *r) {
  printf("%s%.3lf, %.3lf, %.3lf%s", l, v[0], v[1], v[2], r);
} 

void pv(VEC v) { pp(v, "(", ")"); } 

void pm(MAT m) { for (int i = 0; i < 3; ++i) pp(m[i], "\n[", "]"); } 

// Verifier
double dist2(VEC p, VEC a, VEC d) {
  VEC pa = { a[0]-p[0], a[1]-p[1], a[2]-p[2] };
  double dpa = dot(d, pa);
  return dot(d, d) * dot(pa, pa) - dpa * dpa;
}

double sum_dist2(VEC p, VEC a[], VEC d[], int n) {
  double sum = 0;
  for (int i = 0; i < n; ++i) sum += dist2(p, a[i], d[i]);
  return sum;
}

// Check 26 nearby points and verify the provided one is nearest.
int is_nearest(VEC p, VEC a[], VEC d[], int n) {
  double min_d2 = 1e100;
  int ii = 2, jj = 2, kk = 2;
#define D 0.01
  for (int i = -1; i <= 1; ++i) 
    for (int j = -1; j <= 1; ++j)
      for (int k = -1; k <= 1; ++k) {
        VEC pp = { p[0] + D * i, p[1] + D * j, p[2] + D * k };
        double d2 = sum_dist2(pp, a, d, n);
        // Prefer provided point among equals.
        if (d2 < min_d2 || i == 0 && j == 0 && k == 0 && d2 == min_d2) {
          min_d2 = d2;
          ii = i; jj = j; kk = k;
        }
      }
  return ii == 0 && jj == 0 && kk == 0;
}

void normalize(VEC v) {
  double len = sqrt(dot(v, v));
  v[0] /= len;
  v[1] /= len;
  v[2] /= len;
}

int main(void) {
  VEC a[] = {{-14.2, 17, -1}, {1, 1, 1}, {2.3, 4.1, 9.8}, {1,2,3}};
  VEC d[] = {{1.3, 1.3, -10}, {12.1, -17.2, 1.1}, {19.2, 31.8, 3.5}, {4,5,6}};
  // VEC a[] = {{10,10,5}, {5,5,3}};
  // VEC d[] = {{10,10,3}, {5,5,5}};
  // VEC a[] = {{10,10,5}, {10,10,3}};
  // VEC d[] = {{5,5,3}, {5,5,5}};
  int n = 4;
  for (int i = 0; i < n; ++i) normalize(d[i]);
  VEC p;
  find_nearest_point(p, a, d, n);
  pv(p);
  printf("\n");
  if (!is_nearest(p, a, d, n)) printf("Woops. Not nearest.\n");
  return 0;
}

// From rosettacode (with bug fix: added a missing fabs())
#define mat_elem(a, y, x, n) (a + ((y) * (n) + (x)))

void swap_row(double *a, double *b, int r1, int r2, int n)
{
  double tmp, *p1, *p2;
  int i;

  if (r1 == r2) return;
  for (i = 0; i < n; i++) {
    p1 = mat_elem(a, r1, i, n);
    p2 = mat_elem(a, r2, i, n);
    tmp = *p1, *p1 = *p2, *p2 = tmp;
  }
  tmp = b[r1], b[r1] = b[r2], b[r2] = tmp;
}

void solve(double *a, double *b, double *x, int n)
{
#define A(y, x) (*mat_elem(a, y, x, n))
  int i, j, col, row, max_row, dia;
  double max, tmp;

  for (dia = 0; dia < n; dia++) {
    max_row = dia, max = fabs(A(dia, dia));
    for (row = dia + 1; row < n; row++)
      if ((tmp = fabs(A(row, dia))) > max) max_row = row, max = tmp; 
    swap_row(a, b, dia, max_row, n);
    for (row = dia + 1; row < n; row++) {
      tmp = A(row, dia) / A(dia, dia);
      for (col = dia+1; col < n; col++)
        A(row, col) -= tmp * A(dia, col);
      A(row, dia) = 0;
      b[row] -= tmp * b[dia];
    }
  }
  for (row = n - 1; row >= 0; row--) {
    tmp = b[row];
    for (j = n - 1; j > row; j--) tmp -= x[j] * A(row, j);
    x[row] = tmp / A(row, row);
  }
#undef A
}