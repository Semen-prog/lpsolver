#ifndef LUDEC_HPP
#define LUDEC_HPP

typedef struct mdata {
        int* offsets;
        int* columns;
        double* vals;
} mdata;

double* ax_equals_b_solver(int, int, mdata, const double*);
mdata csc_to_csr(int, int, const int*, const int*, const double*);

#endif // LUDEC_HPP
