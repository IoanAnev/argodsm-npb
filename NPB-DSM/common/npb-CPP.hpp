/**
 * NASA Advanced Supercomputing Parallel Benchmarks C++
 *
 * based on NPB 3.3.1
 *
 * original version and technical report:
 * http://www.nas.nasa.gov/Software/NPB/
 *
 * C++ version:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Alves de Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 */

#include "argo.hpp"
#include "wtime.hpp"

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

typedef int boolean;
typedef struct { double real; double imag; } dcomplex;

typedef struct {
	int locks{0};
	int barrs{0};
	double locktime{0.0};
	double barrtime{0.0};
} lock_barr_t;

#define TRUE	1
#define FALSE	0

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define	pow2(a) ((a)*(a))

/* forward prototype */
void wtime(double*);

/* old version of the complex number operations */
#define get_real(c) c.real
#define get_imag(c) c.imag
#define cadd(c,a,b) (c.real = a.real + b.real, c.imag = a.imag + b.imag)
#define csub(c,a,b) (c.real = a.real - b.real, c.imag = a.imag - b.imag)
#define cmul(c,a,b) (c.real = a.real * b.real - a.imag * b.imag, \
                     c.imag = a.real * b.imag + a.imag * b.real)
#define crmul(c,a,b) (c.real = a.real * b, c.imag = a.imag * b)

/* latest version of the complex number operations */
#define dcomplex_create(r,i) (dcomplex){r, i}
#define dcomplex_add(a,b) (dcomplex){(a).real+(b).real, (a).imag+(b).imag}
#define dcomplex_sub(a,b) (dcomplex){(a).real-(b).real, (a).imag-(b).imag}
#define dcomplex_mul(a,b) (dcomplex){((a).real*(b).real)-((a).imag*(b).imag),\
	((a).real*(b).imag)+((a).imag*(b).real)}
#define dcomplex_mul2(a,b) (dcomplex){(a).real*(b), (a).imag*(b)}
static inline dcomplex dcomplex_div(dcomplex z1, dcomplex z2){
	double a = z1.real;
	double b = z1.imag;
	double c = z2.real;
	double d = z2.imag;
	double divisor = c*c + d*d;
	double real = (a*c + b*d) / divisor;
	double imag = (b*c - a*d) / divisor;
	dcomplex result = (dcomplex){real, imag};
	return result;
}
#define dcomplex_div2(a,b) (dcomplex){(a).real/(b), (a).imag/(b)}
#define dcomplex_abs(x)    sqrt(((x).real*(x).real) + ((x).imag*(x).imag))
#define dconjg(x)          (dcomplex){(x).real, -1.0*(x).imag}

extern lock_barr_t argo_stats;
extern double lock_t1, lock_t2;
extern double barr_t1, barr_t2;

/**
 * @note: this will only work if the application is written for
 * only one thread per node to capture the global lock.
 */
static inline __attribute__((always_inline))
void argo_lock(argo::globallock::global_tas_lock *lock) {
	wtime(&lock_t1);
	lock->lock();
}

static inline __attribute__((always_inline))
void argo_unlock(argo::globallock::global_tas_lock *lock) {
	lock->unlock();
	wtime(&lock_t2);
	argo_stats.locktime += lock_t2 - lock_t1;
	argo_stats.locks++;
}

/**
 * @note: we need to overload this function and not supply a
 * default argument to make sure that it is inlined.
 */ 
static inline __attribute__((always_inline))
void argo_barrier() {
	wtime(&barr_t1);
	argo::barrier();
	wtime(&barr_t2);
	argo_stats.barrtime += barr_t2 - barr_t1;
	argo_stats.barrs++;
}

static inline __attribute__((always_inline))
void argo_barrier(int nthreads) {
	#pragma omp master
		wtime(&barr_t1);
	argo::barrier(nthreads);
	#pragma omp master
	{
		wtime(&barr_t2);
		argo_stats.barrtime += barr_t2 - barr_t1;
		argo_stats.barrs++;
	}
}

static inline
void print_argo_stats() {
	printf("#####################STATISTICS#########################\n");
	printf("Argo locks : %d, barriers : %d\n",
		argo_stats.locks, argo_stats.barrs);
	printf("Argo locktime : %.3lf sec., barriertime : %.3lf sec.\n",
		argo_stats.locktime, argo_stats.barrtime);
	printf("########################################################\n\n");
}

extern double randlc(double *, double);
extern void vranlc(int, double *, double, double *);
extern void timer_clear(int);
extern void timer_start(int);
extern void timer_stop(int);
extern double timer_read(int);

extern void c_print_results(char* name,
		char class_npb,
		int n1,
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* libversion,
		char* totalthreads,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand);
