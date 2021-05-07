/**
 * NASA Advanced Supercomputing Parallel Benchmarks C++
 * 
 * based on NPB 3.3.1
 *
 * original version and technical report:
 * http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors:
 *     M. Yarrow
 *     C. Kuszmaul
 *
 * C++ version:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Alves de Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * OpenMP version:
 *      Júnior Löff <loffjh@gmail.com>
 * 
 * ArgoDSM/OpenMP version:
 *      Ioannis Anevlavis <ioannis.anevlavis@etascale.com>
 */

#include "argo.hpp"
#include "omp.h"
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * note: please observe that in the routine conj_grad three 
 * implementations of the sparse matrix-vector multiply have
 * been supplied. the default matrix-vector multiply is not
 * loop unrolled. the alternate implementations are unrolled
 * to a depth of 2 and unrolled to a depth of 8. please
 * experiment with these to find the fastest for your particular
 * architecture. if reporting timing results, any of these three may
 * be used without penalty.
 * ---------------------------------------------------------------------
 * class specific parameters: 
 * it appears here for reference only.
 * these are their values, however, this info is imported in the npbparams.h
 * include file, which is written by the sys/setparams.c program.
 * ---------------------------------------------------------------------
 */
#define	NZ 		(NA*(NONZER+1)*(NONZER+1)+NA*(NONZER+2))
#define T_INIT 		0
#define T_BENCH 	1
#define T_CONJ_GRAD 	2
#define T_LAST 		3

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int colidx[NZ+1];
static int rowstr[NA+1+1];
static int iv[2*NA+1+1];
static int arow[NZ+1];
static int acol[NZ+1];
static double aelt[NZ+1];
static double a[NZ+1];
static double v[NA+1+1];
#else
static int (*colidx)=(int*)malloc(sizeof(int)*(NZ+1));
static int (*rowstr)=(int*)malloc(sizeof(int)*(NA+1+1));
static int (*iv)=(int*)malloc(sizeof(int)*(2*NA+1+1));
static int (*arow)=(int*)malloc(sizeof(int)*(NZ+1));
static int (*acol)=(int*)malloc(sizeof(int)*(NZ+1));
static double (*aelt)=(double*)malloc(sizeof(double)*(NZ+1));
static double (*a)=(double*)malloc(sizeof(double)*(NZ+1));
static double (*v)=(double*)malloc(sizeof(double)*(NA+1+1));
#endif
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
static boolean timeron;

static double (*x);
static double (*z);
static double (*p);
static double (*q);
static double (*r);
static double (*gtemps);

static int workrank;
static int numtasks;
static int nthreads;

/* function prototypes */
static void conj_grad (int colidx[],
		int rowstr[],
		double x[], 
		double z[],
		double a[],
		double p[],
		double q[],
		double r[], 
		double *rnorm);
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int nonzer,
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		double rcond,
		int arow[],
		int acol[],
		double aelt[],
		double v[],
		int iv[],
		double shift );
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int arow[],
		int acol[],
		double aelt[],
		int firstrow,
		int lastrow,
		double x[],
		boolean mark[],
		int nzloc[],
		int nnza);
static void sprnvc(int n,
		int nz,
		double v[],
		int iv[],
		int nzloc[],
		int mark[]);
static int icnvrt(double x,
		int ipwr2);
static void vecset(int n,
		double v[],
		int iv[],
		int *nzv,
		int i,
		double val);
static void distribute(int& beg,
		int& end,
		const int& loop_size,
		const int& beg_offset,
		const int& less_equal);

/* cg */
int main(int argc, char **argv)
{
	/*
	 * -------------------------------------------------------------------------
	 * initialize argodsm
	 * -------------------------------------------------------------------------
	 */
	argo::init(0.5*1024*1024*1024UL);
	/*
	 * -------------------------------------------------------------------------
	 * fetch workrank, number of nodes, and number of threads
	 * -------------------------------------------------------------------------
	 */ 
	workrank = argo::node_id();
	numtasks = argo::number_of_nodes();

	#pragma omp parallel
	{
		#if defined(_OPENMP)
			#pragma omp master
			nthreads = omp_get_num_threads();
		#endif /* _OPENMP */
	}
	/*
	 * -------------------------------------------------------------------------
	 * move global arrays allocation here, since this is a collective operation
	 * -------------------------------------------------------------------------
	 */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	if(workrank == 0){
		printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
	}
#endif
	x = argo::conew_array<double>(NA+2+1);
	z = argo::conew_array<double>(NA+2+1);
	p = argo::conew_array<double>(NA+2+1);
	q = argo::conew_array<double>(NA+2+1);
	r = argo::conew_array<double>(NA+2+1);
	gtemps = argo::conew_array<double>(2*numtasks);
	/*
	 * -------------------------------------------------------------------------
	 * continue with the local allocations
	 * -------------------------------------------------------------------------
	 */
	int	i, j, k, it;
	double zeta;
	double rnorm;
	double norm_temp11, norm_temp12;
	double t, mflops, tmax;
	char class_npb;
	boolean verified;
	double zeta_verify_value, epsilon, err;

	char *t_names[T_LAST];

	for(i=0; i<T_LAST; i++){
		timer_clear(i);
	}

	FILE* fp;
	if((fp = fopen("timer.flag", "r")) != NULL){
		timeron = TRUE;
		t_names[T_INIT] = (char*)"init";
		t_names[T_BENCH] = (char*)"benchmk";
		t_names[T_CONJ_GRAD] = (char*)"conjgd";
		fclose(fp);
	}else{
		timeron = FALSE;
	}

	timer_start(T_INIT);

	firstrow = 1;
	lastrow  = NA;
	firstcol = 1;
	lastcol  = NA;

	if (NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0) {
		class_npb = 'S';
		zeta_verify_value = 8.5971775078648;
	} else if (NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0) {
		class_npb = 'W';
		zeta_verify_value = 10.362595087124;
	} else if (NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0) {
		class_npb = 'A';
		zeta_verify_value = 17.130235054029;
	} else if (NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0) {
		class_npb = 'B';
		zeta_verify_value = 22.712745482631;
	} else if (NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0) {
		class_npb = 'C';
		zeta_verify_value = 28.973605592845;
	} else if (NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500.0) {
		class_npb = 'D';
		zeta_verify_value = 52.514532105794;
	} else if (NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1.5e3) {
		class_npb = 'E';
		zeta_verify_value = 77.522164599383;
	} else if (NA == 54000000 && NONZER == 31 && NITER == 100 && SHIFT == 5.0e3) {
		class_npb = 'F';
		zeta_verify_value = 107.3070826433;
	} else {
		class_npb = 'U';
	}

	if (workrank == 0){ 
		printf("\n\n NAS Parallel Benchmarks 4.0 Parallel C++ version with OpenMP - CG Benchmark\n\n");
		printf(" Size: %11d\n", NA);
		printf(" Iterations: %5d\n", NITER);
	}

	naa = NA;
	nzz = NZ;

	/* initialize random number generator */
	tran    = 314159265.0;
	amult   = 1220703125.0;
	zeta    = randlc( &tran, amult );

	makea(naa,
			nzz,
			a,
			colidx,
			rowstr,
			NONZER,
			firstrow,
			lastrow,
			firstcol,
			lastcol, 
			RCOND,
			arow,
			acol,
			aelt,
			v,
			iv,
			SHIFT);

	/*
	 * ---------------------------------------------------------------------
	 * note: as a result of the above call to makea:
	 * values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
	 * values of colidx which are col indexes go from firstcol --> lastcol
	 * so:
	 * shift the col index vals from actual (firstcol --> lastcol) 
	 * to local, i.e., (1 --> lastcol-firstcol+1)
	 * ---------------------------------------------------------------------
	 */
	#pragma omp parallel private(it,i,j,k)	
	{
		int beg, end;

		#pragma omp for nowait
		for (j = 1; j <= lastrow - firstrow + 1; j++) {
			for (k = rowstr[j]; k < rowstr[j+1]; k++) {
				colidx[k] = colidx[k] - firstcol + 1;
			}
		}

		/* set starting vector to (1, 1, .... 1) */
		distribute(beg, end, NA+1, 1, 1);

		#pragma omp for nowait
		for (i = beg; i <= end; i++) {
			x[i] = 1.0;
		}
		
		#pragma omp single
			zeta  = 0.0;

		/*
		 * -------------------------------------------------------------------
		 * ---->
		 * do one iteration untimed to init all code and data page tables
		 * ----> (then reinit, start timing, to niter its)
		 * -------------------------------------------------------------------*/

		for (it = 1; it <= 1; it++) {
			/* the call to the conjugate gradient routine */
			conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);

			/*
			 * --------------------------------------------------------------------
			 * zeta = shift + 1/(x.z)
			 * so, first: (x.z)
			 * also, find norm of z
			 * so, first: (z.z)
			 * --------------------------------------------------------------------
			 */
			distribute(beg, end, lastcol-firstcol+1, 1, 1);

			#pragma omp single
			{	
				norm_temp11 = 0.0;
				norm_temp12 = 0.0;
			}

			#pragma omp for reduction(+:norm_temp11,norm_temp12)
			for (j = beg; j <= end; j++) {
				norm_temp11 = norm_temp11 + x[j]*z[j];
				norm_temp12 = norm_temp12 + z[j]*z[j];
			}
			
			#pragma omp single
				norm_temp12 = 1.0 / sqrt( norm_temp12 );

			/* normalize z to obtain x */
			#pragma omp for
			for (j = beg; j <= end; j++) {
				x[j] = norm_temp12*z[j];
			}

		} /* end of do one iteration untimed */

		/* set starting vector to (1, 1, .... 1) */
		distribute(beg, end, NA+1, 1, 1);

		#pragma omp for nowait
		for (i = beg; i <= end; i++) {
			x[i] = 1.0;
		}

		#pragma omp single    
			zeta  = 0.0;

	} /* end parallel */
	argo::barrier();

	timer_stop(T_INIT);

	if (workrank == 0){  printf(" Initialization time = %15.3f seconds\n", timer_read(T_INIT)); }
	
	timer_start(T_BENCH);

	/*
	 * --------------------------------------------------------------------
	 * ---->
	 * main iteration for inverse power method
	 * ---->
	 * --------------------------------------------------------------------
	 */
	#pragma omp parallel private(it,i,j,k)
	{
		int beg, end;
		
		for (it = 1; it <= NITER; it++) {
			/* the call to the conjugate gradient routine */
			#pragma omp master
			if(timeron){timer_start(T_CONJ_GRAD);}
			conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
			#pragma omp master
			if(timeron){timer_stop(T_CONJ_GRAD);}

			/*
			 * --------------------------------------------------------------------
			 * zeta = shift + 1/(x.z)
			 * so, first: (x.z)
			 * also, find norm of z
			 * so, first: (z.z)
			 * --------------------------------------------------------------------
			 */
			distribute(beg, end, lastcol-firstcol+1, 1, 1);

			#pragma omp single
			{	
				norm_temp11 = 0.0;
				norm_temp12 = 0.0;
			}

			#pragma omp for reduction(+:norm_temp11,norm_temp12)
			for (j = beg; j <= end; j++) {
				norm_temp11 += x[j]*z[j];
				norm_temp12 += z[j]*z[j];
			}
			#pragma omp single
			{
				gtemps[workrank] = norm_temp11;
				gtemps[workrank+numtasks] = norm_temp12;
			}
			argo::barrier(nthreads);

			#pragma omp single
			{
				for (j = 0; j < numtasks; j++) {
					if (j != workrank) {
						norm_temp11 += gtemps[j];
						norm_temp12 += gtemps[j+numtasks];
					}
				}

				norm_temp12 = 1.0 / sqrt(norm_temp12);
				zeta = SHIFT + 1.0 / norm_temp11;

				if (workrank == 0){ 
					if(it==1){printf("\n   iteration           ||r||                 zeta\n");}
					printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);
				}
			}

			/* normalize z to obtain x */
			#pragma omp for 
			for (j = beg; j <= end; j++) {
				x[j] = norm_temp12*z[j];
			}
			argo::barrier(nthreads);
		} /* end of main iter inv pow meth */
	} /* end parallel */
	timer_stop(T_BENCH);
	
	/*
	 * --------------------------------------------------------------------
	 * end of timed section
	 * --------------------------------------------------------------------
	 */

	t = timer_read(T_BENCH);

	if (workrank == 0){ 
		printf(" Benchmark completed\n");

		epsilon = 1.0e-10;
		if(class_npb != 'U'){
			err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
			if(err <= epsilon){
				verified = TRUE;
				printf(" VERIFICATION SUCCESSFUL\n");
				printf(" Zeta is    %20.13e\n", zeta);
				printf(" Error is   %20.13e\n", err);
			}else{
				verified = FALSE;
				printf(" VERIFICATION FAILED\n");
				printf(" Zeta                %20.13e\n", zeta);
				printf(" The correct zeta is %20.13e\n", zeta_verify_value);
			}
		}else{
			verified = FALSE;
			printf(" Problem size unknown\n");
			printf(" NO VERIFICATION PERFORMED\n");
		}
		if(t != 0.0){
			mflops = (double)(2.0*NITER*NA)
				* (3.0+(double)(NONZER*(NONZER+1))
						+ 25.0
						* (5.0+(double)(NONZER*(NONZER+1)))+3.0)
				/ t / 1000000.0;
		}else{
			mflops = 0.0;
		}
		c_print_results((char*)"CG",
				class_npb,
				NA,
				0,
				0,
				NITER,
				t,
				mflops,
				(char*)"          floating point",
				verified,
				(char*)NPBVERSION,
				(char*)COMPILETIME,
				(char*)COMPILERVERSION,
				(char*)LIBVERSION,
				std::getenv("OMP_NUM_THREADS"),
				(char*)CS1,
				(char*)CS2,
				(char*)CS3,
				(char*)CS4,
				(char*)CS5,
				(char*)CS6,
				(char*)CS7);

		/*
		* ---------------------------------------------------------------------
		* more timers
		* ---------------------------------------------------------------------
		*/
		if(timeron){
			tmax = timer_read(T_BENCH);
			if(tmax == 0.0){tmax = 1.0;}
			printf("  SECTION   Time (secs)\n");
			for(i = 0; i < T_LAST; i++){
				t = timer_read(i);
				if(i == T_INIT){
					printf("  %8s:%9.3f\n", t_names[i], t);
				}else{
					printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
					if(i == T_CONJ_GRAD){
						t = tmax - t;
						printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t*100.0/tmax);
					}
				}
			}
		}
	}
	/*
	 * -------------------------------------------------------------------------
	 * deallocate data structures
	 * -------------------------------------------------------------------------
	 */
	argo::codelete_array(x);
	argo::codelete_array(z);
	argo::codelete_array(p);
	argo::codelete_array(q);
	argo::codelete_array(r);
	argo::codelete_array(gtemps);
	/*
	 * -------------------------------------------------------------------------
	 * finalize argodsm
	 * -------------------------------------------------------------------------
	 */
	argo::finalize();

	return 0;
}

/*
 * ---------------------------------------------------------------------
 * floating point arrays here are named as in NPB1 spec discussion of 
 * CG algorithm
 * ---------------------------------------------------------------------
 */
static void conj_grad (int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double *rnorm ){
	int j, k;
	int cgit, cgitmax = 25;
	static double d, sum, rho, rho0, alpha, beta;

	int beg_naa, end_naa;
	int beg_row, end_row;
	int beg_col, end_col;

	distribute(beg_naa, end_naa, naa+1, 1, 1);
	distribute(beg_row, end_row, lastrow-firstrow+1, 1, 1);
	distribute(beg_col, end_col, lastcol-firstcol+1, 1, 1);

	#pragma omp single nowait
		rho = 0.0;

	/* initialize the CG algorithm */
	#pragma omp for
	for (j = beg_naa; j <= end_naa; j++) {
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
	}
	argo::barrier(nthreads);

	/*
	 * --------------------------------------------------------------------
	 * rho = r.r
	 * now, obtain the norm of r: First, sum squares of r elements locally...
	 * --------------------------------------------------------------------
	 */
	#pragma omp for reduction(+:rho)
	for (j = beg_col; j <= end_col; j++) {
		rho += x[j]*x[j];
	}
	#pragma omp master
	gtemps[workrank] = rho;
	argo::barrier(nthreads);

	#pragma omp single
	for (j = 0; j < numtasks; j++)
		if (j != workrank)
			rho += gtemps[j];

	/* the conj grad iteration loop */
    	for (cgit = 1; cgit <= cgitmax; cgit++) {
		#pragma omp single nowait
		{	
			d = 0.0;
			/*
			 * --------------------------------------------------------------------
			 * save a temporary of rho
			 * --------------------------------------------------------------------
			 */
			rho0 = rho;
			rho = 0.0;
		}
      
		/*
		 * ---------------------------------------------------------------------
		 * q = A.p
		 * the partition submatrix-vector multiply: use workspace w
		 * ---------------------------------------------------------------------
		 * 
		 * note: this version of the multiply is actually (slightly: maybe %5) 
		 * faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
		 * below. on the Cray t3d, the reverse is TRUE, i.e., the 
		 * unrolled-by-two version is some 10% faster.  
		 * the unrolled-by-8 version below is significantly faster
		 * on the Cray t3d - overall speed of code is 1.5 times faster.
		 */

		/* rolled version */      
		#pragma omp for private(sum,k) nowait
		for (j = beg_row; j <= end_row; j++) {
			sum = 0.0;
			for (k = rowstr[j]; k < rowstr[j+1]; k++) {
				sum += a[k]*p[colidx[k]];
		    	}
			q[j] = sum;
		}
		argo::barrier(nthreads);
		
		/*
		 * --------------------------------------------------------------------
		 * obtain p.q
		 * --------------------------------------------------------------------
		 */
		#pragma omp for reduction(+:d)
		for (j = beg_col; j <= end_col; j++) {
			d += p[j]*q[j];
		}
		#pragma omp master
		gtemps[workrank] = d;
		argo::barrier(nthreads);

		#pragma omp single
		for (j = 0; j < numtasks; j++)
			if (j != workrank)
				d += gtemps[j];

		/*
		 * --------------------------------------------------------------------
		 * obtain alpha = rho / (p.q)
		 * -------------------------------------------------------------------
		 */
		#pragma omp single	
			alpha = rho0 / d;

		/*
		 * ---------------------------------------------------------------------
		 * obtain z = z + alpha*p
		 * and    r = r - alpha*q
		 * ---------------------------------------------------------------------
		 */
		#pragma omp for
		for (j = beg_col; j <= end_col; j++) {
			z[j] += alpha*p[j];
			r[j] -= alpha*q[j];
		}
		/* why correctness fails for bind-all when this barrier is removed? */
		argo::barrier(nthreads);

		/*
		 * ---------------------------------------------------------------------
		 * rho = r.r
		 * now, obtain the norm of r: first, sum squares of r elements locally...
		 * ---------------------------------------------------------------------
		 */
		#pragma omp for reduction(+:rho)
		for (j = beg_col; j <= end_col; j++) {
			rho += r[j]*r[j];
		}
		#pragma omp master
		gtemps[workrank] = rho;
		argo::barrier(nthreads);

		#pragma omp single
		for (j = 0; j < numtasks; j++)
			if (j != workrank)
				rho += gtemps[j];

		/*
		 * ---------------------------------------------------------------------
		 * obtain beta
		 * ---------------------------------------------------------------------
		 */
		#pragma omp single	
			beta = rho / rho0;

		/*
		 * ---------------------------------------------------------------------
		 * p = r + beta*p
		 * ---------------------------------------------------------------------
		 */
		#pragma omp for
		for (j = beg_col; j <= end_col; j++) {
			p[j] = r[j] + beta*p[j];
		}
		argo::barrier(nthreads);
	} /* end of do cgit=1, cgitmax */

	/*
	 * ---------------------------------------------------------------------
	 * compute residual norm explicitly: ||r|| = ||x - A.z||
	 * first, form A.z
	 * the partition submatrix-vector multiply
	 * ---------------------------------------------------------------------
	 */
	#pragma omp single nowait
		sum = 0.0;
    
	#pragma omp for private(d, k) nowait
	for (j = beg_row; j <= end_row; j++) {
		d = 0.0;
		for (k = rowstr[j]; k <= rowstr[j+1]-1; k++) {
			d += a[k]*z[colidx[k]];
		}
		r[j] = d;
	}

	/*
	 * ---------------------------------------------------------------------
	 * at this point, r contains A.z
	 * ---------------------------------------------------------------------
	 */
	#pragma omp for reduction(+:sum) private(d)
	for (j = beg_col; j <= end_col; j++) {
		d = x[j] - r[j];
		sum += d*d;
	}
	#pragma omp master
	gtemps[workrank] = sum;
	argo::barrier(nthreads);

	#pragma omp single
	for (j = 0; j < numtasks; j++)
		if (j != workrank)
			sum += gtemps[j];

	#pragma omp single
		*rnorm = sqrt(sum);
}

/*
 * ---------------------------------------------------------------------
 * generate the test problem for benchmark 6
 * makea generates a sparse matrix with a
 * prescribed sparsity distribution
 *
 * parameter    type        usage
 *
 * input
 *
 * n            i           number of cols/rows of matrix
 * nz           i           nonzeros as declared array size
 * rcond        r*8         condition number
 * shift        r*8         main diagonal shift
 *
 * output
 *
 * a            r*8         array for nonzeros
 * colidx       i           col indices
 * rowstr       i           row pointers
 *
 * workspace
 *
 * iv, arow, acol i
 * aelt           r*8
 * ---------------------------------------------------------------------
 */
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int nonzer,
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		double rcond,
		int arow[],
		int acol[],
		double aelt[],
		double v[],
		int iv[],
		double shift){
	int jcol, i, nnza, iouter, ivelt, ivelt1, irow, nzv;
	double size, ratio, scale;

	/*
	 * --------------------------------------------------------------------
	 * nonzer is approximately  (int(sqrt(nnza /n)));
	 * --------------------------------------------------------------------
	 */
	size = 1.0;
	ratio = pow(rcond, (1.0 / (double)n));
	nnza = 0;

	/*
	 * --------------------------------------------------------------------
	 * initialize colidx(n+1 .. 2n) to zero.
	 * used by sprnvc to mark nonzero positions
	 * --------------------------------------------------------------------
	 */
	#pragma omp parallel for
	for (i = 1; i <= n; i++) {
		colidx[n+i] = 0;
	}
	/*
	 * -------------------------------------------------------------------
	 * generate nonzero positions and save for the use in sparse
	 * -------------------------------------------------------------------
	 */
	for (iouter = 1; iouter <= n; iouter++) {
		nzv = nonzer;
		sprnvc(n, nzv, v, iv, &(colidx[0]), &(colidx[n]));
		vecset(n, v, iv, &nzv, iouter, 0.5);
		for (ivelt = 1; ivelt <= nzv; ivelt++){
			jcol = iv[ivelt];
			if (jcol >= firstcol && jcol <= lastcol) {
				scale = size * v[ivelt];
				for (ivelt1 = 1; ivelt1 <= nzv; ivelt1++) {
					irow = iv[ivelt1];
					if (irow >= firstrow && irow <= lastrow) {
						nnza = nnza + 1;
						if (nnza > nz) {
							printf("Space for matrix elements exceeded in" " makea\n");
							printf("nnza, nzmax = %d, %d\n", nnza, nz);
							printf("iouter = %d\n", iouter);
							exit(1);
						}
						acol[nnza] = jcol;
						arow[nnza] = irow;
						aelt[nnza] = v[ivelt1] * scale;
					}
				}
			}
		}
		size = size * ratio;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... add the identity * rcond to the generated matrix to bound
	 * the smallest eigenvalue from below by rcond
	 * ---------------------------------------------------------------------
	 */
	for (i = firstrow; i <= lastrow; i++) {
		if (i >= firstcol && i <= lastcol) {
			iouter = n + i;
			nnza = nnza + 1;
			if (nnza > nz) {
				printf("Space for matrix elements exceeded in makea\n");
				printf("nnza, nzmax = %d, %d\n", nnza, nz);
				printf("iouter = %d\n", iouter);
				exit(1);
			}
			acol[nnza] = i;
			arow[nnza] = i;
			aelt[nnza] = rcond - shift;
		}
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... make the sparse matrix from list of elements with duplicates
	 * (iv is used as  workspace)
	 * ---------------------------------------------------------------------
	 */
	sparse(a,
			colidx,
			rowstr,
			n,
			arow,
			acol,
			aelt,
			firstrow,
			lastrow,
			v,
			&(iv[0]),
			&(iv[n]),
			nnza);
}

/*
 * ---------------------------------------------------
 * generate a sparse matrix from a list of
 * [col, row, element] tri
 * ---------------------------------------------------
 */
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int arow[],
		int acol[],
		double aelt[],
		int firstrow,
		int lastrow,
		double x[],
		boolean mark[],
		int nzloc[],
		int nnza){
	/*
	* ---------------------------------------------------------------------
	* rows range from firstrow to lastrow
	* the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
	* ---------------------------------------------------------------------
	*/
	int nrows;
	int i, j, jajp1, nza, k, nzrow;
	double xi;

	/*
	 * --------------------------------------------------------------------
	 * how many rows of result
	 * --------------------------------------------------------------------
	 */
	nrows = lastrow - firstrow + 1;

	/*
	 * --------------------------------------------------------------------
	 * ...count the number of triples in each row
	 * --------------------------------------------------------------------
	 */
	#pragma omp parallel for     
	for (j = 1; j <= n; j++) {
		rowstr[j] = 0;
		mark[j] = FALSE;
	}
	rowstr[n+1] = 0;

	for (nza = 1; nza <= nnza; nza++) {
		j = (arow[nza] - firstrow + 1) + 1;
		rowstr[j] = rowstr[j] + 1;
	}
	rowstr[1] = 1;
	for (j = 2; j <= nrows+1; j++) {
		rowstr[j] = rowstr[j] + rowstr[j-1];
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... rowstr(j) now is the location of the first nonzero
	 * of row j of a
	 * ---------------------------------------------------------------------
	 */

	/*
	 * ---------------------------------------------------------------------
	 * ... do a bucket sort of the triples on the row index
	 * ---------------------------------------------------------------------
	 */
	for (nza = 1; nza <= nnza; nza++) {
		j = arow[nza] - firstrow + 1;
		k = rowstr[j];
		a[k] = aelt[nza];
		colidx[k] = acol[nza];
		rowstr[j] = rowstr[j] + 1;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... rowstr(j) now points to the first element of row j+1
	 * ---------------------------------------------------------------------
	 */
	for (j = nrows; j >= 1; j--) {
		rowstr[j+1] = rowstr[j];
	}
	rowstr[1] = 1;

	/*
	 * ---------------------------------------------------------------------
	 * ... generate actual values by summing duplicates
	 * ---------------------------------------------------------------------
	 */
	nza = 0;
	#pragma omp parallel for    
	for (i = 1; i <= n; i++) {
		x[i] = 0.0;
		mark[i] = FALSE;
	}

	jajp1 = rowstr[1];
	for (j = 1; j <= nrows; j++) {
		nzrow = 0;

		/*
		 * ---------------------------------------------------------------------
		 * ...loop over the jth row of a
		 * ---------------------------------------------------------------------
		 */
		for (k = jajp1; k < rowstr[j+1]; k++) {
			i = colidx[k];
			x[i] = x[i] + a[k];
			if ( mark[i] == FALSE && x[i] != 0.0) {
				mark[i] = TRUE;
				nzrow = nzrow + 1;
				nzloc[nzrow] = i;
			}
		}

		/*
		 * ---------------------------------------------------------------------
		 * ... extract the nonzeros of this row
		 * ---------------------------------------------------------------------
		 */
		for (k = 1; k <= nzrow; k++) {
			i = nzloc[k];
			mark[i] = FALSE;
			xi = x[i];
			x[i] = 0.0;
			if (xi != 0.0) {
				nza = nza + 1;
				a[nza] = xi;
				colidx[nza] = i;
			}
		}
		jajp1 = rowstr[j+1];
		rowstr[j+1] = nza + rowstr[1];
	}
}

/*
 * ---------------------------------------------------------------------
 * generate a sparse n-vector (v, iv)
 * having nzv nonzeros
 *
 * mark(i) is set to 1 if position i is nonzero.
 * mark is all zero on entry and is reset to all zero before exit
 * this corrects a performance bug found by John G. Lewis, caused by
 * reinitialization of mark on every one of the n calls to sprnvc
 * ---------------------------------------------------------------------
 */
static void sprnvc(int n,
		int nz,
		double v[],
		int iv[],
		int nzloc[],
		int mark[]){
	int nn1;
	int nzrow, nzv, ii, i;
	double vecelt, vecloc;

	nzv = 0;
	nzrow = 0;
	nn1 = 1;
	do {
		nn1 = 2 * nn1;
	} while (nn1 < n);

	/*
	 * --------------------------------------------------------------------
	 * nn1 is the smallest power of two not less than n
	 * --------------------------------------------------------------------
	 */

	while (nzv < nz) {
		vecelt = randlc(&tran, amult);

		/*
		 * --------------------------------------------------------------------
		 * generate an integer between 1 and n in a portable manner
		 * --------------------------------------------------------------------
		 */
		vecloc = randlc(&tran, amult);
		i = icnvrt(vecloc, nn1) + 1;
		if (i > n) continue;

		/*
		 * --------------------------------------------------------------------
		 * was this integer generated already?
		 * --------------------------------------------------------------------
		 */
		if (mark[i] == 0) {
			mark[i] = 1;
			nzrow = nzrow + 1;
			nzloc[nzrow] = i;
			nzv = nzv + 1;
			v[nzv] = vecelt;
			iv[nzv] = i;
		}
	}

	for (ii = 1; ii <= nzrow; ii++) {
		i = nzloc[ii];
		mark[i] = 0;
	}
}

/*
 * ---------------------------------------------------------------------
 * scale a double precision number x in (0,1) by a power of 2 and chop it
 * ---------------------------------------------------------------------
 */
static int icnvrt(double x,
		int ipwr2){
	return ((int)(ipwr2 * x));
}

/*
 * --------------------------------------------------------------------
 * set ith element of sparse vector (v, iv) with
 * nzv nonzeros to val
 * --------------------------------------------------------------------
 */
static void vecset(int n,
		double v[],
		int iv[],
		int *nzv,
		int i,
		double val){
	int k;
	boolean set;

	set = FALSE;
	for (k = 1; k <= *nzv; k++) {
		if (iv[k] == i) {
			v[k] = val;
			set  = TRUE;
		}
	}
	if (set == FALSE) {
		*nzv = *nzv + 1;
		v[*nzv] = val;
		iv[*nzv] = i;
	}
}

static void distribute(int& beg,
		int& end,
		const int& loop_size,
		const int& beg_offset,
    		const int& less_equal){
	int chunk = loop_size / numtasks;
	beg = workrank * chunk + ((workrank == 0) ? beg_offset : less_equal);
	end = (workrank != numtasks - 1) ? workrank * chunk + chunk : loop_size;
}
