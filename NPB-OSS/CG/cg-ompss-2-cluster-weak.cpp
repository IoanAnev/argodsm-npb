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
 * OmpSs-2@Cluster version:
 *      Ioannis Anevlavis <ioannis.anevlavis@etascale.com>
 */

#include <cassert>
#include <algorithm>
#include "../common/memory.hpp"
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
#define NZ (NA*(NONZER+1)*(NONZER+1))
#define NAZ (NA*(NONZER+1))
#define T_INIT 0
#define T_BENCH 1
#define T_CONJ_GRAD 2
#define T_LAST 3
#define BSIZE_UNIT 512

// undef: fetch tasks disabled
//   def: fetch tasks  enabled
// #define ENABLE_FETCH_TASKS

//   1: nanos6-mpi  optimal
// 512: nanos6-argo optimal
#define SCALARS_PAD 512

/* local array allocations (nanos6)   */
int *colidx;
int *rowstr;
int *iv;
int *arow;
int *acol;
double *aelt;
double *a;
/* global array allocations (nanos6)  */
double *x;
double *z;
double *p;
double *q;
double *r;
/* global scalar allocations (nanos6) */
double *alpha;
double *beta;
double *d;
double *sum;
double *rho;
double *rho0;

/* global variables (def)  */
static int naa;
static int nzz;
static int nazz;
static int SCPAD;
static int BSIZE;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
static boolean timeron;

/* global variables (weak) */
static int nodes;
static int node_id;
static int region_per_node_naa;
static int region_per_node_naa_1;

inline uint32_t flp2(uint32_t x)
{
	   x = x | (x >>  1);
	   x = x | (x >>  2);
	   x = x | (x >>  4);
	   x = x | (x >>  8);
	   x = x | (x >> 16);
	return x - (x >>  1);
}

#define ALIGN_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(size, align) ((size) & ~((align) - 1))

#define unaligned_chunk ((NA+2) / (nodes))
#define aligned_down_chunk ALIGN_DOWN((unaligned_chunk), 512)
#define power_of_two_chunk flp2(unaligned_chunk)

#define aligned_chunk ((nodes*512 <= NA+1) ? (aligned_down_chunk) : (power_of_two_chunk))
#define remainder_chunk ((NA+1) - (nodes)*(aligned_chunk))

/* function prototypes */
static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm,
		int dist_table[]);
static int icnvrt(double x,
		int ipwr2);
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]);
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift);
static void sprnvc(int n,
		int nz,
		int nn1,
		double v[],
		int iv[]);
static void vecset(int n,
		double v[],
		int iv[],
		int* nzv,
		int i,
		double val);
static void task_chunk(int& beg,
		int& end,
		int& chunk,
		const int& size,
		const int& index,
		const int& bsize,
		bool& breaklp);
static void node_chunk(int node_id,
		int& beg,
		int& chunk,
		const int& size,
		int dist_table[]);
/* cg */
int main(int argc, char **argv){
	/*
	 * ---------------------------------------------------------------------
	 * fetch number of nodes
	 * ---------------------------------------------------------------------
	 */
	nodes = nanos6_get_num_cluster_nodes();

	/*
	 * ---------------------------------------------------------------------
	 * global array allocations
	 * ---------------------------------------------------------------------
	 */
	x = dmalloc<double>(NA+2);
	z = dmalloc<double>(NA+2);
	p = dmalloc<double>(NA+2);
	q = dmalloc<double>(NA+2);
	r = dmalloc<double>(NA+2);

	/*
	 * ---------------------------------------------------------------------
	 * global scalar allocations
	 * ---------------------------------------------------------------------
	 */
	double *norm_temp1 = dmalloc<double>((nodes+1)*SCALARS_PAD);
	double *norm_temp2 = dmalloc<double>((nodes+1)*SCALARS_PAD);
	double *rnorm      = dmalloc<double>(SCALARS_PAD);
	double *zeta       = dmalloc<double>(SCALARS_PAD);

	/*
	 * --------------------------------------------------------------------
	 * global scalar allocations for the conj_grad() function
	 * --------------------------------------------------------------------
	 */
	d     = dmalloc<double>((nodes+1)*SCALARS_PAD);
	sum   = dmalloc<double>((nodes+1)*SCALARS_PAD);
	rho   = dmalloc<double>((nodes+1)*SCALARS_PAD);
	alpha = dmalloc<double>(SCALARS_PAD);
	beta  = dmalloc<double>(SCALARS_PAD);
	rho0  = dmalloc<double>(SCALARS_PAD);

	/*
	 * ---------------------------------------------------------------------
	 * "local" array allocations
	 * ---------------------------------------------------------------------
	 */
	colidx = dmalloc<int>(NZ);
	rowstr = dmalloc<int>(NA+1);
	iv     = dmalloc<int>(NA);
	arow   = dmalloc<int>(NA);
	acol   = dmalloc<int>(NAZ);
	aelt   = dmalloc<double>(NAZ);
	a      = dmalloc<double>(NZ);

	/*
	 * --------------------------------------------------------------------
	 * continue with the local allocations
	 * --------------------------------------------------------------------
	 */
	char class_npb;
	double t, tmax, zeta_verify_value;

	SCPAD = SCALARS_PAD;

	if (argc > 1) {
		BSIZE = atoi(argv[1]);
	} else {
		BSIZE = BSIZE_UNIT;
	}

	/*
	 * ---------------------------------------------------------------------
	 * fetch number of nodes and create distribution table
	 * ---------------------------------------------------------------------
	 */
	int dist_table[nodes];
	int min_chunk_size = BSIZE;
	int remainder = remainder_chunk;
	
	dist_table[0] = 0;
	for (int i = 1; i < nodes; ++i) {
		if (remainder >= min_chunk_size) {
			dist_table[i] = dist_table[i-1] + aligned_chunk + min_chunk_size;
			remainder -= min_chunk_size;
		} else
			dist_table[i] = dist_table[i-1] + aligned_chunk;
	}

	char *t_names[T_LAST];

	for(int i=0; i<T_LAST; i++){
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

	firstrow = 0;
	lastrow  = NA-1;
	firstcol = 0;
	lastcol  = NA-1;

	if(NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0){
		class_npb = 'S';
		zeta_verify_value = 8.5971775078648;
	}else if(NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0){
		class_npb = 'W';
		zeta_verify_value = 10.362595087124;
	}else if(NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0){
		class_npb = 'A';
		zeta_verify_value = 17.130235054029;
	}else if(NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0){
		class_npb = 'B';
		zeta_verify_value = 22.712745482631;
	}else if(NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0){
		class_npb = 'C';
		zeta_verify_value = 28.973605592845;
	}else if(NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500.0){
		class_npb = 'D';
		zeta_verify_value = 52.514532105794;
	}else if(NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500.0){
		class_npb = 'E';
		zeta_verify_value = 77.522164599383;
	}else{
		class_npb = 'U';
	}

	printf("\n\n NAS Parallel Benchmarks 4.1 Parallel C++ version with OpenMP - CG Benchmark\n\n");
	printf(" Size: %17d\n", NA);
	printf(" Iterations: %11d\n", NITER);
	printf(" Task granularity: %5d\n", BSIZE);

	naa  = NA;
	nzz  = NZ;
	nazz = NAZ;

	/* initialize random number generator */
	tran    = 314159265.0;
	amult   = 1220703125.0;
	randlc( &tran, amult );

	#pragma oss task inout(colidx[0;nzz  ],	\
			       rowstr[0;naa+1],	\
			       iv    [0;naa  ],	\
			       arow  [0;naa  ],	\
			       acol  [0;nazz ],	\
			       aelt  [0;nazz ],	\
			       a     [0;nzz  ])	\
			 node(nanos6_cluster_no_offload)
	makea(naa, 
			nzz, 
			a, 
			colidx, 
			rowstr, 
			firstrow, 
			lastrow, 
			firstcol, 
			lastcol, 
			arow, 
			(int(*)[NONZER+1])(void*)acol, 
			(double(*)[NONZER+1])(void*)aelt,
			iv);
	#pragma oss taskwait /* solves finalization issue with nanos6-argodsm */

	/*
	 * ---------------------------------------------------------------------
	 * note: as a result of the above call to makea:
	 * values of j used in indexing rowstr go from 0 --> lastrow-firstrow
	 * values of colidx which are col indexes go from firstcol --> lastcol
	 * so:
	 * shift the col index vals from actual (firstcol --> lastcol) 
	 * to local, i.e., (0 --> lastcol-firstcol)
	 * ---------------------------------------------------------------------
	 */
	int gg;
	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa_1, NA+1, dist_table);

		/* Spawn a task for the whole chunk and bind to `node_id` */
		#pragma oss task weakout(x[gg;region_per_node_naa_1])		\
				 firstprivate(gg, region_per_node_naa_1, BSIZE)	\
				 node(node_id)
		{
			/* set starting vector to (1, 1, .... 1) */
			bool innerlp = 0;
			for (int j = gg; j < gg+region_per_node_naa_1; j += BSIZE){
				if (innerlp) break;

				int beg, end, chunk;
				task_chunk(beg, end, chunk, gg+region_per_node_naa_1, j, BSIZE, innerlp);

				#pragma oss task out(x[beg:end-1])		\
						 firstprivate(beg, end)		\
						 node(nanos6_cluster_no_offload)
				for (int k = beg; k < end; k++){
					x[k] = 1.0;
				}
			}
		}
	}

	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

		#pragma oss task weakout(q[gg;region_per_node_naa],		\
				         z[gg;region_per_node_naa],		\
				         r[gg;region_per_node_naa],		\
				         p[gg;region_per_node_naa])		\
			         firstprivate(gg, region_per_node_naa, BSIZE)	\
			         node(node_id)
		{
			bool innerlp = 0;
			for (int j = gg; j < gg+region_per_node_naa; j += BSIZE){
				if (innerlp) break;

				int beg, end, chunk;
				task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);

				#pragma oss task out(q[beg:end-1],		\
						     z[beg:end-1],		\
						     r[beg:end-1], 		\
						     p[beg:end-1])		\
						 firstprivate(beg, end)		\
						 node(nanos6_cluster_no_offload)
				for (int k = beg; k < end; k++){
					q[k] = 0.0;
					z[k] = 0.0;
					r[k] = 0.0;
					p[k] = 0.0;
				}
			}
		}
	}

	#pragma oss task out(*zeta)	\
			 node(nanos6_cluster_no_offload)
		*zeta = 0.0;

	/*
	 * -------------------------------------------------------------------
	 * ---->
	 * do one iteration untimed to init all code and data page tables
	 * ----> (then reinit, start timing, to niter its)
	 * -------------------------------------------------------------------*/
	for(int it = 1; it <= 1; it++){
		/* the call to the conjugate gradient routine */
		conj_grad(colidx, rowstr, x, z, a, p, q, r, rnorm, dist_table);

		/*
		 * --------------------------------------------------------------------
		 * zeta = shift + 1/(x.z)
		 * so, first: (x.z)
		 * also, find norm of z
		 * so, first: (z.z)
		 * --------------------------------------------------------------------
		 */
		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(x[gg;region_per_node_naa],				\
						z[gg;region_per_node_naa])				\
					 weakinout(norm_temp1[node_id*SCPAD],				\
						   norm_temp2[node_id*SCPAD])				\
					 firstprivate(gg, region_per_node_naa, BSIZE, node_id, SCPAD)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(x[gg;region_per_node_naa],	\
						    z[gg;region_per_node_naa])	\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				#pragma oss task in(x[gg;region_per_node_naa],					\
						    z[gg;region_per_node_naa])					\
						    out(norm_temp1[node_id*SCPAD],				\
						    norm_temp2[node_id*SCPAD])					\
						    firstprivate(gg, region_per_node_naa, node_id, SCPAD)	\
						    node(nanos6_cluster_no_offload)
				{
					double temp1 = 0.0;
					double temp2 = 0.0;
					for (int k = gg; k < gg+region_per_node_naa; k++){
						temp1 += x[k] * z[k];
						temp2 += z[k] * z[k];
					}
					norm_temp1[node_id*SCPAD] = temp1;
					norm_temp2[node_id*SCPAD] = temp2;
				}
			}
		}

		#pragma oss task in(norm_temp1[0;nodes*SCPAD],		\
				    norm_temp2[0;nodes*SCPAD])		\
				 out(norm_temp2[nodes*SCPAD])	\
				 out(*zeta)				\
				 firstprivate(nodes, SCPAD)		\
				 node(nanos6_cluster_no_offload)
		{
			double temp1 = 0.0;
			double temp2 = 0.0;
			for (int node_id = 0; node_id < nodes; ++node_id) {
				temp1 += norm_temp1[node_id*SCPAD];
				temp2 += norm_temp2[node_id*SCPAD];
			}
			norm_temp2[nodes*SCPAD] = 1.0 / sqrt(temp2);
			*zeta = SHIFT + 1.0 / temp1;
		}

		/* normalize z to obtain x */
		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(z[gg;region_per_node_naa],				\
						norm_temp2[nodes*SCPAD])				\
					 weakout(x[gg;region_per_node_naa])				\
					 firstprivate(gg, region_per_node_naa, BSIZE, nodes, SCPAD)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(z[gg;region_per_node_naa])	\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				bool innerlp = 0;
				for (int j = gg; j < gg+region_per_node_naa; j += BSIZE){
					if (innerlp) break;

					int beg, end, chunk;
					task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);

					#pragma oss task in(z[beg:end-1],			\
							    norm_temp2[nodes*SCPAD])		\
							 out(x[beg:end-1]) 			\
							 firstprivate(beg, end, nodes, SCPAD)	\
							 node(nanos6_cluster_no_offload)
					for (int k = beg; k < end; k++){
						x[k] = norm_temp2[nodes*SCPAD] * z[k];
					}
				}
			}
		}
	} /* end of do one iteration untimed */

	/* set starting vector to (1, 1, .... 1) */
	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa_1, NA+1, dist_table);

		/* Spawn a task for the whole chunk and bind to `node_id` */
		#pragma oss task weakout(x[gg;region_per_node_naa_1])		\
				 firstprivate(gg, region_per_node_naa_1, BSIZE)	\
				 node(node_id)
		{
			/* set starting vector to (1, 1, .... 1) */
			bool innerlp = 0;
			for (int j = gg; j < gg+region_per_node_naa_1; j += BSIZE){
				if (innerlp) break;

				int beg, end, chunk;
				task_chunk(beg, end, chunk, gg+region_per_node_naa_1, j, BSIZE, innerlp);

				#pragma oss task out(x[beg:end-1])		\
						 firstprivate(beg, end)		\
						 node(nanos6_cluster_no_offload)
				for (int k = beg; k < end; k++){
					x[k] = 1.0;
				}
			}
		}
	}

	#pragma oss task out(*zeta)	\
			 node(nanos6_cluster_no_offload)
		*zeta = 0.0;
	
	#pragma oss taskwait
	timer_stop(T_INIT);
	printf(" Initialization time = %15.3f seconds\n", timer_read(T_INIT));
	
	timer_start(T_BENCH);

	/*
	 * --------------------------------------------------------------------
	 * ---->
	 * main iteration for inverse power method
	 * ---->
	 * --------------------------------------------------------------------
	 */
	for(int it = 1; it <= NITER; it++){
		/* the call to the conjugate gradient routine */
		if(timeron){timer_start(T_CONJ_GRAD);}
		conj_grad(colidx, rowstr, x, z, a, p, q, r, rnorm, dist_table);
		if(timeron){timer_stop(T_CONJ_GRAD);}

		/*
		 * --------------------------------------------------------------------
		 * zeta = shift + 1/(x.z)
		 * so, first: (x.z)
		 * also, find norm of z
		 * so, first: (z.z)
		 * --------------------------------------------------------------------
		 */
		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(x[gg;region_per_node_naa],				\
						z[gg;region_per_node_naa])				\
					 weakinout(norm_temp1[node_id*SCPAD],				\
						   norm_temp2[node_id*SCPAD])				\
					 firstprivate(gg, region_per_node_naa, BSIZE, node_id, SCPAD)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(x[gg;region_per_node_naa],	\
						    z[gg;region_per_node_naa])	\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				#pragma oss task in(x[gg;region_per_node_naa],					\
						    z[gg;region_per_node_naa])					\
						    out(norm_temp1[node_id*SCPAD],				\
						    norm_temp2[node_id*SCPAD])					\
						    firstprivate(gg, region_per_node_naa, node_id, SCPAD)	\
						    node(nanos6_cluster_no_offload)
				{
					double temp1 = 0.0;
					double temp2 = 0.0;
					for (int k = gg; k < gg+region_per_node_naa; k++){
						temp1 += x[k] * z[k];
						temp2 += z[k] * z[k];
					}
					norm_temp1[node_id*SCPAD] = temp1;
					norm_temp2[node_id*SCPAD] = temp2;
				}
			}
		}

		#pragma oss task in(norm_temp1[0;nodes*SCPAD],		\
				    norm_temp2[0;nodes*SCPAD])		\
				 out(norm_temp2[nodes*SCPAD])		\
				 out(*zeta)				\
				 firstprivate(nodes, SCPAD)		\
				 node(nanos6_cluster_no_offload)
		{
			double temp1 = 0.0;
			double temp2 = 0.0;
			for (int node_id = 0; node_id < nodes; ++node_id) {
				temp1 += norm_temp1[node_id*SCPAD];
				temp2 += norm_temp2[node_id*SCPAD];
			}
			norm_temp2[nodes*SCPAD] = 1.0 / sqrt(temp2);
			*zeta = SHIFT + 1.0 / temp1;
		}

		if(it==1){printf("\n   iteration           ||r||                 zeta\n");}
		#pragma oss task in(*rnorm, *zeta)	\
				 firstprivate(it)	\
				 node(nanos6_cluster_no_offload)
		printf("    %5d       %20.14e%20.13e\n", it, *rnorm, *zeta);

		/* normalize z to obtain x */
		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(z[gg;region_per_node_naa],				\
						norm_temp2[nodes*SCPAD])				\
					 weakout(x[gg;region_per_node_naa])				\
					 firstprivate(gg, region_per_node_naa, BSIZE, nodes, SCPAD)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(z[gg;region_per_node_naa])	\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				bool innerlp = 0;
				for (int j = gg; j < gg+region_per_node_naa; j += BSIZE){
					if (innerlp) break;

					int beg, end, chunk;
					task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);

					#pragma oss task in(z[beg:end-1],			\
							    norm_temp2[nodes*SCPAD])		\
							 out(x[beg:end-1]) 			\
							 firstprivate(beg, end, nodes, SCPAD)	\
							 node(nanos6_cluster_no_offload)
					for (int k = beg; k < end; k++){
						x[k] = norm_temp2[nodes*SCPAD] * z[k];
					}
				}
			}
		}
	} /* end of main iter inv pow meth */
	
	#pragma oss taskwait
	timer_stop(T_BENCH);

	/*
	 * --------------------------------------------------------------------
	 * end of timed section
	 * --------------------------------------------------------------------
	 */
	t = timer_read(T_BENCH);
	printf(" Benchmark completed\n");

	#pragma oss task in(*zeta)					\
			 firstprivate(class_npb, zeta_verify_value)	\
			 node(nanos6_cluster_no_offload) // (opt.)
	{
		double mflops;
		boolean verified;
		double epsilon = 1.0e-10;
		if(class_npb != 'U'){
			double err = fabs(*zeta - zeta_verify_value) / zeta_verify_value;
			if(err <= epsilon){
				verified = TRUE;
				printf(" VERIFICATION SUCCESSFUL\n");
				printf(" Zeta is    %20.13e\n",*zeta);
				printf(" Error is   %20.13e\n", err);
			}else{
				verified = FALSE;
				printf(" VERIFICATION FAILED\n");
				printf(" Zeta                %20.13e\n", *zeta);
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
	}
	#pragma oss taskwait

	/*
	 * ---------------------------------------------------------------------
	 * more timers
	 * ---------------------------------------------------------------------
	 */
	if(timeron){
		tmax = timer_read(T_BENCH);
		if(tmax == 0.0){tmax = 1.0;}
		printf("  SECTION   Time (secs)\n");
		for(int i = 0; i < T_LAST; i++){
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

	/*
	 * ---------------------------------------------------------------------
	 * "local" array deallocations
	 * ---------------------------------------------------------------------
	 */
	dfree<int>(colidx,  NZ);
	dfree<int>(rowstr,  NA+1);
	dfree<int>(iv,      NA);
	dfree<int>(arow,    NA);
	dfree<int>(acol,    NAZ);
	dfree<double>(aelt, NAZ);
	dfree<double>(a,    NZ);
	
	/*
	 * ---------------------------------------------------------------------
	 * global array deallocations
	 * ---------------------------------------------------------------------
	 */
	dfree<double>(x, NA+2);
	dfree<double>(z, NA+2);
	dfree<double>(p, NA+2);
	dfree<double>(q, NA+2);
	dfree<double>(r, NA+2);

	/*
	 * ---------------------------------------------------------------------
	 * global scalar deallocations
	 * ---------------------------------------------------------------------
	 */
	dfree<double>(norm_temp1, (nodes+1)*SCALARS_PAD);
	dfree<double>(norm_temp2, (nodes+1)*SCALARS_PAD);
	dfree<double>(zeta,       1);
	dfree<double>(rnorm,      1);

	/*
	 * --------------------------------------------------------------------
	 * global scalar deallocations for the conj_grad() function
	 * --------------------------------------------------------------------
	 */
	dfree<double>(d,     (nodes+1)*SCALARS_PAD);
	dfree<double>(sum,   (nodes+1)*SCALARS_PAD);
	dfree<double>(rho,   (nodes+1)*SCALARS_PAD);
	dfree<double>(rho0,  1);
	dfree<double>(alpha, 1);
	dfree<double>(beta,  1);

	return 0;
}

/*
 * ---------------------------------------------------------------------
 * floating point arrays here are named as in NPB1 spec discussion of 
 * CG algorithm
 * ---------------------------------------------------------------------
 */
static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm,
		int dist_table[]){
	int gg;
	int cgit, cgitmax = 25;

	/* initialize the CG algorithm */
	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa_1, NA+1, dist_table);

		#pragma oss task weakin(x[gg;region_per_node_naa_1])		\
				 weakout(r[gg;region_per_node_naa_1],		\
					 q[gg;region_per_node_naa_1],		\
				 	 z[gg;region_per_node_naa_1],		\
					 p[gg;region_per_node_naa_1])		\
				 firstprivate(gg, region_per_node_naa_1, BSIZE)	\
				 node(node_id)
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task in(x[gg;region_per_node_naa_1])	\
					 node(nanos6_cluster_no_offload)
			{
				// fetch all data in one go
			}
#endif

			bool innerlp = 0;
			for(int j = gg; j < gg+region_per_node_naa_1; j += BSIZE){
				if (innerlp) break;

				int beg, end, chunk;
				task_chunk(beg, end, chunk, gg+region_per_node_naa_1, j, BSIZE, innerlp);

				#pragma oss task in(x[beg:end-1])		\
						 out(r[beg:end-1],		\
						     q[beg:end-1],		\
						     z[beg:end-1],		\
						     p[beg:end-1])		\
						 firstprivate(beg, end)		\
						 node(nanos6_cluster_no_offload)
				for (int k = beg; k < end; k++){
					q[k] = 0.0;
					z[k] = 0.0;
					r[k] = x[k];
					p[k] = x[k];
				}
			}
		}
	}
 
	/*
	 * --------------------------------------------------------------------
	 * rho = r.r
	 * now, obtain the norm of r: First, sum squares of r elements locally...
	 * --------------------------------------------------------------------
	 */
	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

		#pragma oss task weakin(r[gg;region_per_node_naa])				\
				 weakout(rho[node_id*SCPAD])					\
				 firstprivate(gg, region_per_node_naa, BSIZE, node_id, SCPAD)	\
				 node(node_id)
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task in(r[gg;region_per_node_naa])		\
					 node(nanos6_cluster_no_offload)
			{
				// fetch all data in one go
			}
#endif

			#pragma oss task in(r[gg;region_per_node_naa])				\
					 out(rho[node_id*SCPAD])				\
					 firstprivate(gg, region_per_node_naa, node_id, SCPAD)	\
					 node(nanos6_cluster_no_offload)
			{
				double temp = 0.0;
				for (int k = gg; k < gg+region_per_node_naa; k++){
					temp += r[k]*r[k];
				}
				rho[node_id*SCPAD] = temp;
			}
		}
	}

	/* the conj grad iteration loop */
	for(cgit = 1; cgit <= cgitmax; cgit++){
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

		#pragma oss task in(rho[0;nodes*SCPAD])		\
				 out(rho[nodes*SCPAD],		\
				     *rho0)			\
				 firstprivate(nodes, SCPAD)	\
				 node(nanos6_cluster_no_offload)
		{
			double temp = 0.0;
			for (int node_id = 0; node_id < nodes; ++node_id) {
				temp += rho[node_id*SCPAD];
			}
			rho[nodes*SCPAD] = temp;
			/*
			 * --------------------------------------------------------------------
			 * save a temporary of rho
			 * --------------------------------------------------------------------
			 */
			*rho0 = temp;
		}

		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);
			int beg_row_out{rowstr[gg]}, end_row_out{rowstr[gg+region_per_node_naa]};
			
			/* colidx elements are not stored in ascending order as in rowstr */
			int beg_col_out = *std::min_element(colidx+beg_row_out, colidx+end_row_out);
			int end_col_out = *std::max_element(colidx+beg_row_out, colidx+end_row_out);

			#pragma oss task weakin(rowstr[gg;region_per_node_naa+1],	\
						colidx[beg_row_out:end_row_out-1],	\
						a[beg_row_out:end_row_out-1],		\
						p[beg_col_out:end_col_out])		\
					 weakout(q[gg;region_per_node_naa])		\
					 firstprivate(gg, region_per_node_naa, BSIZE,	\
						      naa, beg_row_out, end_row_out,	\
						      nzz, beg_col_out, end_col_out)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(rowstr[gg;region_per_node_naa+1],	\
						    colidx[beg_row_out:end_row_out-1],	\
						    a[beg_row_out:end_row_out-1],	\
						    p[beg_col_out:end_col_out])		\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				bool innerlp = 0;
				for(int j = gg; j < gg+region_per_node_naa; j += BSIZE){
					if (innerlp) break;
					
					int beg, end, chunk;
					task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);
					int beg_row{rowstr[beg]}, end_row{rowstr[end]};
					
					/* colidx elements are not stored in ascending order as in rowstr */
					int beg_col = *std::min_element(colidx+beg_row, colidx+end_row);
					int end_col = *std::max_element(colidx+beg_row, colidx+end_row);
					
					#pragma oss task in(rowstr[beg:end],		\
							    colidx[beg_row:end_row-1],	\
							    a[beg_row:end_row-1],	\
							    p[beg_col:end_col])		\
							 out(q[beg:end-1]) 		\
							 firstprivate(beg, end)		\
							 node(nanos6_cluster_no_offload)
					for (int j = beg; j < end; j++){
						double suml = 0.0;
						for(int k = rowstr[j]; k < rowstr[j+1]; k++){
							suml += a[k]*p[colidx[k]];
						}
						q[j] = suml;
					}
				}
			}
		}

		/*
		 * --------------------------------------------------------------------
		 * obtain p.q
		 * --------------------------------------------------------------------
		 */
		for (int node_id = 0; node_id < nodes; ++node_id) {			
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(p[gg;region_per_node_naa],				\
						q[gg;region_per_node_naa])				\
					 weakout(d[node_id*SCPAD])					\
					 firstprivate(gg, region_per_node_naa, BSIZE, node_id, SCPAD)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(p[gg;region_per_node_naa],	\
						    q[gg;region_per_node_naa])	\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				#pragma oss task in(p[gg;region_per_node_naa],				\
						    q[gg;region_per_node_naa])				\
						 out(d[node_id*SCPAD])					\
						 firstprivate(gg, region_per_node_naa, node_id, SCPAD)	\
						 node(nanos6_cluster_no_offload)
				{
					double temp = 0.0;
					for (int k = gg; k < gg+region_per_node_naa; k++){
						temp += p[k]*q[k];
					}
					d[node_id*SCPAD] = temp;
				}
			}
		}

		/*
		 * --------------------------------------------------------------------
		 * obtain alpha = rho / (p.q)
		 * -------------------------------------------------------------------
		 */
		#pragma oss task in(d[0;nodes*SCPAD])		\
				 out(d[nodes*SCPAD])		\
				 firstprivate(nodes, SCPAD)	\
				 node(nanos6_cluster_no_offload)
		{
			double temp = 0.0;
			for (int node_id = 0; node_id < nodes; ++node_id) {
				temp += d[node_id*SCPAD];
			}
			d[nodes*SCPAD] = temp;
		}

		// #pragma oss taskwait on(*rho, *d)
		#pragma oss task in(*rho0, d[nodes*SCPAD])		\
				 out(*alpha)				\
				 firstprivate(nodes, SCPAD)		\
				 node(nanos6_cluster_no_offload)
			*alpha = *rho0 / d[nodes*SCPAD];
			
		/*
		 * ---------------------------------------------------------------------
		 * obtain z = z + alpha*p
		 * and    r = r - alpha*q
		 * ---------------------------------------------------------------------
		 */
		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(p[gg;region_per_node_naa],				\
						q[gg;region_per_node_naa], *alpha)			\
					 weakinout(z[gg;region_per_node_naa],				\
					 	   r[gg;region_per_node_naa])				\
					 weakout(rho[node_id*SCPAD])					\
					 firstprivate(gg, region_per_node_naa, BSIZE, node_id, SCPAD)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(p[gg;region_per_node_naa],	\
						    q[gg;region_per_node_naa],	\
						    z[gg;region_per_node_naa],	\
						    r[gg;region_per_node_naa])	\
						 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				bool innerlp = 0;
				for(int j = gg; j < gg+region_per_node_naa; j += BSIZE){
					if (innerlp) break;
					
					int beg, end, chunk;
					task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);

					#pragma oss task in(p[beg:end-1],		\
							    q[beg:end-1], *alpha) 	\
							 inout(z[beg:end-1],		\
							       r[beg:end-1])		\
							 firstprivate(beg, end)		\
							 node(nanos6_cluster_no_offload)
					for (int k = beg; k < end; k++){
						z[k] += *alpha*p[k];
						r[k] -= *alpha*q[k];
					}
				}

				#pragma oss task in(r[gg;region_per_node_naa])				\
						 out(rho[node_id*SCPAD])				\
						 firstprivate(gg, region_per_node_naa, node_id, SCPAD)	\
						 node(nanos6_cluster_no_offload)
				{
					double temp = 0.0;
					for (int k = gg; k < gg+region_per_node_naa; k++){
						/*
						* ---------------------------------------------------------------------
						* rho = r.r
						* now, obtain the norm of r: first, sum squares of r elements locally...
						* ---------------------------------------------------------------------
						*/
						temp += r[k]*r[k];
					}
					rho[node_id*SCPAD] = temp;
				}
			}
		}

		/*
		 * ---------------------------------------------------------------------
		 * obtain beta
		 * ---------------------------------------------------------------------
		 */
		#pragma oss task in(rho[0;nodes*SCPAD])		\
				 out(rho[nodes*SCPAD])		\
				 firstprivate(nodes, SCPAD)	\
				 node(nanos6_cluster_no_offload)
		{
			double temp = 0.0;
			for (int node_id = 0; node_id < nodes; ++node_id) {
				temp += rho[node_id*SCPAD];
			}
			rho[nodes*SCPAD] = temp;
		}

		// #pragma oss taskwait on(*rho, *rho0)
		#pragma oss task in(rho[nodes*SCPAD], *rho0)		\
				 out(*beta)				\
				 firstprivate(nodes, SCPAD)		\
				 node(nanos6_cluster_no_offload)
			*beta = rho[nodes*SCPAD] / *rho0;

		/*
		 * ---------------------------------------------------------------------
		 * p = r + beta*p
		 * ---------------------------------------------------------------------
		 */
		for (int node_id = 0; node_id < nodes; ++node_id) {
			/* Calculate row region for each node and node_id */
			node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

			#pragma oss task weakin(r[gg;region_per_node_naa], *beta)	\
					 weakinout(p[gg;region_per_node_naa])		\
					 firstprivate(gg, region_per_node_naa, BSIZE)	\
					 node(node_id)
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(r[gg;region_per_node_naa],	\
						    p[gg;region_per_node_naa])	\
					    	 node(nanos6_cluster_no_offload)
				{
					// fetch all data in one go
				}
#endif

				bool innerlp = 0;
				for(int j = gg; j < gg+region_per_node_naa; j += BSIZE){
					if (innerlp) break;
					
					int beg, end, chunk;
					task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);

					#pragma oss task in(r[beg:end-1], *beta)	\
							 inout(p[beg:end-1])		\
							 firstprivate(beg, end)		\
							 node(nanos6_cluster_no_offload)
					for (int k = beg; k < end; k++){
						p[k] = r[k] + *beta*p[k];
					}
				}
			}
		}
	} /* end of do cgit=1, cgitmax */

	/*
	 * ---------------------------------------------------------------------
	 * compute residual norm explicitly: ||r|| = ||x - A.z||
	 * first, form A.z
	 * the partition submatrix-vector multiply
	 * ---------------------------------------------------------------------
	 */
	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);
		int beg_row_out{rowstr[gg]}, end_row_out{rowstr[gg+region_per_node_naa]};
		
		/* colidx elements are not stored in ascending order as in rowstr */
		int beg_col_out = *std::min_element(colidx+beg_row_out, colidx+end_row_out);
		int end_col_out = *std::max_element(colidx+beg_row_out, colidx+end_row_out);

		#pragma oss task weakin(rowstr[gg;region_per_node_naa+1],	\
					colidx[beg_row_out:end_row_out-1],	\
					a[beg_row_out:end_row_out-1],		\
					z[beg_col_out:end_col_out])		\
				 weakout(r[gg;region_per_node_naa])		\
				 firstprivate(gg, region_per_node_naa, BSIZE,	\
					      naa, beg_row_out, end_row_out,	\
					      nzz, beg_col_out, end_col_out)	\
				 node(node_id)
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task in(rowstr[gg;region_per_node_naa+1],	\
					    colidx[beg_row_out:end_row_out-1],	\
					    a[beg_row_out:end_row_out-1],	\
					    z[beg_col_out:end_col_out])		\
					 node(nanos6_cluster_no_offload)
			{
				// fetch all data in one go
			}
#endif

			bool innerlp = 0;
			for(int j = gg; j < gg+region_per_node_naa; j += BSIZE){
				if (innerlp) break;
				
				int beg, end, chunk;
				task_chunk(beg, end, chunk, gg+region_per_node_naa, j, BSIZE, innerlp);
				int beg_row{rowstr[beg]}, end_row{rowstr[end]};
				
				// colidx elements are not stored in ascending order as in rowstr
				int beg_col = *std::min_element(colidx+beg_row, colidx+end_row);
				int end_col = *std::max_element(colidx+beg_row, colidx+end_row);

				#pragma oss task in(rowstr[beg:end],		\
						    colidx[beg_row:end_row-1],	\
						    a[beg_row:end_row-1],	\
						    z[beg_col:end_col])		\
						 out(r[beg:end-1]) 		\
						 firstprivate(beg, end)		\
						 node(nanos6_cluster_no_offload)
				for (int j = beg; j < end; j++){
					double suml = 0.0;
					for(int k = rowstr[j]; k < rowstr[j+1]; k++){
						suml += a[k]*z[colidx[k]];
					}
					r[j] = suml;
				}
			}
		}
	}

	/*
	 * ---------------------------------------------------------------------
	 * at this point, r contains A.z
	 * ---------------------------------------------------------------------
	 */
	for (int node_id = 0; node_id < nodes; ++node_id) {
		/* Calculate row region for each node and node_id */
		node_chunk(node_id, gg, region_per_node_naa, NA, dist_table);

		#pragma oss task weakin(x[gg;region_per_node_naa],				\
					r[gg;region_per_node_naa])				\
				 weakout(sum[node_id*SCPAD])					\
				 firstprivate(gg, region_per_node_naa, BSIZE, node_id, SCPAD)	\
				 node(node_id)
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task in(x[gg;region_per_node_naa],	\
					    r[gg;region_per_node_naa])	\
					 node(nanos6_cluster_no_offload)
			{
				// fetch all data in one go
			}
#endif

			#pragma oss task in(x[gg;region_per_node_naa],				\
					    r[gg;region_per_node_naa])				\
					 out(sum[node_id*SCPAD])				\
					 firstprivate(gg, region_per_node_naa, node_id, SCPAD)	\
					 node(nanos6_cluster_no_offload)
			{
				double temp = 0.0;
				for (int k = gg; k < gg+region_per_node_naa; k++){
					double suml = x[k] - r[k];
					temp += suml*suml;
				}
				sum[node_id*SCPAD] = temp;
			}
		}
	}

	#pragma oss task in(sum[0;nodes*SCPAD])		\
			 out(sum[nodes*SCPAD])		\
			 firstprivate(nodes, SCPAD)	\
			 node(nanos6_cluster_no_offload)
	{
		double temp = 0.0;
		for (int node_id = 0; node_id < nodes; ++node_id) {
			temp += sum[node_id*SCPAD];
		}
		sum[nodes*SCPAD] = temp;
	}

	// #pragma oss taskwait on(*sum)
	#pragma oss task in(sum[nodes*SCPAD])		\
			 out(*rnorm)			\
			 firstprivate(nodes, SCPAD)	\
			 node(nanos6_cluster_no_offload)
		*rnorm = sqrt(sum[nodes*SCPAD]);
}

/*
 * ---------------------------------------------------------------------
 * scale a double precision number x in (0,1) by a power of 2 and chop it
 * ---------------------------------------------------------------------
 */
static int icnvrt(double x, int ipwr2){
	return (int)(ipwr2 * x);
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
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]){
	int iouter, ivelt, nzv, nn1;
	int ivc[NONZER+1];
	double vc[NONZER+1];

	/*
	 * --------------------------------------------------------------------
	 * nonzer is approximately  (int(sqrt(nnza /n)));
	 * --------------------------------------------------------------------
	 * nn1 is the smallest power of two not less than n
	 * --------------------------------------------------------------------
	 */
	nn1 = 1;
	do{
		nn1 = 2 * nn1;
	}while(nn1 < n);

	/*
	 * -------------------------------------------------------------------
	 * generate nonzero positions and save for the use in sparse
	 * -------------------------------------------------------------------
	 */
	for(iouter = 0; iouter < n; iouter++){
		nzv = NONZER;
		sprnvc(n, nzv, nn1, vc, ivc);
		vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
		arow[iouter] = nzv;
		for(ivelt = 0; ivelt < nzv; ivelt++){
			acol[iouter][ivelt] = ivc[ivelt] - 1;
			aelt[iouter][ivelt] = vc[ivelt];
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
			nz,
			NONZER,
			arow,
			acol,
			aelt,
			firstrow,
			lastrow,
			iv,
			RCOND,
			SHIFT);
}

/*
 * ---------------------------------------------------------------------
 * rows range from firstrow to lastrow
 * the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
 * ---------------------------------------------------------------------
 */
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift){	
	int nrows;

	/*
	 * ---------------------------------------------------
	 * generate a sparse matrix from a list of
	 * [col, row, element] tri
	 * ---------------------------------------------------
	 */
	int i, j, j1, j2, nza, k, kk, nzrow, jcol;
	double size, scale, ratio, va;
	boolean goto_40;

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
	for(j = 0; j < nrows+1; j++){
		rowstr[j] = 0;
	}
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza] + 1;
			rowstr[j] = rowstr[j] + arow[i];
		}
	}
	rowstr[0] = 0;
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] + rowstr[j-1];
	}
	nza = rowstr[nrows] - 1;

	/*
	 * ---------------------------------------------------------------------
	 * ... rowstr(j) now is the location of the first nonzero
	 * of row j of a
	 * ---------------------------------------------------------------------
	 */
	if(nza > nz){
		printf("Space for matrix elements exceeded in sparse\n");
		printf("nza, nzmax = %d, %d\n", nza, nz);
		exit(EXIT_FAILURE);
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... preload data pages
	 * ---------------------------------------------------------------------
	 */
	for(j = 0; j < nrows; j++){
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			a[k] = 0.0;
			colidx[k] = -1;
		}
		nzloc[j] = 0;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... generate actual values by summing duplicates
	 * ---------------------------------------------------------------------
	 */
	size = 1.0;
	ratio = pow(rcond, (1.0 / (double)(n)));
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza];

			scale = size * aelt[i][nza];
			for(nzrow = 0; nzrow < arow[i]; nzrow++){
				jcol = acol[i][nzrow];
				va = aelt[i][nzrow] * scale;

				/*
				 * --------------------------------------------------------------------
				 * ... add the identity * rcond to the generated matrix to bound
				 * the smallest eigenvalue from below by rcond
				 * --------------------------------------------------------------------
				 */
				if(jcol == j && j == i){
					va = va + rcond - shift;
				}

				goto_40 = FALSE;
				for(k = rowstr[j]; k < rowstr[j+1]; k++){
					if(colidx[k] > jcol){
						/*
						 * ----------------------------------------------------------------
						 * ... insert colidx here orderly
						 * ----------------------------------------------------------------
						 */
						for(kk = rowstr[j+1]-2; kk >= k; kk--){
							if(colidx[kk] > -1){
								a[kk+1] = a[kk];
								colidx[kk+1] = colidx[kk];
							}
						}
						colidx[k] = jcol;
						a[k]  = 0.0;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == -1){
						colidx[k] = jcol;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == jcol){
						/*
						 * --------------------------------------------------------------
						 * ... mark the duplicated entry
						 * -------------------------------------------------------------
						 */
						nzloc[j] = nzloc[j] + 1;
						goto_40 = TRUE;
						break;
					}
				}
				if(goto_40 == FALSE){
					printf("internal error in sparse: i=%d\n", i);
					exit(EXIT_FAILURE);
				}
				a[k] = a[k] + va;
			}
		}
		size = size * ratio;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... remove empty entries and generate final results
	 * ---------------------------------------------------------------------
	 */
	for(j = 1; j < nrows; j++){
		nzloc[j] = nzloc[j] + nzloc[j-1];
	}

	for(j = 0; j < nrows; j++){
		if(j > 0){
			j1 = rowstr[j] - nzloc[j-1];
		}else{
			j1 = 0;
		}
		j2 = rowstr[j+1] - nzloc[j];
		nza = rowstr[j];
		for(k = j1; k < j2; k++){
			a[k] = a[nza];
			colidx[k] = colidx[nza];
			nza = nza + 1;
		}
	}
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] - nzloc[j-1];
	}
	nza = rowstr[nrows] - 1;
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
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]){
	int nzv, ii, i;
	double vecelt, vecloc;

	nzv = 0;

	while(nzv < nz){
		vecelt = randlc(&tran, amult);

		/*
		 * --------------------------------------------------------------------
		 * generate an integer between 1 and n in a portable manner
		 * --------------------------------------------------------------------
		 */
		vecloc = randlc(&tran, amult);
		i = icnvrt(vecloc, nn1) + 1;
		if(i>n){continue;}

		/*
		 * --------------------------------------------------------------------
		 * was this integer generated already?
		 * --------------------------------------------------------------------
		 */
		boolean was_gen = FALSE;
		for(ii = 0; ii < nzv; ii++){
			if(iv[ii] == i){
				was_gen = TRUE;
				break;
			}
		}
		if(was_gen){continue;}
		v[nzv] = vecelt;
		iv[nzv] = i;
		nzv = nzv + 1;
	}
}

/*
 * --------------------------------------------------------------------
 * set ith element of sparse vector (v, iv) with
 * nzv nonzeros to val
 * --------------------------------------------------------------------
 */
static void vecset(int n, double v[], int iv[], int* nzv, int i, double val){
	int k;
	boolean set;

	set = FALSE;
	for(k = 0; k < *nzv; k++){
		if(iv[k] == i){
			v[k] = val;
			set  = TRUE;
		}
	}
	if(set == FALSE){
		v[*nzv]  = val;
		iv[*nzv] = i;
		*nzv     = *nzv + 1;
	}
}

static void task_chunk(int& beg,
		int& end,
		int& chunk,
		const int& size,
		const int& index,
		const int& bsize,
		bool& breaklp){
	if (size - index >= 2*bsize) {
		chunk = bsize;
		breaklp = 0;
	} else {
		chunk = size - index;
		breaklp = 1;
	}
	beg = index;
	end = index + chunk;
}

static void node_chunk(int node_id,
		int& beg,
		int& chunk,
		const int& size,
		int dist_table[]){
	beg = dist_table[node_id];
	chunk = (node_id != nodes-1)
		? (dist_table[node_id+1]
		 - dist_table[node_id  ])
		: (size - dist_table[node_id]);
}
