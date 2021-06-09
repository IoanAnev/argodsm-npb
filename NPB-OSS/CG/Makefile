SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

# default benchmark version
EXECUTABLE=${BENCHMARK}-ompss-2

ifdef version
	ifeq "$(version)" "ompss-2-cluster-strong"
		EXECUTABLE=${BENCHMARK}-ompss-2-cluster-strong
	endif
endif
ifdef version
	ifeq "$(version)" "ompss-2-cluster-weak"
		EXECUTABLE=${BENCHMARK}-ompss-2-cluster-weak
	endif
endif

OBJS = ${EXECUTABLE}.o ${COMMON}/c_print_results.o  \
       ${COMMON}/c_${RAND}.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

${EXECUTABLE}.o: ${EXECUTABLE}.cpp npbparams.hpp
	${CCOMPILE} ${EXECUTABLE}.cpp

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.hpp core
