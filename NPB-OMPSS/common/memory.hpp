/**
 * @file
 * @brief This file implements wrappers for
 *        the nanos6 memory allocation calls
 */

#ifndef __NPB_MEMORY_HPP__
#define __NPB_MEMORY_HPP__

#include <iostream>
using namespace std;

/** @brief wrapper for the nanos6_lmalloc call */
template<typename T>
static inline T* lmalloc(size_t size)
{
	T* alloc = (T*)nanos6_lmalloc(sizeof(T) * size);
	if (!alloc)
		std::cerr << "Could not allocate nanos6 "
			  << "local memory"
			  << std::endl;
	return alloc;
}

/** @brief wrapper for the nanos6_dmalloc call */
template<typename T>
static inline T* dmalloc(size_t size,
		nanos6_data_distribution_t policy =
				nanos6_equpart_distribution,
		size_t num_dimensions = 0,
		size_t *dimensions = NULL)
{
	T* alloc = (T*)nanos6_dmalloc(sizeof(T) * size,
			policy, num_dimensions, dimensions);
	if (!alloc)
		std::cerr << "Could not allocate nanos6 "
			  << "distributed memory"
			  << std::endl;
	return alloc;
}

/** @brief wrapper for the nanos6_lfree call */
template<typename T>
static inline void lfree(void* ptr, size_t size)
{
	nanos6_lfree(ptr, sizeof(T) * size);
}

/** @brief wrapper for the nanos6_dfree call */
template<typename T>
static inline void dfree(void* ptr, size_t size)
{
	nanos6_dfree(ptr, sizeof(T) * size);
}

#endif /* __NPB_MEMORY_HPP__ */
