#pragma once

#include "bench_core.cuh"


void run_sgemm(Csv& csv, curandGenerator_t gen, 
               Timer& timer, 
               const BenchmarkConfig& config);