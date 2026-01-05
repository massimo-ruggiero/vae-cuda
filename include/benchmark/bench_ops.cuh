#pragma once

#include "bench_core.cuh"


void run_sgemm(Csv& csv, curandGenerator_t gen, 
               Timer& timer, 
               const BenchmarkConfig& config);

void run_transpose(Csv& csv, curandGenerator_t gen, 
                   Timer& timer, 
                   const BenchmarkConfig& config);

void run_add_in_place(Csv& csv, curandGenerator_t gen, 
                      Timer& timer, 
                      const BenchmarkConfig& config);

void run_add_bias(Csv& csv, curandGenerator_t gen, 
                  Timer& timer, 
                  const BenchmarkConfig& config);

void run_leaky_relu_forward(Csv& csv, curandGenerator_t gen, 
                            Timer& timer, 
                            const BenchmarkConfig& config);

void run_leaky_relu_backward(Csv& csv, curandGenerator_t gen, 
                             Timer& timer, 
                             const BenchmarkConfig& config);

void run_sigmoid_forward(Csv& csv, curandGenerator_t gen, 
                         Timer& timer, 
                         const BenchmarkConfig& config);

void run_sigmoid_backward(Csv& csv, curandGenerator_t gen, 
                          Timer& timer, 
                          const BenchmarkConfig& config);

void run_loss_forward(Csv& csv, curandGenerator_t gen, 
                      Timer& timer, 
                      const BenchmarkConfig& config);

void run_bce_backward(Csv& csv, curandGenerator_t gen, 
                      Timer& timer, 
                      const BenchmarkConfig& config);

void run_kl_backward(Csv& csv, curandGenerator_t gen, 
                     Timer& timer, 
                     const BenchmarkConfig& config);

void run_reparam_forward(Csv& csv, curandGenerator_t gen, 
                         Timer& timer, 
                         const BenchmarkConfig& config);

void run_reparam_backward(Csv& csv, curandGenerator_t gen, 
                          Timer& timer, 
                          const BenchmarkConfig& config);

void run_adam_step(Csv& csv, curandGenerator_t gen, 
                   Timer& timer, 
                   const BenchmarkConfig& config);
