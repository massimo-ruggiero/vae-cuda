namespace fused {

    namespace forward {

        void linear_leaky_relu();
        void linear_mu_sigma();
        void linear_sigmoid();

        // tensore cores
        void linear_leaky_relu_tc();
        void linear_mu_sigma_tc();
        void linear_sigmoid_tc();

        // cuBLAS
        void linear_leaky_relu_cublas();
        void linear_mu_sigma_cublas();
        void linear_sigmoid_cublas();

    }

    namespace backward {

        void linear_leaky_relu();
        void linear_mu_sigma();
        void linear_sigmoid();

        // tensore cores
        void linear_leaky_relu_tc();
        void linear_mu_sigma_tc();
        void linear_sigmoid_tc();

        // cuBLAS
        void linear_leaky_relu_cublas();
        void linear_mu_sigma_cublas();
        void linear_sigmoid_cublas();

    }

}