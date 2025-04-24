#include "linal.hpp"
namespace flib
{
    
    namespace linal
    {
        template<typename T>
        void conjugate_grad(const Set<T>& A, const Set<T> & b, Set<T> & x, int max_iter, T tol){
            std::size_t N = b.getRows();
            
            if(A.getRows() != N || A.getCols() != N || b.getCols() != 1 || x.getCols() != 1 || x.getRows() != N){
                throw std::invalid_argument("Matrix and vector dimensions do not match");
            }
            
            flib::Set<T> r(N);
            flib::Set<T> p(N);
            flib::Set<T> Ap(N);

            //Computing the inital residual

            r = flib::set_operations::prod(A, x);
            //std::cout << "Initial residual: "<< "\n";
            //r.print();
            for(std::size_t i = 0; i < N; i++){
                r[i] = b[i] - r[i];
            }
            //std::cout << "Residual after b-r: "<< "\n";
            //r.print();
            p = r;

            T old_r = flib::set_operations::dot(r, r);
            // std::cout << "Initial residual: "<< old_r << "\n";
            int iter = 0;
            while(iter < max_iter){

                Ap = flib::set_operations::prod(A, p);
                //std::cout << "Matrix times vector: "<< "\n";
                //Ap.print();
                T denominator = flib::set_operations::dot(p, Ap);
                T alpha = old_r / denominator;

                for(std::size_t i = 0; i < N; i++){
                    x[i] = x[i] + alpha * p[i]; //updates the solution
                    r[i] = r[i] - alpha * Ap[i]; //updates the residual
                }   
                T new_r = flib::set_operations::dot(r, r);
                T currTol = std::sqrt(new_r);
                if(currTol < tol){
                    std::cout << "Converged in " << iter + 1 << " iterations.\n";
                   return;
                }

                T beta = new_r / old_r;
                for(std::size_t i = 0; i < N; i++){
                    p[i] = r[i] + beta * p[i]; //updates the search direction
                }
                old_r = new_r;
                iter++;
                
            }

            std::cout<< "Reached max iterations without convergence.\n";

        }
       
        
        // Other linear algebra functions can be defined here...
        
    } // namespace linal
     //Explicit instantiations (VERY IMPORTANT)

    template void linal::conjugate_grad(const Set<double>&, const Set<double>&, Set<double>&, int, double);
    template void linal::conjugate_grad(const Set<float>&, const Set<float>&, Set<float>&, int, float);
   

} // namespace flib
