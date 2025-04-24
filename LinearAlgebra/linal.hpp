#if !defined(MACRO)
#define MACRO
#include "../Set/set.hpp"
#include "../Set/set_operations.hpp"
#include <cmath>
namespace flib
{
    namespace linal
    {

        template<typename T>
        void conjugate_grad(const Set<T>& A, const Set<T> & b, Set<T> & x, int max_iter = 1000, T tol = 1e-10);
        

        
    } // namespace linal
    
} // namespace flib


#endif // MACRO
