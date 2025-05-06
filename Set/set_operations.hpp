#if !defined(_SET_OPERATIONS_HPP_)
#define _SET_OPERATIONS_HPP_

#include "../Set/set.hpp"
namespace flib{

    
    class set_operations{

        class sycl_handler;

        //This class is used to perform operations on the Set class
        //It is a friend class of the Set class
        //It is used to perform operations on the Set class
        //It is a friend class of the sycl_handler class
        //It is used to perform operations on the Set class
        template<typename T>
        static Set<T> gemm(const Set<T>& A, const Set<T>& B);
        template<typename T>
        static Set<T> matXvec(const Set<T>& A, const Set<T>& B);
        public:
            template<typename T>
            static Set<T> prod(const Set<T>& A, const Set<T>& B);
        
            //dot product of two vectors or two one dimensional sets
            template<typename T>
            static T dot(const Set<T>& A, const Set<T>& B);
            template<typename T>
            static T reduction(const Set<T>& A);
           
    
    };
    //class sycl_handler;
    //Matrix times a vector Ax = b or Matrix times a matrix AB = C
   



};


#endif // _SET_OPERATIONS_HPP_)
