#if !defined(_SYCL_HANDLER_H_)
#define _SYCL_HANDLER_H_
#include <sycl/sycl.hpp>
#include "../Set/set.hpp"
namespace flib
{
    class sycl_handler {

        static sycl::device _device;
        static sycl::queue _queue;
    protected:
        static sycl::queue get_queue();
       
    public:

        
        template<typename T, typename Func>
        friend class ParticleSystem;

        
        friend class set_operations;

        static void select_device(std::string device_name);
        static void get_device_info();
        static void sys_info();
    
        //Set operations
        /*template <typename T>
        friend Set<T> prod(const Set<T>& A, const Set<T>& B);
        template <typename T>
        friend T dot(const Set<T>& A, const Set<T>& B);*/
        
    
    
    };

} // namespace flib

#include "../Set/set_operations.hpp"
#endif // _SYCL_HANDLER_H_
