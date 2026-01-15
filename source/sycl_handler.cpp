#include <funlib/sycl/sycl_handler.hpp>



void flib::sycl_handler::select_device(std::string device_name)
{
    //Selects a SYCL Device to work with
    //If no device is found, it throws an error
    //If this function is not called, uses a default device
    sycl::device selected_device;
        for (const auto& platform : sycl::platform::get_platforms()) {
            for (const auto& device : platform.get_devices()) {
                std::string devname = device.get_info<sycl::info::device::name>();
                //Lets pas the whole string to uppercase
                std::transform(devname.begin(), devname.end(), devname.begin(), ::toupper);
                std::transform(device_name.begin(), device_name.end(), device_name.begin(), ::toupper);
                //Check if the devname s  tring contains the device_name string
                if (devname.find(device_name) != std::string::npos) {
                    _device = device;
                    _queue = sycl::queue(_device);
                    std::cout << "Selected Device : " << devname << std::endl;
                    return;
                }
    
            }
        }
    throw std::runtime_error("SYCL: Device not found!");
}

void flib::sycl_handler::get_platform_info()
{
const auto platforms = sycl::platform::get_platforms();

    for (const auto &platform : platforms) {
        std::string platform_name = platform.get_info<sycl::info::platform::name>();
        std::cout << "Platform: " << platform_name << "\n";

        const auto devices = platform.get_devices();
        if (devices.empty()) {
            std::cout << "   (No devices found on this platform)\n";
        }

        for (const auto &device : devices) {
            std::string device_name = device.get_info<sycl::info::device::name>();
            std::string vendor      = device.get_info<sycl::info::device::vendor>();
            auto type               = device.get_info<sycl::info::device::device_type>();

            std::string type_str;
            switch (type) {
                case sycl::info::device_type::cpu: type_str = "CPU"; break;
                case sycl::info::device_type::gpu: type_str = "GPU"; break;
                case sycl::info::device_type::accelerator: type_str = "Accelerator"; break;
                default: type_str = "Other"; break;
            }

            std::cout << "   - Device: " << device_name << " [" << type_str << "]"
                      << " (Vendor: " << vendor << ")\n";
        }
        std::cout << std::endl;
    }

}
void flib::sycl_handler::select_backend_device(const std::string &platform_filter, const std::string &device_filter)
{
    std::string target_platform = platform_filter;
    std::string device_type_filter = device_filter;
    std::transform(target_platform.begin(), target_platform.end(), target_platform.begin(), ::toupper);
    std::transform(device_type_filter.begin(), device_type_filter.end(), device_type_filter.begin(), ::toupper);
     
    sycl::info::device_type target_type = device_type_from_string(device_type_filter);

    for (const auto& platform : sycl::platform::get_platforms()) {
        std::string pname = platform.get_info<sycl::info::platform::name>();
        std::transform(pname.begin(), pname.end(), pname.begin(), ::toupper);
        if (pname.find(target_platform) != std::string::npos) {
            for (const auto& device : platform.get_devices()) {
                if (device.get_info<sycl::info::device::device_type>() == target_type) {
                    _device = device;
                    _queue = sycl::queue(_device);
                    _platform = platform;
                    std::cout << "Selected Device : "
                                << _device.get_info<sycl::info::device::name>() << "\n";
                    std::cout << "   Platform        : "
                                << _platform.get_info<sycl::info::platform::name>() << "\n";
                    std::cout << "   Device Type     : " << device_type_filter << "\n";
                    return;
                }
            }
        }
    }

    throw std::runtime_error("SYCL: No matching platform/device found for platform '" +
                                platform_filter + "' and device type '" + device_type_filter + "'");
}

void flib::sycl_handler::create_gl_interop_context()
{
    auto glxContext = glXGetCurrentContext();
    auto glxDisplay = glXGetCurrentDisplay(); //returns the display for the current context.
    if (!glxContext || !glxDisplay)
        throw std::runtime_error("OpenGL context is not current in this thread.");

    //Creates based on the user selected device and platform
    cl_platform_id clPlatform = sycl::get_native<sycl::backend::opencl>(_platform);
    cl_device_id clDev = sycl::get_native<sycl::backend::opencl>(_device);

    char extensions[2048];
    clGetDeviceInfo(clDev, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);
    if(std::string(extensions).find("cl_khr_gl_sharing") == std::string::npos) {
        throw std::runtime_error("OpenCL device does not support OpenGL interoperability.");
    }

    cl_context_properties props[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)glxContext,
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glxDisplay,
            CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatform,
            0
    };

    cl_int err = CL_SUCCESS;
    _clCtx = clCreateContext(props, 1, &clDev, nullptr, nullptr, &err);
    if (!_clCtx || err != CL_SUCCESS)
        throw std::runtime_error("Failed to create OpenCL context for OpenGL interoperability.");

    _syclCtx = sycl::make_context<sycl::backend::opencl>(_clCtx);
    _queue = sycl::queue(_syclCtx, _device, sycl::property::queue::in_order());

}
bool flib::sycl_handler::is_rtc_available()
{
   if(_queue.get_device().ext_oneapi_can_compile(sycl::ext::oneapi::experimental::source_language::sycl)) {
        return true;
    }
    return false;
}
void flib::sycl_handler::get_device_info()
{
    //Prints the current device 
   
    std::cout << "Current Device for computations : " 
        << _queue.get_device().get_info<sycl::info::device::name>()<<std::endl;


    sycl::device dev = _queue.get_device();
    sycl::context ctx = _queue.get_context();
    sycl::platform plt = dev.get_platform();
    

    std::cout << "------------------------------------\n";
    std::cout << "Device Name     : " << dev.get_info<sycl::info::device::name>() << "\n";
    std::cout << "Vendor          : " << dev.get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "Driver Version  : " << dev.get_info<sycl::info::device::driver_version>() << "\n";
    std::cout << "Platform Name   : " << plt.get_info<sycl::info::platform::name>() << "\n";
    std::cout << "SYCL Backend    : ";
    auto platform_name = plt.get_info<sycl::info::platform::name>();
   if (platform_name.find("OpenCL") != std::string::npos) {
       std::cout << "OpenCL\n";
    } else if (platform_name.find("Level-Zero") != std::string::npos) {
        std::cout << "Level-Zero\n";
    } else if (platform_name.find("CUDA") != std::string::npos) {
        std::cout << "CUDA\n";
    } else if (platform_name.find("HIP") != std::string::npos) {
       std::cout << "HIP\n";
    } else {
        std::cout << "Unknown Platform\n";
    }

    std::cout << "------------------------------------\n";
}

void flib::sycl_handler::sys_info()
{
        std::cout << "======================" << std::endl;
        std::cout << "   SYCL SYSTEM INFO   " << std::endl;
        std::cout << "======================" << std::endl;

        int device_count = 0;
        try {
            for (const auto& platform : sycl::platform::get_platforms()) {
                for (const auto& device : platform.get_devices()) {
                    std::cout << "Device Name      : " << device.get_info<sycl::info::device::name>() << std::endl;
                    std::cout << "Vendor           : " << device.get_info<sycl::info::device::vendor>() << std::endl;
                    std::cout << "Driver Version   : " << device.get_info<sycl::info::device::driver_version>() << std::endl;

                    std::string deviceType;
                    switch (device.get_info<sycl::info::device::device_type>()) {
                        case sycl::info::device_type::cpu: deviceType = "CPU"; break;
                        case sycl::info::device_type::gpu: deviceType = "GPU"; break;
                        case sycl::info::device_type::accelerator: deviceType = "Accelerator"; break;
                        default: deviceType = "Unknown";
                    }
                    std::cout << "Device Type       : " << deviceType << std::endl;
                    
                    std::cout << "Max Compute Units : " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
                    std::cout << "Global Memory     : " 
                              << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024) 
                              << " MB" << std::endl;
                    std::cout << "Local Memory      : " 
                              << device.get_info<sycl::info::device::local_mem_size>() / (1024)
                              << " MB" << std::endl;
                    // Work Item Info
                    std::cout << "Max Work Group Size   : " 
                              << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;

                    auto max_work_item_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
                    std::cout << "Max Work Item Sizes   : (" 
                              << max_work_item_sizes[2] << ", "
                              << max_work_item_sizes[1] << ", "
                              << max_work_item_sizes[0] << ")" << std::endl;

                    // Removed invalid query: max_work_items_per_compute_unit
                    // std::cout << "Max Work Items per CU : " << device.get_info<sycl::info::device::max_work_items_per_compute_unit>() << std::endl;

                    std::cout << "----------------------" << std::endl;
                    device_count++;
                }
            }
        } catch (sycl::exception const& e) {
            std::cerr << "SYCL Exception: " << e.what() << std::endl;
        }

        std::cout << "Total SYCL devices found: " << device_count << std::endl;
}
sycl::info::device_type flib::sycl_handler::device_type_from_string(const std::string &_type_str)
{
        std::string type_str = _type_str;
        std::transform(type_str.begin(), type_str.end(), type_str.begin(), ::toupper);
        if (type_str == "CPU")
            return sycl::info::device_type::cpu;
        else if (type_str == "GPU")
            return sycl::info::device_type::gpu;
        else if (type_str == "ACCELERATOR")
            return sycl::info::device_type::accelerator;
        else
            throw std::runtime_error("Invalid device type string: " + type_str);
    
}
sycl::queue flib::sycl_handler::get_queue()
{
    return _queue;
}
cl_context flib::sycl_handler::get_clContext()
{
    return _clCtx;
}
sycl::context flib::sycl_handler::get_sycl_context()
{
    return _syclCtx;
}
// Initialize static members
sycl::device flib::sycl_handler::_device;
sycl::platform flib::sycl_handler::_platform;
cl_context flib::sycl_handler::_clCtx = nullptr; // Initialize OpenCL context
sycl::queue  flib::sycl_handler::_queue{sycl::default_selector_v}; //Default queue
sycl::context flib::sycl_handler::_syclCtx;

