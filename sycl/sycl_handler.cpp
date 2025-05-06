#include "sycl_handler.hpp"

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
                //Check if the devname string contains the device_name string
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

void flib::sycl_handler::get_device_info()
{
    //Prints the current device 
   
    std::cout << "Current Device for computations : " 
        << _queue.get_device().get_info<sycl::info::device::name>()<<std::endl;
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

sycl::queue flib::sycl_handler::get_queue()
{
    return _queue;
}

// Initialize static members
sycl::device flib::sycl_handler::_device;
sycl::queue  flib::sycl_handler::_queue{sycl::default_selector_v}; //Default queue

