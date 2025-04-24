#include "../include/funlib.hpp"
#include <iostream>


int main() {
    
    
    flib::sycl_handler::sys_info(); // Prints system info
    flib::sycl_handler::select_device("Intel"); // Selects a vendor for your computations
    flib::sycl_handler::get_device_info(); // Prints current device info

    std::size_t nparticles = 10;

    flib::ParticleSet<float> particles;

    particles.SetNumParticles(nparticles);

    std::cout << "Initial particles:" << std::endl;
    particles.print();
    std::cout << "----------------------------------------" << std::endl;

    /* This is the user defined function that will be used to update the particles
     in the SYCL kernel */
    auto myFunc = [](flib::Particle<float> &p, float dt) {
        for (int i = 0; i < 3; i++) {
            p.velocity[i] += 1.0f; // Update velocity
            p.position[i] += p.velocity[i];
        }
    };

    std::cout << "Updated particles:" << std::endl;

    flib::ParticleSystem<float, decltype(myFunc)>::update(particles, myFunc, 0.01f);

    particles.print(); 
    std::cout << "----------------------------------------" << std::endl;



    return 0;
}