#include <iostream>
#include <math.h>
#include <random>
#include <vector> 
#include <algorithm>

const int SAMPLE_SIZE = 1000; // How many top solutions to select
const int NUM = 60000;        // Number of solutions per generation

struct Solution
{
    double rank, x, y, z;
    
    Solution() : x(0), y(0), z(0), rank(0) {
        generateRandomValues();
        fitness();
    }

    void generateRandomValues() {
        std::random_device device;
        std::uniform_real_distribution<double> unif(-100000, 100000);
        x = unif(device);
        y = unif(device);
        z = unif(device);
    }

    void fitness()
    {
        double ans = (6 * x - y + std::pow(z, 200)) - 25; // Expression to optimize

        // Farther away the number is from yields more lower rank.
        // Maximum answer is 9999 (we use 9999 = lim(ans)->0)
        rank = (ans == 0) ? 9999 : std::abs(1 / ans); 
    }
};

int main()
{
    // Create initial random solutions to our problem through 
    // randomly generated values between -10000 to 10000, and store in a vector
    std::vector<Solution> solutions(NUM);
    std::vector<Solution> sample;

    int a = 100;
    while (a > 0)
    {
        /**
         * ALTERATION: // Use std::nth_element to efficiently find the top SAMPLE_SIZE solutions without fully sorting the vector.
        */
        std::nth_element(solutions.begin(), solutions.begin() + SAMPLE_SIZE, solutions.end(), 
        [](const auto &lhs, const auto &rhs) {
            return lhs.rank > rhs.rank;
        });

        // Print ranks of the SAMPLE_SIZE solutions
        for (int i = 0; i < SAMPLE_SIZE; ++i)
        {
            const auto &s = solutions[i];
            std::cout << std::fixed // fixed-point notation
                      << "Rank " << static_cast<int>(s.rank)
                      << "\nx:" << s.x << " y:" << s.y << " z:" << s.z << " " << a << " \n";
        }

        /**
        * ALTERATION: Reserve memory in the sample vector for SAMPLE_SIZE elements.
        * Avoids reallocation during the move operation & improve performance
        */
        sample.reserve(SAMPLE_SIZE);

        /**
        * ALTERATION: Use std::move to move top SAMPLE_SIZE solutions from solutions vector to sample vector,
        * utilizing move semantics instead of copying
        */
        std::move(solutions.begin(), solutions.begin() + SAMPLE_SIZE, std::back_inserter(sample));

        // Clear solutions vector & prepare it for the next iteration.
        solutions.clear();


        // Mutate the top solutions by %
        std::random_device device; // Random number generator device
        std::uniform_real_distribution<double> m(0.99, 1.01); // Uniform distribution for mutation factor

        // Mutate each solution in the sample vector
        for (auto &s : sample)
        {
            // Apply mutation to solution parameters x,y,z
            s.x *= m(device);
            s.y *= m(device);
            s.z *= m(device);

            // Recalculate solution fitness
            s.fitness();
        }


        // Cross over operation to create new solutions based on selected parents
        for (int i = 0; i < NUM; i++)
        {
            // Randomly select a parent solution from sample vector
            const auto &selected = sample[std::uniform_int_distribution<int>(0, SAMPLE_SIZE - 1)(device)];
            
            // Create new solution by emplacing a copy of the selected parent into the solutions vector
            solutions.emplace_back(selected);
        }

        // Clear the sample vector after cross over
        sample.clear();

        // Decrement the iteration count for our loop
        a -= 1;
        
    }
}
