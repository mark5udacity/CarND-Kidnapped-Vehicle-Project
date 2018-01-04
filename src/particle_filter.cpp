/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang, Mark Veronda
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    if (is_initialized) {
        cout << "WARN: Already been initialized, why is this being called again!?" << endl;
        return;
    }

    dist_x = *new normal_distribution<double>(x, std[0]);
    dist_y = *new normal_distribution<double>(y, std[1]);
    dist_theta = *new normal_distribution<double>(theta, std[2]);

    num_particles = NUM_PARTICLES;
    const double uniform_wieght = 1. / num_particles;
    for (int i = 0; i < num_particles; i++) {
        particles.push_back(Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), uniform_wieght});
        //cout << "ps: " << particles[particles.size() - 1] << endl;
    }

    cout << "Created " << num_particles << " particles!" << "\n";

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: ... and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    /*
     * Transformation equations
     * x_f​ = x_0 + θ˙/v​  [ sin(θ_0​ + θ˙(dt)) − sin(θ_0​)          ]
     * y_f​ = y_0​ + θ˙/v​  [ cos(θ_0​)          − cos(θ_0​ + θ˙(dt)) ]
     * θ_f​ = θ_0​ + θ˙(dt)
     */

    for (Particle particle : particles) {
        const double v_over_theta = velocity / yaw_rate;
        const double theta_dt = yaw_rate * delta_t;
        const double new_orientation = particle.theta + theta_dt;
        particle.x += v_over_theta * (sin(new_orientation) - sin(particle.theta));
        particle.y += v_over_theta * (cos(particle.theta) - cos(new_orientation));
        particle.theta = new_orientation;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
//   observed measurement to this particular landmark.
// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
//   according to the MAP'S coordinate system. You will need to transform between the two systems.
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation 
//   3.33
//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
// TODO: Resample particles with replacement with probability proportional to their weight. 
// NOTE: You may find std::discrete_distribution helpful here.
//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
vector<int> v = best.associations;
stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
vector<double> v = best.sense_x;
stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
vector<double> v = best.sense_y;
stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
