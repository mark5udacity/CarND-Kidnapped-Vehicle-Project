/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang, Mark Veronda
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

static const int NUM_PARTICLES = 55;

#include "helper_functions.h"
#include <random>

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;

	friend std::ostream& operator<<(std::ostream& out, const Particle &p) {
		return out << p.id << "(" << p.x << "," << p.y << "," << p.theta << ")" << "\n";
	}
};



class ParticleFilter {

private:

	int num_turns;

	// Number of particles to draw
	int num_particles;

	// Flag, if filter is initialized
	bool is_initialized;

	// Distributions
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x;
	std::normal_distribution<double> dist_y;
	std::normal_distribution<double> dist_theta;

	// helper functions

    // Calculates weight of an observation to given landmark, basic L2 distance measurement
    double calculateWeight(const LandmarkObs obs, const LandmarkObs landmarkObs, const double std[]);

    // Find closet landmark given id
    const LandmarkObs findClosest(int landmark_id, std::vector<Map::single_landmark_s> vector);

	// Takes vehicle-based-coord observations and translate them into map-based-coordinates
	std::vector<LandmarkObs> convert_observations(const Particle particle, const std::vector<LandmarkObs> &observations);

    // Find map landmarks with-in sensor range
    std::vector<LandmarkObs> find_in_range(const std::vector<Map::single_landmark_s> map_landmarks,
                                           const Particle particle,
                                           double sensor_range);

public:
	
	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param num_particles Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double velocity, double yaw_rate);
	
	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param in_range Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */
	void dataAssociation(std::vector<LandmarkObs> in_range, std::vector<LandmarkObs>& observations);
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
			const Map &map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle SetAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y);

	
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized;
	}
};

#endif /* PARTICLE_FILTER_H_ */
