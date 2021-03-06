/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang, Mark Veronda
 */

#include <random>
#include <iostream>
#include <sstream>

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
    const double uniform_weight = 1. ; /// num_particles;
    for (int i = 0; i < num_particles; i++) {
        particles.push_back(Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), uniform_weight});
        //cout << "ps: " << particles[particles.size() - 1] << endl;
    }

    cout << "Created " << num_particles << " particles!" << "\n";

    // For later use with Gaussian noise
    dist_x = *new normal_distribution<double>(0., std[0]);
    dist_y = *new normal_distribution<double>(0., std[1]);
    dist_theta = *new normal_distribution<double>(0., std[2]);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double velocity, double yaw_rate) {

    /*
     * Transformation equations
     * x_f​ = x_0 + θ˙/v​  [ sin(θ_0​ + θ˙(dt)) − sin(θ_0​)          ]
     * y_f​ = y_0​ + θ˙/v​  [ cos(θ_0​)          − cos(θ_0​ + θ˙(dt)) ]
     * θ_f​ = θ_0​ + θ˙(dt)
     */

    int curPid = 0;
    for (Particle& particle : particles) {
        particle.id = curPid++;
        const double theta_dt = yaw_rate * delta_t;
        const double new_orientation = particle.theta + theta_dt;
        if (fabs(yaw_rate) > 0.001) {
            const double v_over_theta = velocity / yaw_rate;
            particle.x += v_over_theta * (sin(new_orientation) - sin(particle.theta));
            particle.y += v_over_theta * (cos(particle.theta) - cos(new_orientation));
        } else {
            const double velo_dt = velocity * delta_t;
            particle.x += velo_dt * cos(new_orientation);
            particle.y += velo_dt * sin(new_orientation);
        }

        // Add noise
        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta = new_orientation + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> in_range, std::vector<LandmarkObs>& observations) {
    for (auto it = observations.begin() ; it != observations.end(); it++) {
        const LandmarkObs curObs = *it;
        double shortest = INFINITY;
        int closest_landmark = 1; //TODO: better default when in_range is empty??
        for (const LandmarkObs curL : in_range) {
            const double curDist = dist(curL.x, curL.y, curObs.x, curObs.y);
            if (curDist < shortest) {
                closest_landmark = curL.id;
                shortest = curDist;
            }
        }

        it->id = closest_landmark;
    }
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    for (auto it = particles.begin(); it != particles.end(); it++) {
        const Particle particle = *it;
        std::vector<LandmarkObs> converted = convert_observations(particle, observations);
        std::vector<LandmarkObs> in_range = find_in_range(map_landmarks.landmark_list, particle, sensor_range);
        dataAssociation(in_range, converted);

        double totalWeight = 1.; // TODO: Unspecified what to do when empty landmarks??
        std::vector<int> assoc_landmarks;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        for (const LandmarkObs curObs : converted) {
            const LandmarkObs closestLandmark = findClosest(curObs.id, map_landmarks.landmark_list);
            const double curWeight = calculateWeight(curObs, closestLandmark, std_landmark);
            if (curWeight > 0.) {
                totalWeight *= curWeight;
            } else {
                //cout << "Found non-positive weight!! : " << curWeight << "\n";
            }

            //set particle associations
            assoc_landmarks.push_back(curObs.id);
            sense_x.push_back(curObs.x);
            sense_y.push_back(curObs.y);
        }

        SetAssociations(*it, assoc_landmarks, sense_x, sense_y);
        if (!converted.empty()) {
            it->weight = totalWeight;
        } else {
            it->weight = 0;
            //cout << "Found non-positive weight!! : " << totalWeight << "\n";
        }
    }
}

std::vector<LandmarkObs> ParticleFilter::convert_observations(const Particle particle,
                                                              const std::vector<LandmarkObs> &observations) {
    vector<LandmarkObs> converted;

    /*
     * _m == map coordinates
     * _p == map particle coordinates (aka: vehicle)
     * _c == car observation coordinates (aka: sensor provided)
     *
     * x_m​ = x_p​ + (cos(θ) * x_c​) − (sin(θ) * y_c​)
     * y_m​ = y_p​ + (sin(θ) * x_c) + (cos(θ) * y_c​)
     */

    for (LandmarkObs curObs : observations) {
        const double x = particle.x + cos(particle.theta) * curObs.x - sin(particle.theta) * curObs.y;
        const double y = particle.y + sin(particle.theta) * curObs.x + cos(particle.theta) * curObs.y;
        converted.push_back({curObs.id, x, y});
        //cout << "observed: (" << curObs.x <<", " << curObs.y
        //     << ") converted: " << curObs.id << "(" << x << ", " << y << ")" << "\n";
    }

    return converted;
}


std::vector<LandmarkObs> ParticleFilter::find_in_range(const std::vector<Map::single_landmark_s> map_landmarks,
                                                       const Particle particle,
                                                       const double sensor_range) {
    vector<LandmarkObs> in_range;
    for (const Map::single_landmark_s curLandmark : map_landmarks) {
        const double cur_dist = dist(particle.x, particle.y, curLandmark.x_f, curLandmark.y_f);
        if (cur_dist <= sensor_range) {
            in_range.push_back({curLandmark.id_i, curLandmark.x_f, curLandmark.y_f});
        }
    }

    return in_range;
}

// Perf consideration: maybe landmarks should be part of initialization, that way we can move this into a set and lookup becomes O(1)
const LandmarkObs ParticleFilter::findClosest(int landmark_id, std::vector<Map::single_landmark_s> landmarks) {
    Map::single_landmark_s* closest = NULL;

    // Try by index first
    int closestIdx = landmark_id - 1;
    if (closestIdx < 0) {
        // Or could throw exception and check this much earlier than this point
        cout << "WARN: Did not expect any Landmark ID to be less than 1, but was: " << closestIdx + 1 << "\n";
    } else if (landmarks[closestIdx].id_i == landmark_id) {
        closest = &landmarks[closestIdx];
    } else {
        cout << "WARN: Landmark ID did not match at expected index " << closestIdx + 1 << "\n";
    }

    // If above fails, do a O(n) list search
    if (!closest) {
        for (Map::single_landmark_s curLm : landmarks) {
            if (curLm.id_i == landmark_id) {
                closest = &curLm;
                break;
            }
        }
    }

    if (!closest) {
        throw std::invalid_argument("Couldn't find landmark ID!  Should never happen unless something is wrong with identifying landmarks!");
    }

    return {closest->id_i, closest->x_f, closest->y_f};
}

double ParticleFilter::calculateWeight(const LandmarkObs obs, const LandmarkObs lmark, const double std[]) {
    /*
     * P(x,y)= 1 / (2 * π * σ_x​ * σ_y​)​ exp −( (x − μ_x​ )^2 / 2 * σ_x^2​
     *                                             +  (y − μ_y​)^2 / 2 * σ_y^2​​ )
     *
     * x, y = observation in map's x,y coords
     * μ = nearest landmark coords
     */
    const double gauss_norm = (1 / (2 * M_PI * std[0] * std[1]));
    const double exponent = (pow(obs.x - lmark.x, 2) / (2 * pow(std[0], 2)))
                                 + (pow(obs.y - lmark.y, 2) / (2 * pow(std[1], 2)));

    return gauss_norm * exp(-exponent);
}

void ParticleFilter::resample() {
    std::vector<double> weights;
    std::transform(particles.begin(), particles.end(),
                   std::back_inserter(weights),
                   [](const Particle& p){ return p.weight; });

    std::discrete_distribution<> weights_dist(weights.begin(), weights.end());

    std::vector<Particle> resampled;
    for (int i = 0; i < num_particles; i++) {
        resampled.push_back(particles[weights_dist(gen)]);
    }

    cout << "all done!" << endl;
    particles = resampled;
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
    return particle;
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
