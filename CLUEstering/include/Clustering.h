/// \file Clustering.h
/// \brief Class that represents the clustering algorithm
///

#ifndef clustering_h
#define clustering_h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>
#include <utility>

#include "Kernels.h"
#include "Point.h"
#include "Tiles.h"
#include "deltaPhi.h"

/// @brief The domain_t struct contains the minimum and maximum values of a
/// coordinate.
struct domain_t {
  float min = -std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::max();

  /// @brief empty checks if the domain is empty.
  /// @return True if the domain is empty, false otherwise.
  bool empty() {
    return (min == -std::numeric_limits<float>::max()) && (max == std::numeric_limits<float>::max());
  }
};

/// @brief The ClusteringAlgo class is the main class of the clustering algorithm.
/// It contains the data structures and methods to perform the clustering.
/// @tparam Ndim Number of dimensions of the data.
template <uint8_t Ndim>
class ClusteringAlgo {
public:
  ClusteringAlgo(float dc, float rhoc, float outlierDeltaFactor, int pPBin, std::vector<domain_t> domains) {
    dc_ = dc;
    rhoc_ = rhoc;
    outlierDeltaFactor_ = outlierDeltaFactor;
    pointsPerTile_ = pPBin;
    domains_ = std::move(domains);
  }

  // public data members
  float dc_;    // cut-off distance in the calculation of local density
  float rhoc_;  // minimum density to promote a point as a seed or the
                // maximum density to demote a point as an outlier
  float outlierDeltaFactor_;
  int pointsPerTile_;              // average number of points found in a tile
  std::vector<domain_t> domains_;  // Array containing the domain extremes of every coordinate
  Points<Ndim> points_;

  /// @brief setPoints sets the points to be clustered.
  /// @param n Number of points.
  /// @param coordinates Coordinates of the points.
  /// @param weight Weight of the points.
  /// @return True if the number of points is zero, false otherwise.
  bool setPoints(int n, std::vector<std::vector<float>> coordinates, std::vector<float> weight);
  /// @brief clearPoints clears the points.
  /// @return void
  ///
  /// @note This method is called automatically by setPoints.
  void clearPoints();
  /// @brief calculateNTiles calculates the number of tiles.
  /// @param pointPerBin Average number of points found in a tile.
  /// @return Number of tiles.
  ///
  /// @note This method is called automatically by makeClusters.
  int calculateNTiles(int pointPerBin);
  /// @brief calculateTileSize calculates the size of the tiles.
  /// @param NTiles Number of tiles.
  /// @param tiles_ Tiles object.
  /// @return Array containing the size of the tiles.
  ///
  /// @note This method is called automatically by makeClusters.
  std::array<float, Ndim> calculateTileSize(int NTiles, tiles<Ndim>& tiles_);
  /// @brief makeClusters performs the clustering.
  /// @param ker Kernel object.
  /// @return Array containing the cluster index and the seed flag.
  ///
  /// @note This method is called automatically by makeClusters.
  std::vector<std::vector<int>> makeClusters(kernel const& ker);

private:
  /// @brief prepareDataStructures prepares the data structures.
  /// @param tiles Tiles object.
  /// @return void
  ///
  /// @note This method is called automatically by makeClusters.
  void prepareDataStructures(tiles<Ndim>& tiles);
  /// @brief calculateLocalDensity calculates the local density of the points.
  /// @param tiles Tiles object.
  /// @param ker Kernel object.
  /// @return void
  ///
  /// @note This method is called automatically by makeClusters.
  void calculateLocalDensity(tiles<Ndim>& tiles, kernel const& ker);
  /// @brief calculateDistanceToHigher calculates the distance to the nearest higher point.
  /// @param tiles Tiles object.
  /// @return void
  ///
  /// @note This method is called automatically by makeClusters.
  void calculateDistanceToHigher(tiles<Ndim>& tiles);
  /// @brief findAndAssignClusters finds and assigns the clusters.
  /// @return void
  ///
  /// @note This method is called automatically by makeClusters.
  void findAndAssignClusters();
  /// @brief distance calculates the distance between two points.
  /// @param i Index of the first point.
  /// @param j Index of the second point.
  /// @return Distance between the two points.
  inline float distance(int i, int j) const;
};

// public methods
template <uint8_t Ndim>
bool ClusteringAlgo<Ndim>::setPoints(int n,
                                     std::vector<std::vector<float>> coordinates,
                                     std::vector<float> weight) {
  //points_.clear();
  // input variables
  points_.coordinates_ = std::move(coordinates);
  points_.weight = std::move(weight);

  points_.n = n;
  if (points_.n == 0)
    return 1;

  // result variables
  points_.rho.resize(points_.n, 0);
  points_.delta.resize(points_.n, std::numeric_limits<float>::max());
  points_.nearestHigher.resize(points_.n, -1);
  points_.followers.resize(points_.n);
  points_.clusterIndex.resize(points_.n, -1);
  points_.isSeed.resize(points_.n, 0);
  return 0;
}

template <uint8_t Ndim>
void ClusteringAlgo<Ndim>::clearPoints() {
  points_.clear();
}

template <uint8_t Ndim>
int ClusteringAlgo<Ndim>::calculateNTiles(int pointPerBin) {
  int ntiles{points_.n / pointPerBin};
  try {
    if (ntiles == 0) {
      throw 100;
    }
  } catch (...) {
    std::cout << "pointPerBin is set too high for you number of points. You must lower it in the "
                 "clusterer constructor.\n";
  }
  return ntiles;
}

template <uint8_t Ndim>
std::array<float, Ndim> ClusteringAlgo<Ndim>::calculateTileSize(int NTiles, tiles<Ndim>& tiles_) {
  std::array<float, Ndim> tileSizes;
  int NperDim{static_cast<int>(std::pow(NTiles, 1.0 / Ndim))};

  for (int i{}; i != Ndim; ++i) {
    float tileSize;
    float dimMax{*std::max_element(points_.coordinates_[i].begin(), points_.coordinates_[i].end())};
    float dimMin{*std::min_element(points_.coordinates_[i].begin(), points_.coordinates_[i].end())};
    tiles_.minMax[i] = {dimMin, dimMax};
    tileSize = (dimMax - dimMin) / NperDim;

    tileSizes[i] = tileSize;
  }
  return tileSizes;
}

template <uint8_t Ndim>
std::vector<std::vector<int>> ClusteringAlgo<Ndim>::makeClusters(kernel const& ker) {
  tiles<Ndim> Tiles;
  Tiles.nTiles = calculateNTiles(pointsPerTile_);
  Tiles.resizeTiles();
  Tiles.tilesSize = calculateTileSize(Tiles.nTiles, Tiles);

  prepareDataStructures(Tiles);
  calculateLocalDensity(Tiles, ker);
  calculateDistanceToHigher(Tiles);
  findAndAssignClusters();

  return {points_.clusterIndex, points_.isSeed};
}

// private methods
template <uint8_t Ndim>
void ClusteringAlgo<Ndim>::prepareDataStructures(tiles<Ndim>& tiles) {
  for (int i{}; i < points_.n; ++i) {
    // push index of points into tiles
    std::vector<float> coords;
    for (int j{}; j != Ndim; ++j) {
      coords.push_back(points_.coordinates_[j][i]);
    }
    tiles.fill(coords, i);
  }
}

template <uint8_t Ndim>
void ClusteringAlgo<Ndim>::calculateLocalDensity(tiles<Ndim>& tiles, kernel const& ker) {
  // loop over all points
  for (int i{}; i < points_.n; ++i) {
    // get search box
    std::array<std::vector<int>, Ndim> xjBins;
    for (int j{}; j != Ndim; ++j) {
      xjBins[j] =
          tiles.getBinsFromRange(points_.coordinates_[j][i] - dc_, points_.coordinates_[j][i] + dc_, j);

      // Overflow
      if (points_.coordinates_[j][i] + dc_ > domains_[j].max) {
        std::vector<int> overflowBins =
            std::move(tiles.getBinsFromRange(domains_[j].min, domains_[j].min + dc_, j));
        xjBins[j].insert(xjBins[j].end(), overflowBins.begin(), overflowBins.end());
        // Underflow
      } else if (points_.coordinates_[j][i] - dc_ < domains_[j].min) {
        std::vector<int> underflowBins =
            std::move(tiles.getBinsFromRange(domains_[j].max - dc_, domains_[j].max, j));
        xjBins[j].insert(xjBins[j].end(), underflowBins.begin(), underflowBins.end());
      }
    }
    std::vector<int> search_box;
    tiles.searchBox(xjBins, search_box);

    // loop over bins in the search box
    for (int binId : search_box) {
      // iterate over the points inside this bin
      for (auto binIterId : tiles[binId]) {
        int j{binIterId};
        // query N_{dc_}(i)
        float dist_ij{distance(i, j)};

        if (dist_ij <= dc_) {
          // sum weights within N_{dc_}(i) using the chosen kernel
          points_.rho[i] += ker(dist_ij, i, j) * points_.weight[j];
        }
      }  // end of interate inside this bin
    }    // end of loop over bins in the search box
  }      // end of loop over points
}

template <uint8_t Ndim>
void ClusteringAlgo<Ndim>::calculateDistanceToHigher(tiles<Ndim>& tiles) {
  float dm{outlierDeltaFactor_ * dc_};

  // loop over all points
  for (int i{}; i < points_.n; ++i) {
    // default values of delta and nearest higher for i
    float delta_i{std::numeric_limits<float>::max()};
    int nearestHigher_i{-1};  // if this doesn't change, the point is either a seed or an outlier
    float rho_i{points_.rho[i]};

    // get search box
    std::array<std::vector<int>, Ndim> xjBins;
    for (int j{}; j != Ndim; ++j) {
      xjBins[j] =
          tiles.getBinsFromRange(points_.coordinates_[j][i] - dm, points_.coordinates_[j][i] + dm, j);

      // Overflow
      if (points_.coordinates_[j][i] + dm > domains_[j].max) {
        std::vector<int> overflowBins =
            std::move(tiles.getBinsFromRange(domains_[j].min, domains_[j].min + dm, j));
        xjBins[j].insert(xjBins[j].end(), overflowBins.begin(), overflowBins.end());
        // Underflow
      } else if (points_.coordinates_[j][i] - dm < domains_[j].min) {
        std::vector<int> underflowBins =
            std::move(tiles.getBinsFromRange(domains_[j].max - dm, domains_[j].max, j));
        xjBins[j].insert(xjBins[j].end(), underflowBins.begin(), underflowBins.end());
      }
    }
    std::vector<int> search_box;
    tiles.searchBox(xjBins, search_box);

    // loop over all bins in the search box
    for (int binId : search_box) {
      // iterate over the points inside this bin
      for (auto binIterId : tiles[binId]) {
        int j{binIterId};
        // query N'_{dm}(i)
        bool foundHigher{(points_.rho[j] > rho_i)};
        // in the rare case where rho is the same, use detid
        foundHigher = foundHigher || ((points_.rho[j] == rho_i) && (j > i));
        float dist_ij{distance(i, j)};
        if (foundHigher && dist_ij <= dm) {  // definition of N'_{dm}(i)
          // find the nearest point within N'_{dm}(i)
          if (dist_ij < delta_i) {
            // update delta_i and nearestHigher_i
            delta_i = dist_ij;
            nearestHigher_i = j;
          }
        }
      }  // end of interate inside this bin
    }    // end of loop over bins in the search box

    points_.delta[i] = delta_i;
    points_.nearestHigher[i] = nearestHigher_i;
  }  // end of loop over points
}

template <uint8_t Ndim>
void ClusteringAlgo<Ndim>::findAndAssignClusters() {
  int nClusters{};

  // find cluster seeds and outlier
  std::vector<int> localStack;  // this vector will contain the indexes of all the seeds
  // loop over all points
  for (int i{}; i < points_.n; ++i) {
    // initialize clusterIndex
    points_.clusterIndex[i] = -1;

    float deltai{points_.delta[i]};
    float rhoi{points_.rho[i]};

    // determine seed or outlier
    bool isSeed{(deltai > dc_) && (rhoi >= rhoc_)};
    bool isOutlier{(deltai > outlierDeltaFactor_ * dc_) && (rhoi < rhoc_)};
    if (isSeed) {
      // set isSeed as 1
      points_.isSeed[i] = 1;
      // set cluster id
      points_.clusterIndex[i] = nClusters;
      // increment number of clusters
      ++nClusters;
      // add seed into local stack
      localStack.push_back(i);
    } else if (!isOutlier) {
      // register as follower at its nearest higher
      points_.followers[points_.nearestHigher[i]].push_back(i);
    }
  }

  // expend clusters from seeds
  while (!localStack.empty()) {
    int i{localStack.back()};
    auto& followers{points_.followers[i]};
    localStack.pop_back();

    // loop over followers
    for (int j : followers) {
      // pass id from i to a i's follower
      points_.clusterIndex[j] = points_.clusterIndex[i];
      // push this follower to localStack
      localStack.push_back(j);
    }
  }
}

template <uint8_t Ndim>
inline float ClusteringAlgo<Ndim>::distance(int i, int j) const {
  float qSum{};  // quadratic sum
  for (int k{}; k != Ndim; ++k) {
    float delta_xk{
        deltaPhi(points_.coordinates_[k][i], points_.coordinates_[k][j], domains_[k].min, domains_[k].max)};
    qSum += std::pow(delta_xk, 2);
  }

  return std::sqrt(qSum);
}

#endif
