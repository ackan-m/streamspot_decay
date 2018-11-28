// #ifndef STREAMSPOT_DECAYED_TRAIN_H_
// #define STREAMSPOT_DECAYED_TRAIN_H_

#include <bitset>
#include <chrono>
#include "param.h"
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "param.h"

namespace std{
  void decayed_trained_streamhash_projection(const edge& e, const vector<graph>& graphs,
                              vector<bitset<L>>& streamhash_sketches,
                              vector<vector<double>>& streamhash_projections,
                             uint32_t chunk_length,
                             const vector<vector<uint64_t>>& H
                           );
}
