#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include "graph.h"
#include "hash.h"
#include <iostream>
#include "param.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "decayed_train.h"

namespace std {
  // tuple<vector<double>, chrono::nanoseconds, chrono::nanoseconds>
  // update_streamhash_sketche(const edge& e, const vector<graph>& graphs,
  //                            vector<bitset<L>>& streamhash_sketches,
  //                            // vector<vector<double>>& streamhash_projections,
  //                            uint32_t chunk_length,
  //                            const vector<vector<uint64_t>>& H
  //                            )
 void decayed_trained_streamhash_projection(const edge& e, const vector<graph>& graphs,
                             vector<bitset<L>>& streamhash_sketches,
                             vector<vector<double>>& streamhash_projections,
                            uint32_t chunk_length,
                            const vector<vector<uint64_t>>& H
                            ){
    // source node = (src_id, src_type)
    // dst_node = (dst_id, dst_type)
    // shingle substring = (src_type, e_type, dst_type)
    //assert(K == 1 && chunk_length >= 4);

    // for timing
    // chrono::time_point<chrono::steady_clock> start;
    // chrono::time_point<chrono::steady_clock> end;
    // chrono::microseconds shingle_construction_time;
    // chrono::microseconds sketch_update_time;

    auto& src_id = get<F_S>(e);
    auto& src_type = get<F_STYPE>(e);
    auto& gid = get<F_GID>(e);

    auto& sketch = streamhash_sketches[gid];
    auto& projection = streamhash_projections[gid];
    auto& g = graphs[gid];

    // start = chrono::steady_clock::now(); // start shingle construction

    // construct the last chunk
    auto& outgoing_edges = g.at(make_pair(src_id, src_type));
    uint32_t n_outgoing_edges = outgoing_edges.size();
    int shingle_length = 2 * (n_outgoing_edges + 1);
    int last_chunk_length = shingle_length - chunk_length *
                            (shingle_length/chunk_length);
    if (last_chunk_length == 0)
      last_chunk_length = chunk_length;

    string last_chunk("x", last_chunk_length);
    int len = last_chunk_length, i = n_outgoing_edges - 1;
    do {
      last_chunk[--len] = get<1>(outgoing_edges[i]); // dst_type
      if (len <= 0)
        break;
      last_chunk[--len] = get<2>(outgoing_edges[i]); // edge_type
      i--;
    } while (len > 0 && i >= 0);
    if (i < 0) {
      if (len == 2) {
        last_chunk[--len] = src_type;
      }
      if (len == 1) {
        last_chunk[--len] = ' ';
      }
    }

    // construct the second last chunk if it exists
    string sec_last_chunk("x", chunk_length);
    if (i >= 0) {
      len = chunk_length;

      if (last_chunk_length % 2 != 0) {
        sec_last_chunk[--len] = get<2>(outgoing_edges[i]); // edge_type
        i--;
      }

      if (i >=0 && len >= 0) {
        do {
          sec_last_chunk[--len] = get<1>(outgoing_edges[i]);
          if (len <= 0)
            break;
          sec_last_chunk[--len] = get<2>(outgoing_edges[i]);
          i--;
        } while (len > 0 && i >= 0);
      }

      if (i < 0) {
        if (len == 2) {
          sec_last_chunk[--len] = src_type;
        }
        if (len == 1) {
          sec_last_chunk[--len] = ' ';
        }
      }
    }

  #ifdef DEBUG
    string shingle(" ", 1);
    shingle.reserve(2 * (n_outgoing_edges + 1));
    shingle.push_back(src_type);
    for (uint32_t i = 0; i < n_outgoing_edges; i++) {
      shingle.push_back(get<2>(outgoing_edges[i]));
      shingle.push_back(get<1>(outgoing_edges[i]));
    }

    cout << "Shingle: " << shingle << endl;
    vector<string> chunks = get_string_chunks(shingle, chunk_length);
    cout << "Last chunk: " << last_chunk << endl;
    assert(last_chunk == chunks[chunks.size() - 1]);
    if (chunks.size() > 1) {
      cout << "Second last chunk: " << sec_last_chunk << endl;
      assert(sec_last_chunk == chunks[chunks.size() - 2]);
    }
  #endif

    vector<string> incoming_chunks; // to be hashed and added
    vector<string> outgoing_chunks; // to be hashed and subtracted

    incoming_chunks.push_back(last_chunk);

    if (n_outgoing_edges > 1) { // this is not the first edge
      if (last_chunk_length == 1) {
        outgoing_chunks.push_back(sec_last_chunk.substr(0,
                                                        sec_last_chunk.length() - 1));
      } else if (last_chunk_length == 2) {
        // do nothing, only incoming chunk is the last chunk
      } else { // 2 < last_chunk_length <= chunk_length, last chunk had 2 chars added
        outgoing_chunks.push_back(last_chunk.substr(0, last_chunk_length - 2));
      }
    }

    // end = chrono::steady_clock::now(); // end shingle construction
    // shingle_construction_time =
      // chrono::duration_cast<chrono::microseconds>(end - start);

  #ifdef DEBUG
    cout << "Incoming chunks: ";
    for (auto& c : incoming_chunks) {
      cout << c << ",";
    }
    cout << endl;

    cout << "Outgoing chunks: ";
    for (auto& c : outgoing_chunks) {
      cout << c << ",";
    }
    cout << endl;
  #endif

    // record the change in the projection vector
    // this is used to update the centroid
    // vector<int> projection_delta(L, 0);
    // vector<double> projection_delta(L, 0);
    // double decayed_delta;

    // start = chrono::steady_clock::now(); // start sketch update

    // update the projection vectors
    for (auto& chunk : incoming_chunks) {
      for (uint32_t i = 0; i < L; i++) {
        // decayed_delta = projection[i];
        // cout << decayed_delta << " ";
        int delta = hashmulti(chunk, H[i]);
        projection[i] *= DECAYED_RATE;  //減衰させる
        projection[i] += delta;
        // cout << projection[i] << " " ;
        // decayed_delta = projection[i] - decayed_delta;
        // cout << decayed_delta << endl;
        // projection_delta[i] += decayed_delta;
      }
    }

    // このプログラムではoutは見ない
    // for (auto& chunk : outgoing_chunks) {
    //   for (uint32_t i = 0; i < L; i++) {
    //     int delta = hashmulti(chunk, H[i]);
    //     projection[i] -= delta;
    //     projection_delta[i] -= delta;
    //   }
    // }

    // update sketch = sign(projection)
    // for (uint32_t i = 0; i < L; i++) {
    //   sketch[i] = projection[i] >= 0 ? 1 : 0;
    // }
  // }
    // end = chrono::steady_clock::now(); // end sketch update
    // sketch_update_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // return make_tuple(projection_delta, shingle_construction_time, sketch_update_time);
  }
}
