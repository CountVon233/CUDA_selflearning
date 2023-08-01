/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef EXAMPLES_ANALYTICAL_APPS_PPR_PPR_AUTO_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_PPR_PPR_AUTO_CONTEXT_H_

#include <iomanip>

#include <iostream>

#include <grape/grape.h>

namespace grape {
/**
 * @brief Context for the auto-parallel version of PPR.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PPRAutoContext : public VertexDataContext<FRAG_T, double> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit PPRAutoContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, double>(fragment, true),
        results(this->data()) {}

  void Init(AutoParallelMessageManager<FRAG_T>& messages, double d,
            int max_round, oid_t source_id) {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();
    auto vertices = frag.Vertices();
    std::cout<<d<<std::endl;
    this->delta = d;
    this->max_round = max_round;
    this->source_id = source_id;
    degree.Init(inner_vertices, 0);
    results.Init(vertices, 0.0, [](double* lhs, double rhs) {
      *lhs = rhs;
      return true;
    });

    messages.RegisterSyncBuffer(
        frag, &results, MessageStrategy::kAlongOutgoingEdgeToOuterVertex);
    step = 0;
  }

  void Output(std::ostream& os) override {
    auto& frag = this->fragment();
    auto inner_vertices = frag.InnerVertices();
    
    /*for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << degree[v] << std::endl << "OUT:";
      auto es = frag.GetOutgoingAdjList(v);
      for (auto& e : es) {
        os << frag.GetId(e.get_neighbor())<<' ';
      }
      os << std::endl << "In:";
      es = frag.GetIncomingAdjList(v);
      for (auto& e : es) {
        os << frag.GetId(e.get_neighbor())<<' ';
      }
      os << std::endl << std::endl;
    }*/
    
    for (auto v : inner_vertices) {
      os << frag.GetId(v) << " " << std::scientific << std::setprecision(15)
         << results[v] << std::endl;
    }
  }

  typename FRAG_T::template inner_vertex_array_t<int> degree;
  SyncBuffer<typename FRAG_T::vertices_t, double> results;
  int step = 0;
  int max_round = 0;
  double delta = 0.85;
  double dangling_sum = 0;
  oid_t source_id = 1;
};
}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PPR_PPR_AUTO_CONTEXT_H_