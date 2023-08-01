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

#ifndef EXAMPLES_ANALYTICAL_APPS_PPR_PPR_AUTO_H_
#define EXAMPLES_ANALYTICAL_APPS_PPR_PPR_AUTO_H_

#include <grape/grape.h>

#include "ppr/ppr_auto_context.h"

#include <fstream>

using namespace std;

ofstream outfile;

namespace grape {
/**
 * @brief An implementation of PPR without using explicit message-passing
 * APIs, the version in LDBC, which can work on both directed and undirected
 * graphs.
 *
 * This is the auto-parallel version inherited from AutoAppBase. In this
 * version, users plug sequential algorithms for PEval and IncEval, and
 * libgrape-lite automatically parallelizes them in the distributed setting.
 * Users are not aware of messages.
 *
 *  @tparam FRAG_T
 */
template <typename FRAG_T>
class PPRAuto : public AutoAppBase<FRAG_T, PPRAutoContext<FRAG_T>>,
                     public Communicator {
 public:
  INSTALL_AUTO_WORKER(PPRAuto<FRAG_T>, PPRAutoContext<FRAG_T>, FRAG_T)
  using vertex_t = typename FRAG_T::vertex_t;

  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kBothOutIn;

  void PEval(const fragment_t& frag, context_t& ctx) {
    auto inner_vertices = frag.InnerVertices();

    outfile.open("result_ppr_auto.txt");

    //size_t graph_vnum = frag.GetTotalVerticesNum();

    outfile<<ctx.delta<<endl;
    ctx.step = 0;
    //ctx.delta = 0.85;
    //ctx.max_round = 100;

    // assign initial ranks
    for (auto& u : inner_vertices) {
      int EdgesNum = frag.GetOutgoingAdjList(u).Size();
      ctx.degree[u] = EdgesNum;
    }
    vertex_t source; bool native_source = frag.GetInnerVertex(ctx.source_id, source);
    if (native_source) {
        if(!ctx.degree[source]){ctx.results[source] = 1; ctx.max_round = 0; return;}
        ctx.results[source] = 1.0/ctx.degree[source];
        outfile<<frag.GetId(source)<<' '<<ctx.results[source]<<' '<< (frag.GetId(source) == ctx.source_id) <<endl;
    }
  }

  void IncEval(const fragment_t& frag, context_t& ctx) {
    //outfile<<"6"<<endl;
    auto inner_vertices = frag.InnerVertices();

    typename FRAG_T::template inner_vertex_array_t<double> next_results;
    next_results.Init(inner_vertices);

    //size_t graph_vnum = frag.GetTotalVerticesNum();

    ++ctx.step;
    if (ctx.step > ctx.max_round) {
      auto& degree = ctx.degree;
      auto& results = ctx.results;

      for (auto v : inner_vertices) {
        if (degree[v] != 0) {
          results[v] *= degree[v];
        }
      }
      outfile.close();
      return;
    }

    double base = 1.0 - ctx.delta;

    //outfile<<endl<<endl<<endl<<"Round "<<ctx.step<<"base="<<base<<endl;

    for (auto& u : inner_vertices) {
      // Get all incoming message's total
        double cur = 0;
        auto es = frag.GetIncomingAdjList(u);
        for (auto& e : es) {
          cur += ctx.results[e.get_neighbor()];
        }
        vertex_t source; bool native_source = frag.GetInnerVertex(ctx.source_id, source);
        if(native_source) next_results[u] = ( ctx.delta * cur + base * (frag.GetId(u) == ctx.source_id) ) / ctx.degree[u];
        else next_results[u] = ctx.delta * cur / ctx.degree[u];
        //outfile<<frag.GetId(u)<<' '<<cur<<' '<<next_results[u]<<endl;
      
    }

    for (auto u : inner_vertices) {
      ctx.results.SetValue(u, next_results[u]);
    }

  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PPR_PPR_AUTO_H_
