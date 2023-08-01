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
#ifndef EXAMPLES_ANALYTICAL_APPS_PPR_PPR_H_
#define EXAMPLES_ANALYTICAL_APPS_PPR_PPR_H_

#include <grape/grape.h>

#include "ppr/ppr_context.h"

#include <fstream>

using namespace std;

//ofstream outfile;


namespace grape {

/**
 * @brief An implementation of PPR, which can work
 * on undirected graphs.
 *
 * This version of PPR inherits BatchShuffleAppBase.
 * Messages are generated in batches and received in-place.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PPR : public BatchShuffleAppBase<FRAG_T, PPRContext<FRAG_T>>,
                 public ParallelEngine,
                 public Communicator {
 public:
  INSTALL_BATCH_SHUFFLE_WORKER(PPR<FRAG_T>, PPRContext<FRAG_T>,
                               FRAG_T)

  using vertex_t = typename FRAG_T::vertex_t;
  using vid_t = typename FRAG_T::vid_t;

  static constexpr bool need_split_edges = true;
  static constexpr bool need_split_edges_by_fragment = true;
  static constexpr MessageStrategy message_strategy =
      MessageStrategy::kAlongOutgoingEdgeToOuterVertex;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kOnlyOut;

  PPR() = default;

  

  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    if (ctx.max_round <= 0) {
      return;
    }

    //outfile.open("result_ppr.txt");

    auto inner_vertices = frag.InnerVertices();

#ifdef PROFILING
    ctx.exec_time -= GetCurrentTime();
#endif


    ctx.step = 0;
    ctx.graph_vnum = frag.GetTotalVerticesNum();
    vid_t dangling_vnum = 0;
    double p = 1.0 / ctx.graph_vnum;
    ctx.delta = 0.85;//
    ctx.max_round = 1000;//

    vertex_t source;
    bool native_source = frag.GetInnerVertex(ctx.source_id, source);
    if (native_source) 
    {
      ctx.result[source] = 1/frag.GetLocalOutDegree(source);
      //outfile<<"qwq:"<<frag.GetLocalOutDegree(source)<<endl;
    }
    //outfile<<"?"<<native_source<<endl;
    std::vector<vid_t> dangling_vnum_tid(thread_num(), 0);
    ForEach(inner_vertices,
            [&ctx, &frag, p, &dangling_vnum_tid](int tid, vertex_t u) {
              int EdgeNum = frag.GetLocalOutDegree(u);
              ctx.degree[u] = EdgeNum;
              if (EdgeNum == 0) ++dangling_vnum_tid[tid];
              
            });

    for (auto vn : dangling_vnum_tid) {
      dangling_vnum += vn;
    }

    Sum(dangling_vnum, ctx.total_dangling_vnum);
    ctx.dangling_sum = p * ctx.total_dangling_vnum;

#ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
#endif

    messages.SyncInnerVertices<fragment_t, double>(frag, ctx.result,
                                                   thread_num());
#ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
#endif
  }

  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    ++ctx.step;

    double base = (1.0 - ctx.delta) / ctx.graph_vnum +
                  ctx.delta * ctx.dangling_sum / ctx.graph_vnum;
    ctx.dangling_sum = base * ctx.total_dangling_vnum;

    //outfile<<"base="<<base<<" dangling_sum="<<ctx.dangling_sum<<" delta="<<ctx.delta<<endl;

#ifdef PROFILING
    ctx.preprocess_time -= GetCurrentTime();
#endif
    messages.UpdateOuterVertices();
#ifdef PROFILING
    ctx.preprocess_time += GetCurrentTime();
    ctx.exec_time -= GetCurrentTime();
#endif

    //outfile<<endl<<endl<<ctx.dangling_sum<<endl;
    ForEach(inner_vertices, [&ctx, &frag, base](int tid, vertex_t u) {
      double cur = 0;
      auto es = frag.GetOutgoingAdjList(u);
      /*if(frag.GetId(u) == 3)*/ //outfile<<endl<<endl<<frag.GetId(u)<<' '<<frag.GetLocalOutDegree(u)<<ctx.result[u]<<" eeaa ";
      for (auto& e : es) {
        cur += ctx.result[e.get_neighbor()];
        /*if(frag.GetId(u) == 3)*/ //outfile<<frag.GetId(e.get_neighbor())<<' ';
      }
      vertex_t source; frag.GetInnerVertex(ctx.source_id, source);
      int en = frag.GetLocalOutDegree(u);
      int t = (u == source);
      ctx.next_result[u] = en > 0 ? (ctx.delta * cur + base * t * ctx.graph_vnum) / en : base;
      });
#ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
#endif

    ctx.result.Swap(ctx.next_result);

    if (ctx.step != ctx.max_round) {
#ifdef PROFILING
      ctx.postprocess_time -= GetCurrentTime();
#endif
      messages.SyncInnerVertices<fragment_t, double>(frag, ctx.result,
                                                     thread_num());
#ifdef PROFILING
      ctx.postprocess_time += GetCurrentTime();
#endif
    } else {
      auto& degree = ctx.degree;
      auto& result = ctx.result;

      for (auto v : inner_vertices) {
        if (degree[v] != 0) {
          result[v] *= degree[v];
        }
      }
      //outfile.close();
      return;
    }
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PAGERANK_PAGERANK_H_
