import nifty
import nifty.graph.opt.multicut as nifty_mc
import numpy as np
import threading

def set_costs_from_uv_ids(graph, costs, uv_pairs, values):
    edge_ids        = graph.findEdges(uv_pairs)
    valid_edges     = edge_ids != -1
    edge_ids        = edge_ids[valid_edges]
    # print('any valid edges', valid_edges, np.any(valid_edges))

    costs[edge_ids] = values[valid_edges] if isinstance(values, (np.ndarray, list, tuple)) else values

def solve_multicut(graph, costs):
    assert graph.numberOfEdges == len(costs)
    objective = nifty_mc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=True).create(objective)
    return solver.optimize()


class AgglomerationModelCache(object):
    
    def __init__(self):
        super(AgglomerationModelCache, self).__init__()

        self.graph    = None
        self.solution = None
        self.edges    = None
        self.lock     = threading.RLock()

    def update_graph(self, edges):
        max_id = edges.max().item()
        graph  = nifty.graph.UndirectedGraph(max_id + 1)
        graph.insertEdges(edges)
        with self.lock:
            self.graph    = graph
            self.solution = None
            self.edges    = edges

    def optimize(self, weights):

        with self.lock:

            if self.graph is None or self.edges is None:
                return

            assert self.edges.shape[0] == weights.shape[0]

            graph = self.graph

        costs    = self._map_weights(weights)
        solution = solve_multicut(graph, costs)

        with self.lock:
            self.solution = solution

        return solution

    def get_solution(self):
        with self.lock:
            return self.solution


    def _map_weights(self, probabilities):

        # scale the probabilities to avoid diverging costs
        # and transform to costs
        p_min = 0.001
        p_max = 1. - p_min
        probabilities = (p_max - p_min) * probabilities + p_min
        costs = np.log((1. - probabilities) / probabilities)

        # weight by edge size
        # if edge_sizes is not None:
        #     assert edge_sizes.shape == costs.shape
        #     weight = edge_sizes / edge_sizes.max()
        #     costs = weight * costs

        return costs



