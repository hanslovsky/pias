import logging
import nifty.graph.opt.multicut as nifty_mc
import numpy as np

_logger = logging.getLogger(__name__)


def set_costs_from_uv_ids(graph, costs, uv_pairs, values):
    edge_ids        = graph.findEdges(uv_pairs)
    valid_edges     = edge_ids != -1
    edge_ids        = edge_ids[valid_edges]
    # print('any valid edges', valid_edges, np.any(valid_edges))

    costs[edge_ids] = values[valid_edges] if isinstance(values, (np.ndarray, list, tuple)) else values

def solve_multicut(graph, costs):
    assert graph.numberOfEdges == len(costs)
    _logger.debug('Creating multi-cut object from graph %s and costs %s (%s)', graph, costs.shape, costs)
    # TODO why do lower costs mean cut the edge?
    # Qutoing @constantinpape
    # the cost can be in ]-inf, inf[ (I usually clip at ~ ]-6, 6[),
    # where negative costs are repulsive (i.e. nodes are more likely to be disconnected)
    # and positive costs are attractive
    objective = nifty_mc.multicutObjective(graph, costs)
    solver = objective.kernighanLinFactory(warmStartGreedy=True).create(objective)
    return solver.optimize()


def _default_map_weights(probabilities):
    '''

    :param probabilities: probability for edge being active, i.e. the two corresponding nodes are in same connected component
    :return:
    '''
    # scale the probabilities to avoid diverging costs
    # and transform to costs
    _logger.debug('Mapping probabilities %s (%s)', probabilities.shape, probabilities)
    p_min = 0.001
    p_max = 1. - p_min
    probabilities = np.clip(probabilities, a_min=p_min, a_max=p_max)
    costs = np.log((1. - probabilities) / probabilities)
    _logger.debug('Mapped probabilities %s (%s)', costs.shape, costs)

    # weight by edge size
    # if edge_sizes is not None:
    #     assert edge_sizes.shape == costs.shape
    #     weight = edge_sizes / edge_sizes.max()
    #     costs = weight * costs

    return costs


class MulticutAgglomeration(object):

    def __init__(self, map_weights=_default_map_weights):
        super(MulticutAgglomeration, self).__init__()
        self.map_weights = map_weights

    def optimize(self, graph, weights, known_labels=None):
        '''

        :param graph:
        :param weights:
        :param tuple known_labels: 0: edge is inactive, 1: edge is active (nodes are in same connected component)
        :return:
        '''


        if graph is None or weights is None:
            return

        costs = self.map_weights(weights)
        if known_labels is not None:
            _logger.debug('Known labels are %s', known_labels)
            known_labels_costs = -1e4 * (2 * np.asarray(known_labels[1], dtype=np.float64) - 1)
            costs[known_labels[0]] = known_labels_costs
        _logger.debug('Optimizing multi-cut with graph %s and weights %s (%s)', graph, costs.shape, costs)
        # Qutoing @constantinpape
        # the cost can be in ]-inf, inf[ (I usually clip at ~ ]-6, 6[),
        # where negative costs are repulsive (i.e. nodes are more likely to be disconnected)
        # and positive costs are attractive
        solution = solve_multicut(graph, -costs)
        return solution



