from sklearn.base import ClassifierMixin
from ...predictors.predictor import DL85Predictor, Cache_Type, Wipe_Type
import json


class DL85Classifier(DL85Predictor, ClassifierMixin):
    """
    An optimal binary decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=1
        Maximum depth of the tree to be found
    min_sup : int, default=1
        Minimum number of examples per leaf
    error_function : function, default=None
        Function used to evaluate the quality of each node. The function must take at least one argument, the list of instances covered by the node. It should return a float value representing the error of the node. In case of supervised learning, it should additionally return a label. If no error function is provided, the default one is used.
    fast_error_function : function, default=None
        Function used to evaluate the quality of each node. The function must take at least one argument, the list of number of instances per class in the node. It should return a float value representing the error of the node and the predicted label. If no error function is provided, the default one is used.
    max_error : int, default=0
        Maximum allowed error. Default value stands for no bound. If no tree can be found that is strictly better, the model remains empty.
    stop_after_better : bool, default=False
        A parameter used to indicate if the search will stop after finding a tree better than max_error
    time_limit : int, default=0
        Allocated time in second(s) for the search. Default value stands for no limit. The best tree found within the time limit is stored, if this tree is better than max_error.
    verbose : bool, default=False
        A parameter used to switch on/off the print of what happens during the search
    desc : function, default=None
        A parameter used to indicate heuristic function used to sort the items in descending order
    asc : function, default=None
        A parameter used to indicate heuristic function used to sort the items in ascending order
    repeat_sort : bool, default=False
        A parameter used to indicate whether the heuristic sort will be applied at each level of the lattice or only at the root
    quiet : bool, default=True
        A parameter used to indicate if the boosting log will be printed or not
    print_output : bool, default=False
        A parameter used to indicate if the search output will be printed or not
    cache_type : Cache_Type, default=Cache_Type.Cache_TrieItemset
        A parameter used to indicate the type of cache used when the `DL85Predictor.usecache` is set to True.
    maxcachesize : int, default=0
        A parameter used to indicate the maximum size of the cache. If the cache size is reached, the cache will be wiped using the `DL85Predictor.wipe_type` and `DL85Predictor.wipe_factor` parameters. Default value 0 stands for no limit.
    wipe_type : Wipe_Type, default=Wipe_Type.Reuses
        A parameter used to indicate the type of cache used when the `DL85Predictor.maxcachesize` is reached.
    wipe_factor : float, default=0.5
        A parameter used to indicate the rate of elements to delete from the cache when the `DL85Predictor.maxcachesize` is reached.
    use_cache : bool, default=True
        A parameter used to indicate if a cache will be used or not
    depth_two_special_algo : bool, default=True
        Define whether the special algo from depth-2 is used or not
    use_ub : bool, default=True
        Define whether the hierarchical upper bound is used or not
    similar_lb : bool, default=True
        Define whether the similarity lower bound is used or not
    dynamic_branch : bool, default=True
        Define whether a dynamic branching is used to decide in which order explore decisions on an attribute
    similar_for_branching : bool, default=True
        Define whether the similarity lower bound is involved in the dynamic branching or not

    Attributes
    ----------
    tree_ : str
        Outputted tree in serialized form; remains empty as long as no model is learned.
    base_tree_ : str
        Basic outputted tree without any additional data (transactions, proba, etc.)
    size_ : int
        The size of the outputted tree
    depth_ : int
        Depth of the found tree
    error_ : float
        Error of the found tree
    accuracy_ : float
        Accuracy of the found tree on training set
    lattice_size_ : int
        The number of nodes explored before found the optimal tree
    runtime_ : float
        Time of the optimal decision tree search
    timeout_ : bool
        Whether the search reached timeout or not
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    is_fitted_ : bool
        Whether the classifier is fitted or not
    """

    def __init__(
            self,
            max_depth=1,
            min_sup=1,
            error_function=None,
            fast_error_function=None,
            max_error=0,
            stop_after_better=False,
            time_limit=0,
            verbose=False,
            desc=False,
            asc=False,
            repeat_sort=False,
            k=0,
            quiet=True,
            print_output=False,
            cache_type=Cache_Type.Cache_TrieItemset,
            maxcachesize=0,
            wipe_type=Wipe_Type.Subnodes,
            wipe_factor=0.5,
            use_cache=True,
            depth_two_special_algo=True,
            use_ub=True,
            similar_lb=True,
            dynamic_branch=True,
            similar_for_branching=True):

        DL85Predictor.__init__(self,
                               max_depth=max_depth,
                               min_sup=min_sup,
                               error_function=error_function,
                               fast_error_function=fast_error_function,
                               max_error=max_error,
                               stop_after_better=stop_after_better,
                               time_limit=time_limit,
                               verbose=verbose,
                               desc=desc,
                               asc=asc,
                               repeat_sort=repeat_sort,
                               k=k,
                               leaf_value_function=None,
                               quiet=quiet,
                               print_output=print_output,
                               cache_type=cache_type,
                               maxcachesize=maxcachesize,
                               wipe_type=wipe_type,
                               wipe_factor=wipe_factor,
                               use_cache=use_cache,
                               depth_two_special_algo=depth_two_special_algo,
                               use_ub=use_ub,
                               similar_lb=similar_lb,
                               dynamic_branch=dynamic_branch,
                               similar_for_branching=similar_for_branching)

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is None:
            return DL85Predictor.fit(self, X, y)
        else:
            self.sample_weight = sample_weight
            return DL85Predictor.fit(self, X, y)
