#
# This file describes types used.
#

from dataclasses import dataclass
from typing import List, Optional, Literal, Any, Dict
from data_manager.spoj_scrapper.scrapper_types import Problem as SPOJScrapperProblem


@dataclass
class Example:
    input: str
    output: str
    explanation: Optional[str] = None


# === Codeforces ===

CodeforcesProblemLabel = Literal[
    '2-sat', 'binary search', 'bitmasks', 'brute force', 'chinese remainder theorem',
    'combinatorics', 'constructive algorithms', 'data structures', 'dfs and similar',
    'divide and conquer', 'dp', 'dsu', 'expression parsing', 'fft', 'flows', 'games',
    'geometry', 'graph matchings', 'graphs', 'greedy', 'hashing', 'implementation',
    'interactive', 'math', 'matrices', 'meet-in-the-middle', 'number theory', 'probabilities',
    'schedules', 'shortest paths', 'sortings', 'string suffix structures', 'strings',
    'ternary search', 'trees', 'two pointers', '*special'
]

@dataclass
class CodeforcesProblem:
    id: str
    title: str
    labels: List[CodeforcesProblemLabel]
    time_limit_per_test: float
    memory_limit_mb_per_test: float
    description: str
    input_format: str
    output_format: str
    interaction_format: str
    note: str
    examples: List[Example]
    tests: List[str]

@dataclass
class OpenR1CodeforcesProblem:
    id: str
    aliases: Optional[List[str]]
    contest_id: int
    contest_name: str
    contest_type: str
    contest_start: int
    contest_start_year: int
    index: str
    time_limit: float
    memory_limit: float
    title: str
    description: str
    input_format: str
    output_format: str
    interaction_format: Optional[str]
    note: str
    examples: List[Example]
    editorial: Optional[str]
    rating: int
    tags: List[CodeforcesProblemLabel]
    testset_size: int
    official_tests: List[Example]
    official_tests_complete: bool
    input_mode: str
    generated_checker: str
    executable: bool


# === Leetcode ===

LeetcodeProblemLabel = Literal[
    'Array', 'String', 'Hash Table', 'Dynamic Programming', 'Math', 'Sorting', 'Greedy',
    'Depth-First Search', 'Binary Search', 'Database', 'Matrix', 'Tree', 'Breadth-First Search',
    'Bit Manipulation', 'Two Pointers', 'Prefix Sum', 'Heap (Priority Queue)', 'Simulation',
    'Binary Tree', 'Stack', 'Graph', 'Counting', 'Sliding Window', 'Design', 'Enumeration',
    'Backtracking', 'Union Find', 'Linked List', 'Ordered Set', 'Number Theory', 'Monotonic Stack',
    'Segment Tree', 'Trie', 'Combinatorics', 'Bitmask', 'Queue', 'Recursion', 'Divide and Conquer',
    'Binary Indexed Tree', 'Memoization', 'Hash Function', 'Geometry', 'Binary Search Tree',
    'String Matching', 'Topological Sort', 'Shortest Path', 'Rolling Hash', 'Game Theory',
    'Interactive', 'Data Stream', 'Monotonic Queue', 'Brainteaser', 'Doubly-Linked List',
    'Randomized', 'Merge Sort', 'Counting Sort', 'Iterator', 'Concurrency', 'Probability and Statistics',
    'Quickselect', 'Suffix Array', 'Line Sweep', 'Bucket Sort', 'Minimum Spanning Tree', 'Shell',
    'Reservoir Sampling', 'Strongly Connected Component', 'Eulerian Circuit', 'Radix Sort',
    'Rejection Sampling', 'Biconnected Component'
]

@dataclass
class LeetcodeProblem:
    id: int
    title: str
    titleKebabCase: str
    labels: List[LeetcodeProblemLabel]
    description: str
    difficulty: Literal['Easy', 'Medium', 'Hard']

@dataclass
class KaysssLeetcodeProblem:
    questionFrontendId: int
    questionTitle: str
    TitleSlug: str
    content: str
    difficulty: Literal['Easy', 'Medium', 'Hard']
    totalAccepted: str
    totalSubmission: str
    totalAcceptedRaw: int
    totalSubmissionRaw: int
    acRate: str
    similarQuestions: List[str]
    mysqlSchemas: str
    category: Literal['Algorithms', 'Database', 'JavaScript']
    codeDefinitions: List[str]
    sampleTestCase: str
    metaData: List[Any]
    envInfo: str
    topicTags: List[LeetcodeProblemLabel]

# === SPOJ ===

SPOJProblemLabel = Literal[
    'binary-search', 'inversion', 'sliding-window-14', 'array', 'divide-and-conquer',
    'number-theory', 'fenwick-tree-11', 'graph', 'reeds-sloane', 'finite-field', 'xor-2',
    'stern-brocot-tree', 'factorisation', 'backtracking', 'datastructures', 'rmq-4',
    'totient', 'hessenberg-matrix', 'brute-force', 'scc', 'priority-queue', 'graph-theory',
    'numbertheory', 'strings-13', 'basic-7', 'heap', 'recursion-39', 'dp-10', 'fast-moebius-transform',
    'simple-math', 'bubble-cup-9-round-2', 'bostan-gaudry-schost-2', 'knapsack', 'nimbers',
    'algorithm-2', 'parametrized', 'adhoc', 'bubble-cup-7-round-1', 'math-6', 'bitwise',
    'set-cover', 'dirichlet-generating-function', 'digit-dp-1', 'matrixexpo', 'josephus-problem-1',
    'quadrangle-inequality', 'clique', 'bipartite-graph', 'disjoint-set-2', 'chess', 'palindrome-5',
    'maths-45', 'linear-algebra', 'linear-equations', 'game', 'matching', 'binets-formula',
    'bitmasks', 'dynamic-programming-141', 'math-number-theory', 'decomposition', 'min-cut',
    'eid2016', 'kmp-algorithm', 'lds', 'bubble-cup-6-round-1', 'parallel-search', 'basic ',
    'fast-prime-factorization', 'tutorial', 'gcd', 'convex-hull', 'greatest-common-divisor', 'sqrt-decomp',
    'counting-1', 'prefix-sum-1', 'assignment-problem', 'pattern-12', 'plane-geometry',
    'integer-programming', 'bit-3', 'newyear2017', 'matrix', 'bubble-cup-9-round-1', 'text-processing',
    'polynomial-multiplication', 'recursion', 'string-matching', 'kernel', 'constructive',
    'linearity-of-expectation', 'graph-25', 'optimization', 'map-8', 'hash-table', 'dirichlet-convolution',
    'randomized-algorithm', 'graph-116', 'minimum-cost-flow', 'lcs', 'board-game', 'bfs', 'dijkstra-s-algorithm',
    'bubble-cup-10-round-2', 'hashing', 'fermate', 'data-structures-19', 'lcm-4', 'lagrange-interpolation', 'dp-202',
    'berlekamp-massey', 'linked-list-1', 'ioitc', 'bubble-cup-8-round-1', 'inclusion-exclusion', 'karatsuba',
    'weighted-matching', 'shortest-path', 'tree-25', 'big-numbers', 'linear-recurrence', 'fast-prime-counting',
    'game-theory-1', 'rmq-1', 'dp-11', 'segment-tree-179', 'csh533', 'basic-datastructure', 'dynamic-programming',
    'stack', 'ancestor', 'branch-and-bound', 'totient-2', 'cycle-decomposition', 'pruning', 'number-theory-13',
    'dynamic-programming-72', 'gcd-3-1', 'bruteforce-4', 'segment-tree-386', 'numbe', 'math', 'primality-test', 'sieve',
    'kruskal-s-algorithm', 'z-1', 'euler-circuit', 'binary-tree', 'string-39', 'branchandbound', 'gcd-29', 'bash',
    'combinatorics', 'greedy', 'manhattan-distance', 'sorting', 'data-1', 'meet-in-the-middle-optimization',
    'cryptography', 'pointers', 'sub-linear', 'lcp', 'bubble-cup-7-round-2', 'max-flow', 'sequences', 'sub-quadratic',
    'adhoc-19', 'hill-climbing', 'hangzhou-2008', 'stirling', 'np-hard', 'simulations', 'twopointers', 'binarysearch-1',
    'data-structure-1', 'sweep', 'fast-fourier-transformation', 'segment-tree-1', 'sieve-of-eratosthenes', 'fft-2',
    'structures', 'generating-function', 'linear-programming', 'probability-theory', 'quicksort', 'tree', 'bubble-cup-6-round-2',
    'maths-2', 'prime-numbers-1', 'binarysearch-9', 'ad-hoc-1', 'precalculation', 'stable-marriage-problem', 'basics-2',
    'polynomial', 'shanghai-2009', 'queue', 'geometry', 'ds', 'segment-tree-73', 'short-coding', 'trie-1', 'sliding-window-1',
    'tree-14', 'lowest-common-ancestor', 'logic', 'mst', 'bubble-cup-8-round-2', 'modulo-1', 'carrom', 'nhspc2016',
    'fast-fourier-transofm', 'suffix-array-8', 'walsh-hadamard', 'algorithm', 'goc-s01e01', 'dfs', 'factorial',
]

@dataclass
class SPOJProblem(SPOJScrapperProblem):
    pass

# === Common problem type ===

ProblemLabel = Literal[
    'strings', 'data structures', 'dynamic programming', 'math', 'sorting', 'greedy',
    'graphs', 'binary search', 'matrices', 'bit manipulation', 'two pointers', 'trees',
    'union find', 'number theory', 'combinatorics', 'divide and conquer', 'hashing',
    'geometry', 'shortest path', 'game theory', 'interactive', 'probabilities'
]


@dataclass
class Problem:
    source: Literal['leetcode', 'codeforces']
    title: str
    description: str
    labels: List[ProblemLabel]


#
# * skip - means that there is no analog label
# * remove - means that we don't use problems with this label for training
#
# +-----------------------------------+----------------------------------+---------------------------+
# | Standard label                    | Leetcode                         | Codeforces                |
# +-----------------------------------+----------------------------------+---------------------------+
# | skip                              | Array                            |                           |
# | strings                           | String                           | strings                   |
# | data structures                   | Hash Table                       |                           |
# | dynamic programming               | Dynamic Programming              | dp                        |
# | math                              | Math                             | math                      |
# | sorting                           | Sorting                          | sortings                  |
# | greedy                            | Greedy                           | greedy                    |
# | graphs                            | Depth-First Search               | dfs and similar           |
# | binary search                     | Binary Search                    | binary search             |
# | remove                            | Database                         |                           |
# | matrices                          | Matrix                           | matrices                  |
# | trees                             | Tree                             | trees                     |
# | graphs                            | Breadth-First Search             |                           |
# | bit manipulation                  | Bit Manipulation                 | bitmasks                  |
# | two pointers                      | Two Pointers                     | two pointers              |
# | data structures                   | Prefix Sum                       |                           |
# | data structures                   | Heap (Priority Queue)            |                           |
# | skip                              | Simulation                       |                           |
# | trees                             | Binary Tree                      |                           |
# | data structures                   | Stack                            |                           |
# | graphs                            | Graph                            | graphs                    |
# | skip                              | Counting                         |                           |
# | skip                              | Sliding Window                   |                           |
# | remove                            | Design                           |                           |
# | skip                              | Enumeration                      |                           |
# | skip                              | Backtracking                     |                           |
# | union find                        | Union Find                       | dsu                       |
# | data structures                   | Linked List                      |                           |
# | data structures                   | Ordered Set                      |                           |
# | number theory                     | Number Theory                    | number theory             |
# | data structures                   | Monotonic Stack                  |                           |
# | trees                             | Segment Tree                     |                           |
# | trees                             | Trie                             |                           |
# | combinatorics                     | Combinatorics                    | combinatorics             |
# | bit manipulation                  | Bitmask                          |                           |
# | data structures                   | Queue                            |                           |
# | skip                              | Recursion                        |                           |
# | divide and conquer                | Divide and Conquer               | divide and conquer        |
# | trees                             | Binary Indexed Tree              |                           |
# | skip                              | Memoization                      |                           |
# | hashing                           | Hash Function                    | hashing                   |
# | geometry                          | Geometry                         | geometry                  |
# | trees, binary search              | Binary Search Tree               |                           |
# | strings                           | String Matching                  |                           |
# | sorting, graphs                   | Topological Sort                 |                           |
# | graphs, shortest path             | Shortest Path                    | shortest paths            |
# | hashing                           | Rolling Hash                     |                           |
# | game theory                       | Game Theory                      | games                     |
# | interactive                       | Interactive                      | interactive               |
# | skip                              | Data Stream                      |                           |
# | data structures                   | Monotonic Queue                  |                           |
# | skip                              | Brainteaser                      |                           |
# | data structures                   | Doubly-Linked List               |                           |
# | skip                              | Randomized                       |                           |
# | sorting                           | Merge Sort                       |                           |
# | sorting                           | Counting Sort                    |                           |
# | skip                              | Iterator                         |                           |
# | remove                            | Concurrency                      |                           |
# | probabilities                     | Probability and Statistics       | probabilities             |
# | skip                              | Quickselect                      |                           |
# | data structures                   | Suffix Array                     |                           |
# | skip                              | Line Sweep                       |                           |
# | sorting                           | Bucket Sort                      |                           |
# | graphs                            | Minimum Spanning Tree            |                           |
# | remove                            | Shell                            |                           |
# | skip                              | Reservoir Sampling               |                           |
# | graphs                            | Strongly Connected Component     |                           |
# | skip                              | Eulerian Circuit                 |                           |
# | sorting                           | Radix Sort                       |                           |
# | skip                              | Rejection Sampling               |                           |
# | skip                              | Biconnected Component            |                           |
# | data structures                   |                                  | data structures           |
# | strings                           |                                  | string suffix structures  |
# | graphs                            |                                  | graph matchings           |
# | skip                              |                                  | 2-sat                     |
# | skip                              |                                  | brute force               |
# | math                              |                                  | chinese remainder theorem |
# | skip                              |                                  | constructive algorithms   |
# | strings                           |                                  | expression parsing        |
# | math                              |                                  | fft                       |
# | graphs                            |                                  | flows                     |
# | skip                              |                                  | implementation            |
# | skip                              |                                  | meet-in-the-middle        |
# | skip                              |                                  | schedules                 |
# | binary search                     |                                  | ternary search            |
# | remove                            |                                  | *special                   |
# +------------------------+----------------------------------+--------------------------------------+
#

codeforces_to_standard: Dict[CodeforcesProblemLabel, List[ProblemLabel]] = {
    "strings": ["strings"],
    "dp": ["dynamic programming"],
    "math": ["math"],
    "sortings": ["sorting"],
    "greedy": ["greedy"],
    "dfs and similar": ["graphs"],
    "binary search": ["binary search"],
    "matrices": ["matrices"],
    "bitmasks": ["bit manipulation"],
    "two pointers": ["two pointers"],
    "graphs": ["graphs"],
    "dsu": ["union find"],
    "number theory": ["number theory"],
    "combinatorics": ["combinatorics"],
    "divide and conquer": ["divide and conquer"],
    "hashing": ["hashing"],
    "geometry": ["geometry"],
    "shortest paths": ["graphs", "shortest path"],
    "games": ["game theory"],
    "interactive": ["interactive"],
    "probabilities": ["probabilities"],
    "data structures": ["data structures"],
    "trees": ["trees"],
    "string suffix structures": ["strings"],
    "graph matchings": ["graphs"],
    "2-sat": [],
    "brute force": [],
    "chinese remainder theorem": ["math"],
    "constructive algorithms": [],
    "expression parsing": ["strings"],
    "fft": ["math"],
    "flows": ["graphs"],
    "implementation": [],
    "meet-in-the-middle": [],
    "schedules": [],
    "ternary search": ["binary search"],
    "*special": None,
}

# None - means that we must not include the problem to the dataset
leetcode_to_standard: Dict[LeetcodeProblemLabel, Optional[List[ProblemLabel]]] = {
    "Array": [],
    "String": ["strings"],
    "Hash Table": ["data structures"],
    "Dynamic Programming": ["dynamic programming"],
    "Math": ["math"],
    "Sorting": ["sorting"],
    "Greedy": ["greedy"],
    "Depth-First Search": ["graphs"],
    "Binary Search": ["binary search"],
    "Database": None,
    "Matrix": ["matrices"],
    "Tree": ["trees"],
    "Breadth-First Search": ["graphs"],
    "Bit Manipulation": ["bit manipulation"],
    "Two Pointers": ["two pointers"],
    "Prefix Sum": ["data structures"],
    "Heap (Priority Queue)": ["data structures"],
    "Simulation": [],
    "Binary Tree": ["trees"],
    "Stack": ["data structures"],
    "Graph": ["graphs"],
    "Counting": [],
    "Sliding Window": [],
    "Design": None,
    "Enumeration": [],
    "Backtracking": [],
    "Union Find": ["union find"],
    "Linked List": ["data structures"],
    "Ordered Set": ["data structures"],
    "Number Theory": ["number theory"],
    "Monotonic Stack": ["data structures"],
    "Segment Tree": ["trees"],
    "Trie": ["trees"],
    "Combinatorics": ["combinatorics"],
    "Bitmask": ["bit manipulation"],
    "Queue": ["data structures"],
    "Recursion": [],
    "Divide and Conquer": ["divide and conquer"],
    "Binary Indexed Tree": ["trees"],
    "Memoization": [],
    "Hash Function": ["hashing"],
    "Geometry": ["geometry"],
    "Binary Search Tree": ["trees", "binary search"],
    "String Matching": ["strings"],
    "Topological Sort": ["sorting", "graphs"],
    "Shortest Path": ["graphs", "shortest path"],
    "Rolling Hash": ["hashing"],
    "Game Theory": ["game theory"],
    "Interactive": ["interactive"],
    "Data Stream": [],
    "Monotonic Queue": ["data structures"],
    "Brainteaser": [],
    "Doubly-Linked List": ["data structures"],
    "Randomized": [],
    "Merge Sort": ["sorting"],
    "Counting Sort": ["sorting"],
    "Iterator": [],
    "Concurrency": None,
    "Probability and Statistics": ["probabilities"],
    "Quickselect": [],
    "Suffix Array": ["data structures"],
    "Line Sweep": [],
    "Bucket Sort": ["sorting"],
    "Minimum Spanning Tree": ["graphs"],
    "Shell": None,
    "Reservoir Sampling": [],
    "Strongly Connected Component": ["graphs"],
    "Eulerian Circuit": [],
    "Radix Sort": ["sorting"],
    "Rejection Sampling": [],
    "Biconnected Component": [],
}

spoj_to_standard: Dict[SPOJProblemLabel, List[ProblemLabel]] = {
    "binary-search": ["binary search"],
    "inversion": [],
    "sliding-window-14": [],
    "array": [],
    "divide-and-conquer": ["divide and conquer"],
    "number-theory": ["number theory"],
    "fenwick-tree-11": ["trees"],
    "graph": ["graphs"],
    "reeds-sloane": ["math"],
    "finite-field": ["math"],
    "xor-2": ["bit manipulation"],
    "stern-brocot-tree": ["number theory"],
    "factorisation": ["number theory"],
    "backtracking": [],
    "datastructures": ["data structures"],
    "rmq-4": ["data structures"],
    "totient": ["number theory"],
    "hessenberg-matrix": ["matrices"],
    "brute-force": [],
    "scc": ["graphs"],
    "priority-queue": ["data structures"],
    "graph-theory": ["graphs"],
    "numbertheory": ["number theory"],
    "strings-13": ["strings"],
    "basic-7": [],
    "heap": ["data structures"],
    "recursion-39": [],
    "dp-10": ["dynamic programming"],
    "fast-moebius-transform": ["math"],
    "simple-math": ["math"],
    "bubble-cup-9-round-2": [],
    "bostan-gaudry-schost-2": ["math"],
    "knapsack": ["dynamic programming"],
    "nimbers": ["game theory"],
    "algorithm-2": [],
    "parametrized": [],
    "adhoc": [],
    "bubble-cup-7-round-1": [],
    "math-6": ["math"],
    "bitwise": ["bit manipulation"],
    "set-cover": [],
    "dirichlet-generating-function": ["number theory"],
    "digit-dp-1": ["dynamic programming"],
    "matrixexpo": ["matrices"],
    "josephus-problem-1": ["math"],
    "quadrangle-inequality": ["math"],
    "clique": ["graphs"],
    "bipartite-graph": ["graphs"],
    "disjoint-set-2": ["union find"],
    "chess": ["game theory"],
    "palindrome-5": ["strings"],
    "maths-45": ["math"],
    "linear-algebra": ["matrices"],
    "linear-equations": ["matrices"],
    "game": ["game theory"],
    "matching": ["graphs"],
    "binets-formula": ["math"],
    "bitmasks": ["bit manipulation"],
    "dynamic-programming-141": ["dynamic programming"],
    "math-number-theory": ["math", "number theory"],
    "decomposition": [],
    "min-cut": ["graphs"],
    "eid2016": [],
    "kmp-algorithm": ["strings"],
    "lds": ["dynamic programming"],
    "bubble-cup-6-round-1": [],
    "parallel-search": [],
    "basic ": [],
    "fast-prime-factorization": ["number theory"],
    "tutorial": [],
    "gcd": ["number theory"],
    "convex-hull": ["geometry"],
    "greatest-common-divisor": ["number theory"],
    "sqrt-decomp": ["data structures"],
    "counting-1": ["combinatorics"],
    "prefix-sum-1": ["data structures"],
    "assignment-problem": ["graphs"],
    "pattern-12": ["strings"],
    "plane-geometry": ["geometry"],
    "integer-programming": [],
    "bit-3": ["bit manipulation"],
    "newyear2017": [],
    "matrix": ["matrices"],
    "bubble-cup-9-round-1": [],
    "text-processing": ["strings"],
    "polynomial-multiplication": ["math"],
    "recursion": [],
    "string-matching": ["strings"],
    "kernel": [],
    "constructive": [],
    "linearity-of-expectation": ["probabilities"],
    "graph-25": ["graphs"],
    "optimization": [],
    "map-8": ["data structures"],
    "hash-table": ["hashing"],
    "dirichlet-convolution": ["number theory"],
    "randomized-algorithm": [],
    "graph-116": ["graphs"],
    "minimum-cost-flow": ["graphs"],
    "lcs": ["strings"],
    "board-game": ["game theory"],
    "bfs": ["graphs"],
    "dijkstra-s-algorithm": ["graphs", "shortest path"],
    "bubble-cup-10-round-2": [],
    "hashing": ["hashing"],
    "fermate": ["number theory"],
    "data-structures-19": ["data structures"],
    "lcm-4": ["number theory"],
    "lagrange-interpolation": ["math"],
    "dp-202": ["dynamic programming"],
    "berlekamp-massey": ["math"],
    "linked-list-1": ["data structures"],
    "ioitc": [],
    "bubble-cup-8-round-1": [],
    "inclusion-exclusion": ["combinatorics"],
    "karatsuba": ["math"],
    "weighted-matching": ["graphs"],
    "shortest-path": ["graphs", "shortest path"],
    "tree-25": ["trees"],
    "big-numbers": ["math"],
    "linear-recurrence": ["math"],
    "fast-prime-counting": ["number theory"],
    "game-theory-1": ["game theory"],
    "rmq-1": ["data structures"],
    "dp-11": ["dynamic programming"],
    "segment-tree-179": ["trees"],
    "csh533": [],
    "basic-datastructure": ["data structures"],
    "dynamic-programming": ["dynamic programming"],
    "stack": ["data structures"],
    "ancestor": ["trees"],
    "branch-and-bound": [],
    "totient-2": ["number theory"],
    "cycle-decomposition": ["graphs"],
    "pruning": [],
    "number-theory-13": ["number theory"],
    "dynamic-programming-72": ["dynamic programming"],
    "gcd-3-1": ["number theory"],
    "bruteforce-4": [],
    "segment-tree-386": ["trees"],
    "numbe": [],
    "math": ["math"],
    "primality-test": ["number theory"],
    "sieve": ["number theory"],
    "kruskal-s-algorithm": ["graphs"],
    "z-1": ["strings"],
    "euler-circuit": ["graphs"],
    "binary-tree": ["trees"],
    "string-39": ["strings"],
    "branchandbound": [],
    "gcd-29": ["number theory"],
    "bash": [],
    "combinatorics": ["combinatorics"],
    "greedy": ["greedy"],
    "manhattan-distance": ["geometry"],
    "sorting": ["sorting"],
    "data-1": [],
    "meet-in-the-middle-optimization": [],
    "cryptography": ["number theory"],
    "pointers": ["data structures"],
    "sub-linear": [],
    "lcp": ["strings"],
    "bubble-cup-7-round-2": [],
    "max-flow": ["graphs"],
    "sequences": [],
    "sub-quadratic": [],
    "adhoc-19": [],
    "hill-climbing": [],
    "hangzhou-2008": [],
    "stirling": ["combinatorics"],
    "np-hard": [],
    "simulations": [],
    "twopointers": ["two pointers"],
    "binarysearch-1": ["binary search"],
    "data-structure-1": ["data structures"],
    "sweep": ["geometry"],
    "fast-fourier-transformation": ["math"],
    "segment-tree-1": ["data structures"],
    "sieve-of-eratosthenes": ["number theory"],
    "fft-2": ["math"],
    "structures": ["data structures"],
    "generating-function": [],
    "linear-programming": [],
    "probability-theory": ["probabilities"],
    "quicksort": ["sorting"],
    "tree": ["trees"],
    "bubble-cup-6-round-2": [],
    "maths-2": ["math"],
    "prime-numbers-1": ["number theory"],
    "binarysearch-9": ["binary search"],
    "ad-hoc-1": [],
    "precalculation": [],
    "stable-marriage-problem": ["graphs"],
    "basics-2": [],
    "polynomial": ["math"],
    "shanghai-2009": [],
    "queue": ["data structures"],
    "geometry": ["geometry"],
    "ds": ["data structures"],
    "segment-tree-73": ["data structures"],
    "short-coding": [],
    "trie-1": ["trees"],
    "sliding-window-1": ["two pointers"],
    "tree-14": ["trees"],
    "lowest-common-ancestor": ["trees"],
    "logic": [],
    "mst": ["graphs"],
    "bubble-cup-8-round-2": [],
    "modulo-1": ["number theory"],
    "carrom": ["game theory"],
    "nhspc2016": [],
    "fast-fourier-transofm": ["math"],
    "suffix-array-8": ["strings"],
    "walsh-hadamard": ["math"],
    "algorithm": [],
    "goc-s01e01": [],
    "dfs": ["graphs"],
    "factorial": ["math"],
}
