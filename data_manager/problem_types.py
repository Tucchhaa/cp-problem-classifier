#
# This file describes types used.
#

from dataclasses import dataclass
from typing import List, Optional, Literal, Union, Any, Dict


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
