from common.common_classes import Scenario, Executor

#################################
#                             1x3   2x3   1x10  2x10  3x10  3x30
# Scenario\Executor configs |  0  |  1  |  2  |  3  |  4  |  5  |
#          07           0   | 001 | 002 | 003 | 004 | 005 | 006 |
#          17           1   | 007 | 008 | 009 | 010 | 011 | 012 |
#          24           2   | 013 | 014 | 015 | 016 | 017 | 018 |
#         167           3   | 019 | 020 | 021 | 022 | 023 | 024 |
#
###########################


_real_executor_configs = {0: [{"executor_id": 1, "parallel_queues_count": 3}],
                          1: [{"executor_id": 1, "parallel_queues_count": 3},
                         {"executor_id": 2, "parallel_queues_count": 3}],
                          2:  [{"executor_id": 1, "parallel_queues_count": 10}],
                          3: [{"executor_id": 1, "parallel_queues_count": 10},
                         {"executor_id": 2, "parallel_queues_count": 10}],
                          4: [{"executor_id": 1, "parallel_queues_count": 10},
                         {"executor_id": 2, "parallel_queues_count": 10},
                         {"executor_id": 3, "parallel_queues_count": 10}],
                          5: [{"executor_id": 1, "parallel_queues_count": 30},
                         {"executor_id": 2, "parallel_queues_count": 30},
                         {"executor_id": 3, "parallel_queues_count": 30}]
                          }

_real_scenario_configs = {0: [{"name": "1", "configuration_times": [4 * 60 + 1]},
                              {"name": "2", "configuration_times": [9 * 60 + 40, 6 * 60 + 21, 8 * 60 + 28, 8 * 60 + 34]},
                              {"name": "3",
                          "configuration_times": [1 * 60 * 60 + 46 * 60 + 32, 1 * 60 * 60 + 47 * 60 + 1,
                                                  1 * 60 * 60 + 45 * 60 + 9,
                                                  1 * 60 * 60 + 45 * 60 + 31]},
                              {"name": "4",
                          "configuration_times": [32 * 60 + 47, 35 * 60 + 52, 1 * 60 * 60 + 12 * 60 + 20,
                                                  1 * 60 * 60 + 12 * 60 + 21]},
                              {"name": "5", "configuration_times": [14 * 60 + 10, 14 * 60 + 22]},
                              {"name": "6", "configuration_times": [6 * 60 + 50, 9 * 60 + 28]},
                              {"name": "7", "configuration_times": [11 * 60 + 11, 3 * 60 + 22]}
                              ],
                          1: [
                         {"name": "1", "configuration_times": [7 * 60 + 10]},
                         {"name": "2", "configuration_times": [18 * 60 + 38, 17 * 60 + 58]},
                         {"name": "3", "configuration_times": [14 * 60 + 12, 16 * 60 + 18, 11 * 60 + 3]},
                         {"name": "4", "configuration_times": [9 * 60 + 47, 10 * 60 + 1]},
                         {"name": "5", "configuration_times": [1 * 60 * 60 + 1 * 60 + 50, 44 * 60, 43 * 60 + 33]},
                         {"name": "6", "configuration_times": [7 * 60 + 33]},
                         {"name": "7", "configuration_times": [25 * 60 + 9, 20 * 60 + 15]},
                         {"name": "8", "configuration_times": [7 * 60 + 54, 9 * 60 + 27]},
                         {"name": "9", "configuration_times": [8 * 60 + 41, 9 * 60 + 48]},
                         {"name": "10", "configuration_times": [12 * 60 + 44, 8 * 60 + 55, 9 * 60 + 8]},
                         {"name": "11", "configuration_times": [15 * 60 + 20, 14 * 60 + 35]},
                         {"name": "12", "configuration_times": [23 * 60 + 52, 20 * 60 + 21, 21 * 60 + 8]},
                         {"name": "13", "configuration_times": [24 * 60 + 18]},
                         {"name": "14", "configuration_times": [23 * 60 + 28, 30 * 60 + 1, 15 * 60 + 19]},
                         {"name": "15", "configuration_times": [7 * 60 + 13, 7 * 60 + 20]},
                         {"name": "16",
                          "configuration_times": [2 * 60 * 60 + 31 * 60 + 46, 2 * 60 * 60 + 46 * 60 + 33]},
                         {"name": "17", "configuration_times": [5 * 60 + 10]},
                     ],
                          2: [
                         {"name": "1", "configuration_times": [4 * 60 + 1]},
                         {"name": "2", "configuration_times": [9 * 60 + 40, 6 * 60 + 21, 8 * 60 + 28, 8 * 60 + 34]},
                         {"name": "3",
                          "configuration_times": [1 * 60 * 60 + 46 * 60 + 32, 1 * 60 * 60 + 47 * 60 + 1,
                                                  1 * 60 * 60 + 45 * 60 + 9,
                                                  1 * 60 * 60 + 45 * 60 + 31]},
                         {"name": "4",
                          "configuration_times": [32 * 60 + 47, 35 * 60 + 52, 1 * 60 * 60 + 12 * 60 + 20,
                                                  1 * 60 * 60 + 12 * 60 + 21]},
                         {"name": "5", "configuration_times": [14 * 60 + 10, 14 * 60 + 22]},
                         {"name": "6", "configuration_times": [6 * 60 + 50, 9 * 60 + 28]},
                         {"name": "7", "configuration_times": [11 * 60 + 11, 3 * 60 + 22]},
                         {"name": "8", "configuration_times": [7 * 60 + 10]},
                         {"name": "9", "configuration_times": [18 * 60 + 38, 17 * 60 + 58]},
                         {"name": "10", "configuration_times": [14 * 60 + 12, 16 * 60 + 18, 11 * 60 + 3]},
                         {"name": "11", "configuration_times": [9 * 60 + 47, 10 * 60 + 1]},
                         {"name": "12", "configuration_times": [1 * 60 * 60 + 1 * 60 + 50, 44 * 60, 43 * 60 + 33]},
                         {"name": "13", "configuration_times": [7 * 60 + 33]},
                         {"name": "14", "configuration_times": [25 * 60 + 9, 20 * 60 + 15]},
                         {"name": "15", "configuration_times": [7 * 60 + 54, 9 * 60 + 27]},
                         {"name": "16", "configuration_times": [8 * 60 + 41, 9 * 60 + 48]},
                         {"name": "17", "configuration_times": [12 * 60 + 44, 8 * 60 + 55, 9 * 60 + 8]},
                         {"name": "18", "configuration_times": [15 * 60 + 20, 14 * 60 + 35]},
                         {"name": "19", "configuration_times": [23 * 60 + 52, 20 * 60 + 21, 21 * 60 + 8]},
                         {"name": "20", "configuration_times": [24 * 60 + 18]},
                         {"name": "21", "configuration_times": [23 * 60 + 28, 30 * 60 + 1, 15 * 60 + 19]},
                         {"name": "22", "configuration_times": [7 * 60 + 13, 7 * 60 + 20]},
                         {"name": "23",
                          "configuration_times": [2 * 60 * 60 + 31 * 60 + 46, 2 * 60 * 60 + 46 * 60 + 33]},
                         {"name": "24", "configuration_times": [5 * 60 + 10]}
                     ],
                          3: [
                         {"name": "1", "configuration_times": [4 * 60 + 1]},
                         {"name": "2", "configuration_times": [9 * 60 + 40, 6 * 60 + 21, 8 * 60 + 28, 8 * 60 + 34]},
                         {"name": "3",
                          "configuration_times": [1 * 60 * 60 + 46 * 60 + 32, 1 * 60 * 60 + 47 * 60 + 1,
                                                  1 * 60 * 60 + 45 * 60 + 9,
                                                  1 * 60 * 60 + 45 * 60 + 31]},
                         {"name": "4",
                          "configuration_times": [32 * 60 + 47, 35 * 60 + 52, 1 * 60 * 60 + 12 * 60 + 20,
                                                  1 * 60 * 60 + 12 * 60 + 21]},
                         {"name": "5", "configuration_times": [14 * 60 + 10, 14 * 60 + 22]},
                         {"name": "6", "configuration_times": [6 * 60 + 50, 9 * 60 + 28]},
                         {"name": "7", "configuration_times": [11 * 60 + 11, 3 * 60 + 22]},
                         {"name": "8", "configuration_times": [7 * 60 + 10]},
                         {"name": "9", "configuration_times": [18 * 60 + 38, 17 * 60 + 58]},
                         {"name": "10", "configuration_times": [14 * 60 + 12, 16 * 60 + 18, 11 * 60 + 3]},
                         {"name": "11", "configuration_times": [9 * 60 + 47, 10 * 60 + 1]},
                         {"name": "12", "configuration_times": [1 * 60 * 60 + 1 * 60 + 50, 44 * 60, 43 * 60 + 33]},
                         {"name": "13", "configuration_times": [7 * 60 + 33]},
                         {"name": "14", "configuration_times": [25 * 60 + 9, 20 * 60 + 15]},
                         {"name": "15", "configuration_times": [7 * 60 + 54, 9 * 60 + 27]},
                         {"name": "16", "configuration_times": [8 * 60 + 41, 9 * 60 + 48]},
                         {"name": "17", "configuration_times": [12 * 60 + 44, 8 * 60 + 55, 9 * 60 + 8]},
                         {"name": "18", "configuration_times": [15 * 60 + 20, 14 * 60 + 35]},
                         {"name": "19", "configuration_times": [23 * 60 + 52, 20 * 60 + 21, 21 * 60 + 8]},
                         {"name": "20", "configuration_times": [24 * 60 + 18]},
                         {"name": "21", "configuration_times": [23 * 60 + 28, 30 * 60 + 1, 15 * 60 + 19]},
                         {"name": "22", "configuration_times": [7 * 60 + 13, 7 * 60 + 20]},
                         {"name": "23",
                          "configuration_times": [2 * 60 * 60 + 31 * 60 + 46, 2 * 60 * 60 + 46 * 60 + 33]},
                         {"name": "24", "configuration_times": [5 * 60 + 10]},
                         {"name": "25", "configuration_times": [5 * 60 + 5]},
                         {"name": "26", "configuration_times": [8 * 60 + 5, 12 * 60 + 14]},
                         {"name": "27", "configuration_times": [8 * 60 + 47, 9 * 60 + 32]},
                         {"name": "28", "configuration_times": [2 * 60 + 43]},
                         {"name": "29", "configuration_times": [2 * 60 + 54]},
                         {"name": "30", "configuration_times": [6 * 60 + 31, 7 * 60 + 7, 7 * 60 + 2]},
                         {"name": "31", "configuration_times": [28 * 60 + 57, 35 * 60 + 33, 38 * 60 + 9]},
                         {"name": "32", "configuration_times": [2 * 60 + 51]},
                         {"name": "33", "configuration_times": [14 * 60 + 53, 15 * 60 + 10]},
                         {"name": "34", "configuration_times": [6 * 60 + 49]},
                         {"name": "35", "configuration_times": [3 * 60 + 31]},
                         {"name": "36",
                          "configuration_times": [5 * 60 + 52, 5 * 60 + 53, 8 * 60 + 5, 7 * 60 + 18, 7 * 60 + 12,
                                                  7 * 60 + 24, 7 * 60 + 1, 6 * 60 + 13, 6 * 60 + 54]},
                         {"name": "37", "configuration_times": [3 * 60 + 15]},
                         {"name": "38", "configuration_times": [7 * 60 + 42]},
                         {"name": "39", "configuration_times": [8 * 60 + 23]},
                         {"name": "40", "configuration_times": [7 * 60 + 10, 7 * 60 + 6, 8 * 60 + 12, 10 * 60 + 32]},
                         {"name": "41", "configuration_times": [6 * 60 + 46]},
                         {"name": "42", "configuration_times": [4 * 60 * 60 + 22 * 60 + 36]},
                         {"name": "43", "configuration_times": [5 * 60 + 4]},
                         {"name": "44", "configuration_times": [8 * 60 + 24, 8 * 60 + 19, 8 * 60 + 37, 9 * 60 + 41]},
                         {"name": "45", "configuration_times": [6 * 60 + 52]},
                         {"name": "46", "configuration_times": [8 * 60 + 52, 9 * 60 + 45]},
                         {"name": "47", "configuration_times": [8 * 60 + 12]},
                         {"name": "48", "configuration_times": [33 * 60 + 50]},
                         {"name": "49", "configuration_times": [10 * 60 + 51]},
                         {"name": "50", "configuration_times": [7 * 60 + 32]},
                         {"name": "51", "configuration_times": [29 * 60 + 49, 10 * 60 + 38, 12 * 60 + 42]},
                         {"name": "52", "configuration_times": [15 * 60 + 12]},
                         {"name": "53", "configuration_times": [14 * 60 + 7]},
                         {"name": "54", "configuration_times": [7 * 60 + 38, 9 * 60 + 41]},
                         {"name": "55", "configuration_times": [42 * 60 + 47]},
                         {"name": "56", "configuration_times": [1 * 60 * 60 + 14]},
                         {"name": "57", "configuration_times": [56 * 60 + 39]},
                         {"name": "58", "configuration_times": [54 * 60 + 39]},
                         {"name": "59", "configuration_times": [54 * 60 + 41]},
                         {"name": "60", "configuration_times": [1 * 60 * 60 + 8]},
                         {"name": "61", "configuration_times": [9 * 60 + 26]},
                         {"name": "62", "configuration_times": [3 * 60 + 1]},
                         {"name": "63", "configuration_times": [3 * 60 + 42]},
                         {"name": "64", "configuration_times": [4 * 60 + 48]},
                         {"name": "65", "configuration_times": [19 * 60 + 22]},
                         {"name": "66", "configuration_times": [6 * 60 + 19, 9 * 60 + 3]},
                         {"name": "67", "configuration_times": [13 * 60 + 52, 14 * 60 + 15]},
                         {"name": "68", "configuration_times": [7 * 60 + 33]},
                         {"name": "69", "configuration_times": [50 * 60 + 54, 25 * 60 + 47]},
                         {"name": "70",
                          "configuration_times": [14 * 60 + 44, 14 * 60 + 23, 20 * 60 + 3, 18 * 60 + 50, 32 * 60 + 11]},
                         {"name": "71",
                          "configuration_times": [8 * 60 + 42, 11 * 60 + 56, 10 * 60 + 1, 12 * 60 + 23, 7 * 60 + 53]},
                         {"name": "72", "configuration_times": [52 * 60 + 24]},
                         {"name": "73", "configuration_times": [6 * 60 + 3, 7 * 60 + 3]},
                         {"name": "74", "configuration_times": [6 * 60 + 27]},
                         {"name": "75", "configuration_times": [5 * 60 + 34]},
                         {"name": "76", "configuration_times": [18 * 60 + 22]},
                         {"name": "77", "configuration_times": [38 * 60 + 5, 39 * 60 + 33]},
                         {"name": "78", "configuration_times": [31 * 60 + 17]},
                         {"name": "79", "configuration_times": [1 * 60 * 60 + 41 * 60 + 38]},
                         {"name": "80", "configuration_times": [11 * 60 + 50]},
                         {"name": "81", "configuration_times": [22 * 60 + 27]},
                         {"name": "82", "configuration_times": [6 * 60 + 48, 8 * 60 + 29, 9 * 60 + 58]},
                         {"name": "83", "configuration_times": [9 * 60 + 26]},
                         {"name": "84", "configuration_times": [10 * 60 + 33]},
                         {"name": "85", "configuration_times": [9 * 60 + 56]},
                         {"name": "86", "configuration_times": [11 * 60]},
                         {"name": "87", "configuration_times": [13 * 60 + 19]},
                         {"name": "88", "configuration_times": [15 * 60 + 38]},
                         {"name": "89", "configuration_times": [12 * 60 + 28]},
                         {"name": "90", "configuration_times": [12 * 60 + 56]},
                         {"name": "91", "configuration_times": [8 * 60 + 15]},
                         {"name": "92", "configuration_times": [7 * 60 + 2]},
                         {"name": "93", "configuration_times": [5 * 60 + 7]},
                         {"name": "94", "configuration_times": [34 * 60 + 45]},
                         {"name": "95", "configuration_times": [32 * 60 + 57, 52 * 60 + 38]},
                         {"name": "96", "configuration_times": [1 * 60 * 60 + 12 * 60 + 37]},
                         {"name": "97", "configuration_times": [6 * 60 + 48]},
                         {"name": "98", "configuration_times": [10 * 60 + 51, 16 * 60 + 55]},
                         {"name": "99", "configuration_times": [15 * 60 + 54, 20 * 60 + 18]},
                         {"name": "100", "configuration_times": [6 * 60 + 14]},
                         {"name": "101", "configuration_times": [17 * 60, 21 * 60 + 28]},
                         {"name": "102", "configuration_times": [25 * 60 + 3, 9 * 60 + 30]},
                         {"name": "103", "configuration_times": [9 * 60 + 20, 11 * 60 + 4, 13 * 60 + 10]},
                         {"name": "104", "configuration_times": [14 * 60 + 45]},
                         {"name": "105", "configuration_times": [25 * 60 + 4, 33 * 60 + 56]},
                         {"name": "106",
                          "configuration_times": [33 * 60 + 8, 46 * 60 + 10, 35 * 60 + 24, 44 * 60 + 43]},
                         {"name": "107", "configuration_times": [5 * 60 + 3]},
                         {"name": "108", "configuration_times": [3 * 60 + 8]},
                         {"name": "109", "configuration_times": [4 * 60 + 54]},
                         {"name": "110", "configuration_times": [6 * 60 + 43]},
                         {"name": "111", "configuration_times": [2 * 60 + 40]},
                         {"name": "112", "configuration_times": [3 * 60 + 1]},
                         {"name": "113", "configuration_times": [3 * 60 + 47]},
                         {"name": "114", "configuration_times": [2 * 60 + 45]},
                         {"name": "115", "configuration_times": [3 * 60 + 2]},
                         {"name": "116", "configuration_times": [3 * 60 + 32]},
                         {"name": "117", "configuration_times": [2 * 60 + 41]},
                         {"name": "118", "configuration_times": [6 * 60 + 57, 7 * 60 + 36]},
                         {"name": "119", "configuration_times": [3 * 60]},
                         {"name": "120", "configuration_times": [7 * 60 + 3]},
                         {"name": "121", "configuration_times": [12 * 60 + 35]},
                         {"name": "122", "configuration_times": [16 * 60 + 52, 28 * 60 + 27, 24 * 60 + 53]},
                         {"name": "123", "configuration_times": [2 * 60 + 40]},
                         {"name": "124", "configuration_times": [9 * 60 + 44]},
                         {"name": "125", "configuration_times": [3 * 60 + 24]},
                         {"name": "126", "configuration_times": [13 * 60 + 7, 12 * 60 + 4]},
                         {"name": "127", "configuration_times": [3 * 60 + 48, 3 * 60 + 42]},
                         {"name": "128", "configuration_times": [3 * 60 + 22]},
                         {"name": "129", "configuration_times": [3 * 60 + 54]},
                         {"name": "130", "configuration_times": [3 * 60 + 34]},
                         {"name": "131", "configuration_times": [3 * 60 + 34]},
                         {"name": "132", "configuration_times": [2 * 60 + 37]},
                         {"name": "133", "configuration_times": [2 * 60 + 50]},
                         {"name": "134", "configuration_times": [3 * 60 + 5]},
                         {"name": "135", "configuration_times": [3 * 60 + 1]},
                         {"name": "136", "configuration_times": [2 * 60 + 35]},
                         {"name": "137", "configuration_times": [3 * 60 + 23]},
                         {"name": "138", "configuration_times": [2 * 60 + 37]},
                         {"name": "139", "configuration_times": [9 * 60 + 17]},
                         {"name": "140", "configuration_times": [2 * 60 + 47]},
                         {"name": "141", "configuration_times": [6 * 60 + 21]},
                         {"name": "142", "configuration_times": [40 * 60 + 10]},
                         {"name": "143", "configuration_times": [6 * 60 + 1]},
                         {"name": "144", "configuration_times": [3 * 60 + 3]},
                         {"name": "145", "configuration_times": [2 * 60 + 46]},
                         {"name": "146", "configuration_times": [2 * 60 + 44]},
                         {"name": "147", "configuration_times": [4 * 60 + 37]},
                         {"name": "148", "configuration_times": [2 * 60 + 38]},
                         {"name": "149", "configuration_times": [18 * 60 + 16]},
                         {"name": "150", "configuration_times": [3 * 60 + 38]},
                         {"name": "151", "configuration_times": [4 * 60 + 4]},
                         {"name": "152", "configuration_times": [10 * 60 + 27, 8 * 60 + 58]},
                         {"name": "153", "configuration_times": [8 * 60 + 46]},
                         {"name": "154", "configuration_times": [11 * 60 + 11]},
                         {"name": "155", "configuration_times": [3 * 60 + 29]},
                         {"name": "156", "configuration_times": [3 * 60 + 21]},
                         {"name": "157", "configuration_times": [3 * 60 + 25]},
                         {"name": "158", "configuration_times": [2 * 60 + 41]},
                         {"name": "159", "configuration_times": [2 * 60 + 50]},
                         {"name": "160", "configuration_times": [3 * 60 + 36]},
                         {"name": "161", "configuration_times": [8 * 60 + 2]},
                         {"name": "162", "configuration_times": [8 * 60 + 58]},
                         {"name": "163", "configuration_times": [11 * 60 + 25]},
                         {"name": "164", "configuration_times": [12 * 60 + 56, 15 * 60 + 12]},
                         {"name": "165", "configuration_times": [11 * 60 + 57]},
                         {"name": "166", "configuration_times": [3 * 60 + 11]},
                         {"name": "167", "configuration_times": [3 * 60 + 48]},
                     ]}

real_executor_configs_len = len(_real_executor_configs)
real_scenario_configs_len = len(_real_scenario_configs)


def real_case(env_class, executor_case=1, scenario_case=1):
    if not (executor_case in _real_executor_configs.keys() and scenario_case in _real_scenario_configs.keys()):
        raise ValueError("executor_case or scenario_case is not supported")
    executors = []
    scenarios = []

    for executor_config in _real_executor_configs[executor_case]:
        executors.append(Executor(executor_id=executor_config["executor_id"],
                                  parallel_queues_count=executor_config["parallel_queues_count"]))

    for scenario_config in _real_scenario_configs[scenario_case]:
        scenarios.append(
            Scenario(name=scenario_config["name"], configuration_times=scenario_config["configuration_times"]))

    env = env_class(scenarios=scenarios, executors=executors)
    if hasattr(env, 'observation_spec') and hasattr(env, 'action_spec'):
        if len(env.observation_spec().shape) == 0:
            observations = 1
        else:
            observations = env.observation_spec().shape[0]
        if len(env.action_spec().shape) == 0:
            actions = 1
        else:
            actions = env.action_spec().shape[0]
    else:
        observations = None
        actions = None

    return env, observations, actions