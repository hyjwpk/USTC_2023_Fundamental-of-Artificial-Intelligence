#include <fstream>
#include <iostream>
#include <queue>
#include <time.h>
#include <unordered_map>
#include <vector>
using namespace std;

#define MAX_N 12

class Operate {
public:
    int i;
    int j;
    int s;
    Operate(int i, int j, int s) : i(i), j(j), s(s) {}
};

class Node {
public:
    float f;
    vector<Operate> path;
    bool map[MAX_N][MAX_N];
    bool operator>(const Node &a) const {
        // select the node with the minimum f value
        if (f == a.f) {
            // select the node with the maximum path length
            return path.size() < a.path.size();
        } else {
            return f > a.f;
        }
    }
};

int N;
bool map[MAX_N][MAX_N];
priority_queue<Node, vector<Node>, greater<Node>> open;
unordered_map<string, float> visited;
int free_count;

// apply the operation to the map
void apply_path(bool cur_map[][MAX_N], int i, int j, int s) {
    switch (s) {
    case 1:
        cur_map[i][j] = !cur_map[i][j];
        cur_map[i][j + 1] = !cur_map[i][j + 1];
        cur_map[i - 1][j] = !cur_map[i - 1][j];
        break;
    case 2:
        cur_map[i][j] = !cur_map[i][j];
        cur_map[i - 1][j] = !cur_map[i - 1][j];
        cur_map[i][j - 1] = !cur_map[i][j - 1];
        break;
    case 3:
        cur_map[i][j] = !cur_map[i][j];
        cur_map[i][j - 1] = !cur_map[i][j - 1];
        cur_map[i + 1][j] = !cur_map[i + 1][j];
        break;
    case 4:
        cur_map[i][j] = !cur_map[i][j];
        cur_map[i + 1][j] = !cur_map[i + 1][j];
        cur_map[i][j + 1] = !cur_map[i][j + 1];
        break;
    default:
        break;
    }
}

float heuristic1(bool cur_map[][MAX_N]) {
    float lock = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (cur_map[i][j]) {
                lock++;
            }
        }
    }
    return lock;
}

float heuristic2(bool cur_map[][MAX_N]) {
    return heuristic1(cur_map) / 3;
}

float heuristic3(bool cur_map[][MAX_N]) {
    bool new_map[MAX_N][MAX_N];
    memcpy(new_map, cur_map, sizeof(new_map));
    float lock = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (new_map[i][j]) {
                int count = 1;
                if (i + 1 < N) {
                    count += new_map[i + 1][j];
                    new_map[i + 1][j] = 0;
                }
                if (j + 1 < N) {
                    count += new_map[i][j + 1];
                    new_map[i][j + 1] = 0;
                }
                if (i + 1 < N && j + 1 < N) {
                    count += new_map[i + 1][j + 1];
                    new_map[i + 1][j + 1] = 0;
                }
                if (count == 1) {
                    lock += 3;
                } else if (count == 2) {
                    lock += 2;
                } else if (count == 3) {
                    lock += 1;
                } else if (count == 4) {
                    lock += 4;
                }
            }
        }
    }
    return lock;
}

float heuristic4(bool cur_map[][MAX_N]) {
    float lock = 0;
    int type1 = 0;
    int type2 = 0;
    int type3 = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (cur_map[i][j]) {
                int type = 0;
                if (i - 1 >= 0 && j - 1 >= 0 && cur_map[i - 1][j - 1] && (cur_map[i][j - 1] || cur_map[i - 1][j])) {
                    type = 3;
                } else if (i - 1 >= 0 && j + 1 < N && cur_map[i - 1][j + 1] && (cur_map[i][j + 1] || cur_map[i - 1][j])) {
                    type = 3;
                } else if (i + 1 < N && j - 1 >= 0 && cur_map[i + 1][j - 1] && (cur_map[i][j - 1] || cur_map[i + 1][j])) {
                    type = 3;
                } else if (i + 1 < N && j + 1 < N && cur_map[i + 1][j + 1] && (cur_map[i][j + 1] || cur_map[i + 1][j])) {
                    type = 3;
                } else if (i - 1 >= 0 && j - 1 >= 0 && cur_map[i][j - 1] && cur_map[i - 1][j]) {
                    type = 3;
                } else if (i - 1 >= 0 && j + 1 < N && cur_map[i][j + 1] && cur_map[i - 1][j]) {
                    type = 3;
                } else if (i + 1 < N && j - 1 >= 0 && cur_map[i][j - 1] && cur_map[i + 1][j]) {
                    type = 3;
                } else if (i + 1 < N && j + 1 < N && cur_map[i][j + 1] && cur_map[i + 1][j]) {
                    type = 3;
                } else if ((i - 1 >= 0 && j - 1 >= 0 && cur_map[i - 1][j - 1]) ||
                           (i - 1 >= 0 && j + 1 < N && cur_map[i - 1][j + 1]) ||
                           (i + 1 < N && j - 1 >= 0 && cur_map[i + 1][j - 1]) ||
                           (i + 1 < N && j + 1 < N && cur_map[i + 1][j + 1]) ||
                           (j - 1 >= 0 && cur_map[i][j - 1]) ||
                           (j + 1 < N && cur_map[i][j + 1]) ||
                           (i - 1 >= 0 && cur_map[i - 1][j]) ||
                           (i + 1 < N && cur_map[i + 1][j])) {
                    type = 2;
                } else {
                    type = 1;
                }
                if (type == 1) {
                    type1++;
                } else if (type == 2) {
                    type2++;
                } else if (type == 3) {
                    type3++;
                }
            }
        }
    }
    lock += type3 / 3;
    type2 += type3 % 3;
    lock += type2 / 2;
    type1 += type2 % 2;
    lock += type1;
    return lock;
}

// check if the operation is available
bool available(bool cur_map[][MAX_N], int i, int j, int s) {
    switch (s) {
    case 1:
        if (i - 1 < 0 || j + 1 >= N) {
            return false;
        }
        if (!cur_map[i][j] && !cur_map[i][j + 1] && !cur_map[i - 1][j]) {
            return false;
        }
        break;
    case 2:
        if (i - 1 < 0 || j - 1 < 0) {
            return false;
        }
        if (!cur_map[i][j] && !cur_map[i - 1][j] && !cur_map[i][j - 1]) {
            return false;
        }
        break;
    case 3:
        if (i + 1 >= N || j - 1 < 0) {
            return false;
        }
        if (!cur_map[i][j] && !cur_map[i][j - 1] && !cur_map[i + 1][j]) {
            return false;
        }
        break;
    case 4:
        if (i + 1 >= N || j + 1 >= N) {
            return false;
        }
        if (!cur_map[i][j] && !cur_map[i + 1][j] && !cur_map[i][j + 1]) {
            return false;
        }
        break;
    default:
        break;
    }
    return true;
}

bool check_visited(bool cur_map[][MAX_N], float f) {
    string s = "";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            s += cur_map[i][j] + '0';
        }
    }
    if (visited.find(s) == visited.end()) {
        return false;
    } else if (visited[s] > f) {
        return false;
    }
    return true;
}

bool check_visited_symmetry(bool cur_map[][MAX_N], float f) {
    if (check_visited(cur_map, f)) {
        return true;
    }
    // bool new_map[MAX_N][MAX_N];
    // // r90
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         new_map[j][N - 1 - i] = cur_map[i][j];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    // // r180
    // for (int i = N - 1; i >= 0; i--) {
    //     for (int j = N - 1; j >= 0; j--) {
    //         new_map[N - 1 - i][N - 1 - j] = cur_map[i][j];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    // // r270
    // for (int i = 0; i < N; i++) {
    //     for (int j = N - 1; j >= 0; j--) {
    //         new_map[N - 1 - j][i] = cur_map[i][j];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    // // x
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         new_map[i][j] = cur_map[i][N - 1 - j];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    // // y
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         new_map[N - 1 - i][j] = cur_map[i][j];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    // // d1
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         new_map[i][j] = cur_map[j][i];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    // // d2
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         new_map[i][j] = cur_map[N - 1 - j][N - 1 - i];
    //     }
    // }
    // if (check_visited(new_map, f)) {
    //     return true;
    // }
    string s = "";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            s += cur_map[i][j] + '0';
        }
    }
    visited[s] = f;
    return false;
}

vector<Operate> SMA_star(float (*heuristic)(bool[][MAX_N])) {
    // float min_lock = 100000;
    // int min_step = 100000;
    // int v_count = 0;

    // initialize the start node
    Node start;
    memcpy(start.map, map, sizeof(map));
    start.f = start.path.size() + heuristic(start.map);
    open.push(start);
    check_visited_symmetry(start.map, start.f);

    while (!open.empty()) {
        Node cur = open.top();
        open.pop();

        float h = heuristic(cur.map);

        // output debug info
        // v_count++;
        // if (h < min_lock || (h == min_lock && cur.path.size() < min_step)) {
        //     min_lock = h;
        //     min_step = cur.path.size();
        //     cout << "min_lock: " << min_lock << endl;
        //     cout << "min_step: " << min_step << endl;
        //     for (int i = 0; i < N; i++) {
        //         for (int j = 0; j < N; j++) {
        //             cout << cur.map[i][j] << " ";
        //         }
        //         cout << endl;
        //     }
        // }

        // if the map is unlocked, return the path
        if (h == 0) {
            // cout << "free_count: " << free_count << endl;
            // cout << "visited_count: " << visited.size() << endl;
            // cout << "v_count: " << v_count << endl;
            return cur.path;
        }

        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < N; j++) {
        //         for (int s = 1; s <= 4; s++) {
        //             if (cur.path.size() > 0) {
        //                 // if the operation is the same as the last one, skip
        //                 if (cur.path.back().i == i && cur.path.back().j == j && cur.path.back().s == s) {
        //                     continue;
        //                 }
        //             }
        //             // if the operation is not available, skip
        //             if (!available(cur.map, i, j, s)) {
        //                 continue;
        //             }
        //             // generate the next node
        //             Node next;
        //             next.path = cur.path;
        //             Operate op(i, j, s);
        //             next.path.push_back(op);
        //             memcpy(next.map, cur.map, sizeof(cur.map));
        //             apply_path(next.map, i, j, s);
        //             next.f = next.path.size() + heuristic(next.map);
        //             // if the next node is visited, skip
        //             if (!check_visited_symmetry(next.map, next.f)) {
        //                 open.push(next);
        //             }
        //         }
        //     }
        // }

        bool find = false;
        for (int i = 0; i < N && !find; i++) {
            for (int j = 0; j < N && !find; j++) {
                if (cur.map[i][j]) {
                    int i_list[12] = {i - 1, i - 1, i, i, i, i, i, i, i, i, i + 1, i + 1};
                    int j_list[12] = {j, j, j - 1, j - 1, j, j, j, j, j + 1, j + 1, j, j};
                    int s_list[12] = {3, 4, 1, 4, 1, 2, 3, 4, 2, 3, 1, 2};
                    for (int k = 0; k < 12; k++) {
                        int i_ = i_list[k];
                        int j_ = j_list[k];
                        int s_ = s_list[k];
                        if (i_ < 0 || i_ >= N || j_ < 0 || j_ >= N) {
                            continue;
                        }
                        if (cur.path.size() > 0) {
                            // if the operation is the same as the last one, skip
                            if (cur.path.back().i == i_ && cur.path.back().j == j_ && cur.path.back().s == s_) {
                                continue;
                            }
                        }
                        // if the operation is not available, skip
                        if (!available(cur.map, i_, j_, s_)) {
                            continue;
                        }
                        // generate the next node
                        Node next;
                        next.path = cur.path;
                        Operate op(i_, j_, s_);
                        next.path.push_back(op);
                        memcpy(next.map, cur.map, sizeof(cur.map));
                        apply_path(next.map, i_, j_, s_);
                        next.f = next.path.size() + heuristic(next.map);
                        // if the next node is visited, skip
                        if (!check_visited_symmetry(next.map, next.f)) {
                            open.push(next);
                        }
                    }
                    find = true;
                }
            }
        }

        // if the open list is too large, free some memory
        if (open.size() > 1e6) {
            priority_queue<Node, vector<Node>, greater<Node>> temp;
            for (int i = 0; i < 1e3; i++) {
                temp.push(open.top());
                open.pop();
            }
            open = temp;
            free_count++;
            // cout << "free";
        }
    }
    return {};
}

int main() {
    string input_file = "AI/astar/input/input";
    string output_file = "AI/astar/output/output";
    string num = "9.txt";
    freopen((input_file + num).c_str(), "r", stdin);
    // freopen((output_file + num).c_str(), "w", stdout);

    cin >> N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cin >> map[i][j];
        }
    }

    time_t start_time = clock();
    vector<Operate> result = SMA_star(heuristic4);
    time_t end_time = clock();
    // cout << "time: " << (end_time - start_time) / 1000.0 << "s" << endl;

    cout << result.size() << endl;
    for (int i = 0; i < result.size(); i++) {
        cout << result[i].i << "," << result[i].j << "," << result[i].s;
        if (i != result.size() - 1) {
            cout << endl;
        }
    }
}