#include <algorithm>
#include <iostream>
#include <time.h>
#include <vector>
using namespace std;

#define MAX_N 200
#define MAX_D 360
#define MAX_S 6

int N, D, S;
bool requirements[MAX_D][MAX_S][MAX_N];
bool shifts_remain[MAX_D][MAX_S][MAX_N];
int shifts[MAX_D][MAX_S];
int times[MAX_N];
int cur_accpet;
bool flag;
time_t start_time, end_time;

void init() {
    char c;
    cin >> N >> c >> D >> c >> S;
    cur_accpet = D * S - 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = 0; k < S; k++) {
                if (k == 0) {
                    cin >> requirements[j][k][i];
                } else {
                    cin >> c >> requirements[j][k][i];
                }
            }
        }
    }
    flag = false;
}

// select the next variable to assign
pair<int, int> mrv(bool shifts_remain[MAX_D][MAX_S][MAX_N]) {
    int min = N + 1;
    int min_i = -1;
    int min_j = -1;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < S; j++) {
            int size = 0;
            for (int k = 0; k < N; k++) {
                if (shifts_remain[i][j][k]) {
                    size++;
                }
            }
            if (size < min && shifts[i][j] == -1) {
                min = size;
                min_i = i;
                min_j = j;
            }
        }
    }
    return {min_i, min_j};
}

void forward_check(bool shifts_remain[MAX_D][MAX_S][MAX_N], int i, int j, int k) {
    // set (i, j) to k
    for (int l = 0; l < N; l++) {
        shifts_remain[i][j][l] = false;
    }
    shifts_remain[i][j][k] = true;
    // remove k from all shifts next to (i, j)
    if (!(i == D - 1 && j == S - 1)) {
        int next_i, next_j;
        if (j == S - 1) {
            next_i = i + 1;
            next_j = 0;
        } else {
            next_i = i;
            next_j = j + 1;
        }
        shifts_remain[next_i][next_j][k] = false;
    }
    if (!(i == 0 && j == 0)) {
        int prev_i, prev_j;
        if (j == 0) {
            prev_i = i - 1;
            prev_j = S - 1;
        } else {
            prev_i = i;
            prev_j = j - 1;
        }
        shifts_remain[prev_i][prev_j][k] = false;
    }
}

bool constraint_propagation(bool shifts_remain[MAX_D][MAX_S][MAX_N]) {
    // check if there is a shift that has no available person
    for (int k = 0; k < N; k++) {
        int count = 0;
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < S; j++) {
                if (shifts_remain[i][j][k]) {
                    count++;
                }
            }
        }
        if (count < D * S / N) {
            return false;
        }
    }
    // check if the possible max accept is less than current accept
    int max_accept = 0;
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < S; j++) {
            for (int k = 0; k < N; k++) {
                if (requirements[i][j][k] && shifts_remain[i][j][k]) {
                    max_accept++;
                    break;
                }
            }
        }
    }
    if (max_accept <= cur_accpet) {
        return false;
    }
    return true;
}

void csp(int N, int D, int S) {
    pair<int, int> mrv_pair = mrv(shifts_remain);
    int i = mrv_pair.first;
    int j = mrv_pair.second;
    if (i == -1 && j == -1) {
        int accept = 0;
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < S; j++) {
                if (requirements[i][j][shifts[i][j]]) {
                    accept++;
                }
            }
        }
        if (accept > cur_accpet) {
            flag = true;
        }
        return;
    }

    // check if the min remaining value is 0
    int size = 0;
    for (int k = 0; k < N; k++) {
        if (shifts_remain[i][j][k]) {
            size++;
        }
    }
    if (size == 0) {
        return;
    }

    // bool shifts_remain_last[MAX_D][MAX_S][MAX_N];
    bool *p = new bool[MAX_D * MAX_S * MAX_N];
    bool(*shifts_remain_last)[MAX_S][MAX_N] = (bool(*)[MAX_S][MAX_N])p;

    memcpy(shifts_remain_last, shifts_remain, sizeof(shifts_remain));
    vector<int> domain;
    for (int k = 0; k < N; k++) {
        if (shifts_remain[i][j][k]) {
            domain.push_back(k);
        }
    }
    // sort the domain by the number of requirements and the number of times
    sort(domain.begin(), domain.end(), [&](int a, int b) -> bool {
        if (max(D * S / N - times[a], 0) != max(D * S / N - times[b], 0)) {
            return max(D * S / N - times[a], 0) > max(D * S / N - times[b], 0);
        } else {
            return requirements[i][j][a] > requirements[i][j][b];
        }
    });
    for (int k : domain) {
        shifts[i][j] = k;
        times[k]++;
        forward_check(shifts_remain, i, j, k);
        if (constraint_propagation(shifts_remain)) {
            csp(N, D, S);
            if (flag) {
                return;
            }
        }
        shifts[i][j] = -1;
        times[k]--;
        memcpy(shifts_remain, shifts_remain_last, sizeof(shifts_remain));
    }
    delete p;
}

int main() {
    string input_file = "AI/csp/input/input";
    string output_file = "AI/csp/output/output";
    string num = "9.txt";
    freopen((input_file + num).c_str(), "r", stdin);
    freopen((output_file + num).c_str(), "w", stdout);

    init();
    start_time = clock();
    while (!flag && cur_accpet >= -1) {
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < S; j++) {
                for (int k = 0; k < N; k++) {
                    shifts_remain[i][j][k] = true;
                }
            }
        }
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < S; j++) {
                shifts[i][j] = -1;
            }
        }
        csp(N, D, S);
        cur_accpet--;
    }
    end_time = clock();
    // cout << "time: " << (end_time - start_time) / 1000.0 << "s" << endl;

    if (!flag) {
        cout << "No valid schedule found." << endl;
    } else {
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < S; j++) {
                if (j == 0) {
                    cout << shifts[i][j] + 1;
                } else {
                    cout << ',' << shifts[i][j] + 1;
                }
            }
            cout << endl;
        }
        int accept = 0;
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < S; j++) {
                if (requirements[i][j][shifts[i][j]]) {
                    accept++;
                }
            }
        }
        cout << accept;
    }
}