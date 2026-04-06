#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

vector<vector<double>> readMatrix(string filename, int &size) {
    ifstream file(filename);
    file >> size;

    vector<vector<double>> matrix(size, vector<double>(size));

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            file >> matrix[i][j];

    return matrix;
}

void writeMatrix(string filename, vector<vector<double>> &matrix) {
    ofstream file(filename);
    int size = matrix.size();

    file << size << endl;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << matrix[i][j] << " ";
        file << endl;
    }
}


vector<vector<double>> multiplyMatrix(vector<vector<double>> &A, vector<vector<double>> &B, int threads) {
    int size = A.size();
    vector<vector<double>> C(size, vector<double>(size, 0));

    omp_set_num_threads(threads);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            double sum = 0;
            for (int k = 0; k < size; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }

    return C;
}

int main() {
    int size1, size2;

    auto A = readMatrix("A.txt", size1);
    auto B = readMatrix("B.txt", size2);

    if (size1 != size2) {
        cout << "Not same size\n";
        return 1;
    }

    int threads;
    cout << "Enter number of threads: ";
    cin >> threads;

    auto start = chrono::high_resolution_clock::now();

    auto C = multiplyMatrix(A, B, threads);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    writeMatrix("result.txt", C);

    long operations = 2L * size1 * size1 * size1;

    cout << "Matrix size: " << size1 << "x" << size1 << endl;
    cout << "Operations: " << operations << endl;
    cout << "Threads: " << threads << endl;
    cout << "Execution time: " << elapsed.count() << " seconds" << endl;

    return 0;
}
