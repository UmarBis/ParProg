#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
using namespace std;

vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> M(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = rand() % 10;
    return M;
}

void writeMatrix(string filename, vector<vector<double>>& M) {
    ofstream file(filename);
    int n = M.size();
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            file << M[i][j] << " ";
        file << endl;
    }
    file.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            cout << "Usage: " << argv[0] << " <matrix_size>" << endl;
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0)
            cout << "Invalid matrix size" << endl;
        MPI_Finalize();
        return 1;
    }

    vector<vector<double>> A, B;
    double start_time, end_time;

    if (rank == 0) {
        srand(time(0));
        A = generateMatrix(n);
        B = generateMatrix(n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> flatA(n * n);
    vector<double> flatB(n * n);

    if (rank == 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                flatA[i * n + j] = A[i][j];
                flatB[i * n + j] = B[i][j];
            }
    }

    MPI_Bcast(flatA.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flatB.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    int rows_per_proc = n / size;
    int remainder = n % size;
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    vector<vector<double>> local_C(local_rows, vector<double>(n, 0.0));

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += flatA[(start_row + i) * n + k] * flatB[k * n + j];
            }
            local_C[i][j] = sum;
        }
    }

    vector<vector<double>> C;

    if (rank == 0) {
        C.resize(n, vector<double>(n));
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < n; j++)
                C[start_row + i][j] = local_C[i][j];

        for (int p = 1; p < size; p++) {
            int p_start = p * rows_per_proc + min(p, remainder);
            int p_rows = rows_per_proc + (p < remainder ? 1 : 0);
            for (int i = 0; i < p_rows; i++) {
                vector<double> row(n);
                MPI_Recv(row.data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int j = 0; j < n; j++)
                    C[p_start + i][j] = row[j];
            }
        }
    } else {
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(local_C[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        writeMatrix("result.txt", C);
        long long operations = 2LL * n * n * n;
        cout << "Matrix size: " << n << "x" << n << endl;
        cout << "Processes: " << size << endl;
        cout << "Time: " << (end_time - start_time) << " sec" << endl;
        cout << "Performance: "
             << (operations / (end_time - start_time) / 1e9)
             << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}
