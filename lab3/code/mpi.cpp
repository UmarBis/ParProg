#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <mpi.h>

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

// Параллельное умножение матриц с использованием MPI
vector<vector<double>> multiplyMatrixMPI(vector<vector<double>> &A, vector<vector<double>> &B, 
                                          int rank, int size, double &parallel_time) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));
    
    // Определяем количество строк для каждого процесса
    int rows_per_proc = n / size;
    int remainder = n % size;
    
    // Определяем диапазон строк для текущего процесса
    int start_row = rank * rows_per_proc + min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_rows = end_row - start_row;
    
    // Локальная матрица для части результата
    vector<vector<double>> local_C(local_rows, vector<double>(n, 0));
    
    auto start = chrono::high_resolution_clock::now();
    
    // Умножение для выделенных строк
    for (int i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[global_i][k] * B[k][j];
            }
            local_C[i][j] = sum;
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> local_elapsed = end - start;
    
    // Сбор результатов на корневом процессе (rank 0)
    if (rank == 0) {
        // Копируем свою часть
        for (int i = 0; i < local_rows; i++) {
            C[start_row + i] = local_C[i];
        }
        
        // Получаем данные от других процессов
        for (int p = 1; p < size; p++) {
            int p_start_row = p * rows_per_proc + min(p, remainder);
            int p_local_rows = rows_per_proc + (p < remainder ? 1 : 0);
            
            vector<vector<double>> recv_C(p_local_rows, vector<double>(n));
            for (int i = 0; i < p_local_rows; i++) {
                MPI_Recv(recv_C[i].data(), n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Вставляем полученные строки в результат
            for (int i = 0; i < p_local_rows; i++) {
                C[p_start_row + i] = recv_C[i];
            }
        }
        
        parallel_time = local_elapsed.count();
        // Находим максимальное время среди всех процессов
        double max_time;
        MPI_Reduce(&parallel_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        parallel_time = max_time;
    } else {
        // Отправляем результаты на корневой процесс
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(local_C[i].data(), n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        
        // Участвуем в reduce для синхронизации
        double dummy;
        MPI_Reduce(&local_elapsed.count(), &dummy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    
    return C;
}

// Функция для генерации тестовых матриц
void generateTestMatrices(int size, const string& filename) {
    ofstream file(filename);
    file << size << endl;
    
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << (double)(rand() % 100) / 10.0 << " ";
        }
        file << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "MPI Matrix Multiplication" << endl;
        cout << "Number of processes: " << size << endl;
        cout << "------------------------" << endl;
    }
    
    int size1, size2;
    vector<vector<double>> A, B, C;
    
    // Чтение матриц (только на корневом процессе)
    if (rank == 0) {
        try {
            A = readMatrix("A.txt", size1);
            B = readMatrix("B.txt", size2);
            
            if (size1 != size2) {
                cout << "Matrix sizes do not match!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        } catch (...) {
            cout << "Error reading matrices. Generating test matrices..." << endl;
            size1 = 200;  // Размер по умолчанию
            generateTestMatrices(size1, "A.txt");
            generateTestMatrices(size1, "B.txt");
            A = readMatrix("A.txt", size1);
            B = readMatrix("B.txt", size2);
        }
    }
    
    // Рассылаем размер матрицы всем процессам
    MPI_Bcast(&size1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Выделяем память на всех процессах
    if (rank != 0) {
        A.resize(size1, vector<double>(size1));
        B.resize(size1, vector<double>(size1));
    }
    
    // Рассылаем матрицы всем процессам
    for (int i = 0; i < size1; i++) {
        MPI_Bcast(A[i].data(), size1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(B[i].data(), size1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);  // Синхронизация перед измерением времени
    
    double parallel_time = 0;
    C = multiplyMatrixMPI(A, B, rank, size, parallel_time);
    
    if (rank == 0) {
        long operations = 2 * size1 * size1 * size1;
        
        cout << "Matrix size: " << size1 << "x" << size1 << endl;
        cout << "Operations: " << operations << endl;
        cout << "Parallel execution time: " << parallel_time << " seconds" << endl;
        
        // Сохраняем результат
        writeMatrix("result_mpi.txt", C);
        
        // Вычисляем производительность
        double gflops = (operations / parallel_time) / 1e9;
        cout << "Performance: " << gflops << " GFLOPS" << endl;
    }
    
    MPI_Finalize();
    return 0;
}
